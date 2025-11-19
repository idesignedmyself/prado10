"""
PRADO9_EVO Module Y2 — Adaptive Confidence Scaling

Enhances Module Y with adaptive confidence threshold determination
based on regime-specific signal quality and historical performance.

This module dynamically adjusts confidence thresholds to:
- Increase aggression in favorable regimes
- Reduce exposure in uncertain regimes
- Adapt to changing market conditions

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceThresholds:
    """Regime-specific confidence thresholds."""
    regime: str
    min_confidence: float  # Minimum confidence to take position
    max_confidence: float  # Maximum confidence (scaling ceiling)
    scale_range: Tuple[float, float]  # (min_scale, max_scale) for position sizing


class AdaptiveConfidence:
    """
    Adaptive confidence threshold determination based on regime and performance.

    Analyzes historical signal quality in different regimes to set optimal
    confidence thresholds for position sizing.

    Example:
        >>> ac = AdaptiveConfidence()
        >>> ac.fit(train_df)
        >>> thresholds = ac.determine_threshold("HIGH_VOL")
        >>> print(f"HIGH_VOL: min={thresholds.min_confidence:.2f}")
    """

    def __init__(
        self,
        default_min_confidence: float = 0.3,
        default_max_confidence: float = 0.9,
        default_scale_range: Tuple[float, float] = (0.5, 1.5)
    ):
        """
        Initialize Adaptive Confidence.

        Args:
            default_min_confidence: Default minimum confidence threshold
            default_max_confidence: Default maximum confidence threshold
            default_scale_range: Default (min_scale, max_scale) range
        """
        self.default_min_confidence = default_min_confidence
        self.default_max_confidence = default_max_confidence
        self.default_scale_range = default_scale_range

        # Regime-specific thresholds (learned from data)
        self.regime_thresholds: Dict[str, ConfidenceThresholds] = {}

        # Performance tracking per regime
        self.regime_performance: Dict[str, Dict[str, float]] = {}

    def relax_threshold(self, past_trades: int = 0) -> float:
        """
        PATCH 2: Dynamically relax threshold based on trading activity.

        If no trades occurred → relax threshold aggressively.
        This prevents adaptive mode from suffocating signals.

        Args:
            past_trades: Number of trades in previous window

        Returns:
            Relaxed threshold value
        """
        # Base threshold (from config or meta-learner)
        th = self.default_min_confidence

        # If no trades → relax threshold aggressively
        if past_trades == 0:
            return 0.20

        # Mild relaxation every window
        return max(0.30, th - 0.15)

    def determine_threshold(self, regime: str) -> ConfidenceThresholds:
        """
        Determine confidence thresholds for a specific regime.

        If regime has been observed in training data, uses learned thresholds.
        Otherwise, uses regime-specific defaults based on volatility/trend characteristics.

        Args:
            regime: Market regime (HIGH_VOL, LOW_VOL, TRENDING, etc.)

        Returns:
            ConfidenceThresholds for the regime
        """
        # If we've learned thresholds for this regime, use them
        if regime in self.regime_thresholds:
            return self.regime_thresholds[regime]

        # Otherwise, use regime-specific defaults
        return self._get_default_thresholds(regime)

    def _get_default_thresholds(self, regime: str) -> ConfidenceThresholds:
        """
        Get default confidence thresholds for a regime.

        Different regimes have different signal quality characteristics:
        - HIGH_VOL: Lower confidence required (signals are noisier)
        - LOW_VOL: Higher confidence required (fewer opportunities)
        - TRENDING: Moderate confidence, wider scaling range
        - MEAN_REVERTING: Higher confidence required
        - NORMAL: Balanced approach

        Args:
            regime: Market regime

        Returns:
            ConfidenceThresholds with regime-specific defaults
        """
        # Regime-specific default parameters
        regime_defaults = {
            "HIGH_VOL": {
                "min_confidence": 0.25,  # Lower bar (more signals)
                "max_confidence": 0.85,  # Conservative ceiling
                "scale_range": (0.4, 1.3)  # Narrower range (volatility risk)
            },
            "LOW_VOL": {
                "min_confidence": 0.35,  # Higher bar (quality over quantity)
                "max_confidence": 0.95,  # High ceiling (stable environment)
                "scale_range": (0.6, 1.8)  # Wider range (low risk)
            },
            "TRENDING": {
                "min_confidence": 0.30,  # Moderate bar
                "max_confidence": 0.90,  # Standard ceiling
                "scale_range": (0.5, 1.6)  # Wider range (momentum)
            },
            "MEAN_REVERTING": {
                "min_confidence": 0.40,  # Higher bar (reversals risky)
                "max_confidence": 0.90,  # Standard ceiling
                "scale_range": (0.5, 1.4)  # Moderate range
            },
            "NORMAL": {
                "min_confidence": self.default_min_confidence,
                "max_confidence": self.default_max_confidence,
                "scale_range": self.default_scale_range
            }
        }

        # Get defaults for this regime (or fall back to NORMAL)
        params = regime_defaults.get(regime, regime_defaults["NORMAL"])

        return ConfidenceThresholds(
            regime=regime,
            min_confidence=params["min_confidence"],
            max_confidence=params["max_confidence"],
            scale_range=params["scale_range"]
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Learn optimal confidence thresholds from training data.

        Analyzes:
        - Signal quality by regime
        - Historical win rates by confidence level
        - Return distributions by regime

        Args:
            train_df: Training DataFrame with columns:
                     - 'close': Close prices
                     - 'returns': Returns (optional, will compute if missing)
                     - 'regime': Market regime (optional)
        """
        # Ensure we have returns
        if 'returns' not in train_df.columns:
            train_df = train_df.copy()
            train_df['returns'] = train_df['close'].pct_change()

        # Calculate signal quality metrics
        signal_quality = self._calculate_signal_quality(train_df)

        # Determine regime if not provided
        if 'regime' not in train_df.columns:
            train_df = train_df.copy()
            train_df['regime'] = self._infer_regime(train_df)

        # Analyze performance by regime
        for regime in train_df['regime'].unique():
            if pd.isna(regime):
                continue

            regime_data = train_df[train_df['regime'] == regime]

            # Calculate regime-specific metrics
            regime_metrics = self._analyze_regime_performance(regime_data)

            # Store performance metrics
            self.regime_performance[regime] = regime_metrics

            # Determine optimal thresholds based on performance
            thresholds = self._optimize_thresholds(regime, regime_metrics)

            self.regime_thresholds[regime] = thresholds

    def _calculate_signal_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall signal quality metrics from data.

        Args:
            df: DataFrame with returns

        Returns:
            Dictionary of signal quality metrics
        """
        returns = df['returns'].dropna()

        if len(returns) < 2:
            return {
                'sharpe': 0.0,
                'win_rate': 0.5,
                'avg_return': 0.0,
                'volatility': 0.15
            }

        # Calculate metrics
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.5
        avg_return = returns.mean()
        volatility = returns.std() * np.sqrt(252)

        return {
            'sharpe': float(sharpe),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'volatility': float(volatility)
        }

    def _infer_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Infer market regime from price data.

        Simple regime detection based on:
        - Volatility (rolling std)
        - Trend (rolling mean)

        Args:
            df: DataFrame with 'close' and 'returns'

        Returns:
            Series of regime labels
        """
        returns = df['returns'].fillna(0)

        # Calculate rolling volatility (20-day window)
        rolling_vol = returns.rolling(20, min_periods=5).std()

        # Calculate rolling trend (momentum)
        rolling_trend = returns.rolling(20, min_periods=5).mean()

        # Simple regime classification
        regimes = pd.Series(['NORMAL'] * len(df), index=df.index)

        # HIGH_VOL: top 25% volatility
        high_vol_threshold = rolling_vol.quantile(0.75)
        regimes[rolling_vol > high_vol_threshold] = 'HIGH_VOL'

        # LOW_VOL: bottom 25% volatility
        low_vol_threshold = rolling_vol.quantile(0.25)
        regimes[rolling_vol < low_vol_threshold] = 'LOW_VOL'

        # TRENDING: strong positive or negative trend
        regimes[rolling_trend > 0.001] = 'TRENDING'
        regimes[rolling_trend < -0.001] = 'MEAN_REVERTING'

        return regimes

    def _analyze_regime_performance(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze performance metrics for a specific regime.

        Args:
            regime_data: DataFrame filtered to one regime

        Returns:
            Dictionary of performance metrics
        """
        if len(regime_data) < 2:
            return {
                'sharpe': 0.0,
                'win_rate': 0.5,
                'avg_return': 0.0,
                'volatility': 0.15,
                'samples': 0
            }

        returns = regime_data['returns'].dropna()

        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.5
        avg_return = returns.mean()
        volatility = returns.std() * np.sqrt(252)

        return {
            'sharpe': float(sharpe),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'volatility': float(volatility),
            'samples': len(regime_data)
        }

    def _optimize_thresholds(
        self,
        regime: str,
        metrics: Dict[str, float]
    ) -> ConfidenceThresholds:
        """
        Optimize confidence thresholds based on regime performance.

        Higher Sharpe/win_rate → Lower min threshold (take more trades)
        Higher volatility → Narrower scale range (reduce risk)

        Args:
            regime: Market regime
            metrics: Performance metrics for the regime

        Returns:
            Optimized ConfidenceThresholds
        """
        # Get default thresholds as baseline
        defaults = self._get_default_thresholds(regime)

        # Adjust based on Sharpe ratio
        sharpe = metrics.get('sharpe', 0.0)
        if sharpe > 1.0:
            # Good performance → lower bar, wider range
            min_conf = max(0.2, defaults.min_confidence - 0.05)
            max_conf = min(0.95, defaults.max_confidence + 0.05)
            scale_min = max(0.3, defaults.scale_range[0] - 0.1)
            scale_max = min(2.0, defaults.scale_range[1] + 0.2)
        elif sharpe < 0.0:
            # Poor performance → higher bar, narrower range
            min_conf = min(0.5, defaults.min_confidence + 0.1)
            max_conf = max(0.7, defaults.max_confidence - 0.1)
            scale_min = min(0.7, defaults.scale_range[0] + 0.1)
            scale_max = max(1.2, defaults.scale_range[1] - 0.2)
        else:
            # Moderate performance → use defaults
            min_conf = defaults.min_confidence
            max_conf = defaults.max_confidence
            scale_min = defaults.scale_range[0]
            scale_max = defaults.scale_range[1]

        # Adjust for volatility
        volatility = metrics.get('volatility', 0.15)
        if volatility > 0.30:  # High volatility
            # Narrow the scale range
            scale_max = min(scale_max, 1.5)
        elif volatility < 0.10:  # Low volatility
            # Widen the scale range
            scale_max = min(scale_max + 0.3, 2.0)

        return ConfidenceThresholds(
            regime=regime,
            min_confidence=min_conf,
            max_confidence=max_conf,
            scale_range=(scale_min, scale_max)
        )

    def get_regime_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics for all observed regimes.

        Returns:
            Dictionary mapping regime to performance metrics
        """
        return self.regime_performance.copy()

    def __repr__(self) -> str:
        """String representation."""
        n_regimes = len(self.regime_thresholds)
        return f"AdaptiveConfidence(regimes={n_regimes}, fitted={n_regimes > 0})"
