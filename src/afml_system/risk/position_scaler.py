"""
PRADO9_EVO Module Y — Position Scaling Engine

Converts ensemble outputs into professional-grade exposure through:
- Meta-learner confidence scaling (pyramid winners, shrink losers)
- Bandit exploration → exploitation scaling
- Regime-based aggression adjustments
- Correlation-adjusted allocation
- Safe position capping

This module sits between the allocator output and volatility targeting,
adding intelligent position adjustments based on confidence signals.

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ScalingFactors:
    """
    Breakdown of position scaling components.

    Useful for debugging and understanding why a position was scaled
    to a particular size.
    """
    meta_confidence_factor: float  # Meta-learner confidence scaling
    bandit_factor: float          # Bandit exploration/exploitation
    regime_factor: float          # Regime-based aggression
    correlation_factor: float     # Correlation adjustment
    final_scale: float           # Combined scaling factor
    raw_position: float          # Input position
    scaled_position: float       # Output position


class PositionScaler:
    """
    Professional-grade position scaling engine.

    Scales positions based on multiple confidence signals:
    1. Meta-learner probability (high confidence → larger, low → smaller)
    2. Bandit weight (exploitation → larger, exploration → smaller)
    3. Regime (trending → aggressive, high vol → conservative)
    4. Correlation penalty (diversification adjustment)

    Example:
        >>> scaler = PositionScaler()
        >>> scaled = scaler.scale(
        ...     position=1.0,
        ...     meta_prob=0.7,
        ...     bandit_weight=0.8,
        ...     regime="TRENDING"
        ... )
        >>> print(f"Scaled position: {scaled:.2f}x")
    """

    def __init__(
        self,
        meta_confidence_range: Tuple[float, float] = (0.5, 1.5),
        bandit_min_scale: float = 0.2,
        regime_scales: Optional[Dict[str, float]] = None,
        correlation_penalty_scale: float = 1.0,
        max_position: float = 3.0
    ):
        """
        Initialize Position Scaler.

        Args:
            meta_confidence_range: (min, max) scaling for meta-learner confidence
                                  Default (0.5, 1.5) means:
                                  - 0% confidence → 0.5x position
                                  - 100% confidence → 1.5x position
            bandit_min_scale: Minimum scale during exploration (default 0.2 = 20%)
            regime_scales: Custom regime multipliers (uses defaults if None)
            correlation_penalty_scale: Multiplier for correlation penalty (1.0 = full penalty)
            max_position: Absolute maximum position size (default 3.0x)
        """
        self.meta_confidence_range = meta_confidence_range
        self.bandit_min_scale = bandit_min_scale
        self.max_position = max_position
        self.correlation_penalty_scale = correlation_penalty_scale

        # Default regime scales
        self.regime_scales = regime_scales or {
            'TRENDING': 1.4,        # Aggressive in trends
            'HIGH_VOL': 0.7,        # Conservative in high volatility
            'LOW_VOL': 1.2,         # Moderately aggressive in low vol
            'MEAN_REVERTING': 1.0,  # Neutral in mean reversion
            'NORMAL': 1.0           # Baseline
        }

    def scale(
        self,
        position: float,
        meta_prob: float = 0.5,
        bandit_weight: float = 1.0,
        regime: str = 'NORMAL',
        correlation_penalty: float = 0.0,
        return_factors: bool = False
    ) -> float:
        """
        Scale position based on confidence signals.

        Scaling Pipeline:
        1. Meta-learner confidence: Pyramid winners, shrink losers
        2. Bandit exploration/exploitation: Reduce during exploration
        3. Regime-based aggression: Adapt to market conditions
        4. Correlation penalty: Reduce for correlated positions
        5. Position capping: Ensure safety limits

        Args:
            position: Raw position size from allocator (-3.0 to +3.0)
            meta_prob: Meta-learner probability (0.0 to 1.0)
                      0.5 = neutral, >0.5 = bullish, <0.5 = bearish
            bandit_weight: Bandit algorithm weight (0.0 to 1.0)
                          Higher = more exploitation (confidence)
            regime: Current market regime
            correlation_penalty: Penalty for correlated positions (0.0 to 1.0)
                                Higher = more penalty (reduce position)
            return_factors: If True, return ScalingFactors instead of float

        Returns:
            Scaled position (or ScalingFactors if return_factors=True)

        Example:
            >>> # High confidence trending trade
            >>> scaled = scaler.scale(1.0, meta_prob=0.8, regime='TRENDING')
            >>> # scaled ≈ 1.0 × 1.3 × 1.0 × 1.4 = 1.82x

            >>> # Low confidence exploration trade
            >>> scaled = scaler.scale(1.0, meta_prob=0.3, bandit_weight=0.3)
            >>> # scaled ≈ 1.0 × 0.8 × 0.3 × 1.0 = 0.24x
        """
        raw_position = position

        # 1. Meta-learner confidence scaling
        # Maps probability [0, 1] to scale range (e.g., [0.5, 1.5])
        min_scale, max_scale = self.meta_confidence_range
        meta_confidence_factor = min_scale + (max_scale - min_scale) * meta_prob
        position *= meta_confidence_factor

        # 2. Bandit exploration → exploitation scaling
        # During exploration (low weight), reduce position
        # During exploitation (high weight), full position
        bandit_factor = max(self.bandit_min_scale, bandit_weight)
        position *= bandit_factor

        # 3. Regime-based aggression scaling
        regime_factor = self.regime_scales.get(regime, 1.0)
        position *= regime_factor

        # 4. Correlation penalty adjustment
        # High penalty → reduce position to improve diversification
        correlation_factor = 1.0 - (correlation_penalty * self.correlation_penalty_scale)
        correlation_factor = max(0.2, correlation_factor)  # Floor at 20%
        position *= correlation_factor

        # 5. Cap position at safety limits
        final_scale = position / (raw_position + 1e-10)  # Avoid division by zero
        scaled_position = float(np.clip(position, -self.max_position, self.max_position))

        if return_factors:
            return ScalingFactors(
                meta_confidence_factor=meta_confidence_factor,
                bandit_factor=bandit_factor,
                regime_factor=regime_factor,
                correlation_factor=correlation_factor,
                final_scale=final_scale,
                raw_position=raw_position,
                scaled_position=scaled_position
            )

        return scaled_position

    def pyramid_winners(
        self,
        position: float,
        meta_prob: float,
        current_pnl: float,
        pnl_threshold: float = 0.02
    ) -> float:
        """
        Pyramid winning positions (add to winners, reduce losers).

        Strategy:
        - If position profitable > threshold → increase size
        - If position losing → decrease size
        - Magnitude based on meta-learner confidence

        Args:
            position: Current position size
            meta_prob: Meta-learner confidence
            current_pnl: Current P&L percentage
            pnl_threshold: Threshold for "winning" (default 2%)

        Returns:
            Adjusted position size

        Example:
            >>> # Winning trade → add to position
            >>> new_pos = scaler.pyramid_winners(1.0, meta_prob=0.7, current_pnl=0.05)
            >>> # new_pos > 1.0 (pyramided)

            >>> # Losing trade → reduce position
            >>> new_pos = scaler.pyramid_winners(1.0, meta_prob=0.4, current_pnl=-0.03)
            >>> # new_pos < 1.0 (reduced)
        """
        if current_pnl > pnl_threshold:
            # Winning position → pyramid up
            pyramid_factor = 1.0 + (meta_prob * 0.5)  # Up to 50% increase
        elif current_pnl < -pnl_threshold:
            # Losing position → cut down
            pyramid_factor = 1.0 - ((1.0 - meta_prob) * 0.3)  # Up to 30% decrease
        else:
            # Neutral → no change
            pyramid_factor = 1.0

        adjusted_position = position * pyramid_factor
        return float(np.clip(adjusted_position, -self.max_position, self.max_position))

    def scale_by_regime_volatility(
        self,
        position: float,
        regime: str,
        current_vol: float,
        baseline_vol: float = 0.15
    ) -> float:
        """
        Adjust position based on regime and current volatility.

        Combines regime bias with realized volatility adjustment.

        Args:
            position: Raw position
            regime: Current market regime
            current_vol: Current realized volatility
            baseline_vol: Baseline volatility reference (default 15%)

        Returns:
            Volatility-adjusted position

        Example:
            >>> # High volatility in HIGH_VOL regime → very conservative
            >>> scaled = scaler.scale_by_regime_volatility(
            ...     1.0, regime='HIGH_VOL', current_vol=0.30
            ... )
            >>> # scaled ≈ 0.35x (0.7 regime × 0.5 vol adjustment)
        """
        # Regime scaling
        regime_factor = self.regime_scales.get(regime, 1.0)

        # Volatility adjustment (inverse relationship)
        vol_adjustment = min(baseline_vol / current_vol, 2.0) if current_vol > 0 else 1.0
        vol_adjustment = max(vol_adjustment, 0.2)  # Floor at 20%

        scaled = position * regime_factor * vol_adjustment
        return float(np.clip(scaled, -self.max_position, self.max_position))

    def scale_batch(
        self,
        positions: pd.Series,
        meta_probs: pd.Series,
        bandit_weights: pd.Series,
        regimes: pd.Series,
        correlation_penalties: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Vectorized batch scaling for backtesting efficiency.

        Args:
            positions: Series of raw positions
            meta_probs: Series of meta-learner probabilities
            bandit_weights: Series of bandit weights
            regimes: Series of regime labels
            correlation_penalties: Optional series of correlation penalties

        Returns:
            Series of scaled positions

        Example:
            >>> scaled_positions = scaler.scale_batch(
            ...     df['raw_position'],
            ...     df['meta_prob'],
            ...     df['bandit_weight'],
            ...     df['regime']
            ... )
        """
        if correlation_penalties is None:
            correlation_penalties = pd.Series(0.0, index=positions.index)

        # Vectorized scaling
        scaled = []
        for pos, meta_p, bandit_w, regime, corr_pen in zip(
            positions, meta_probs, bandit_weights, regimes, correlation_penalties
        ):
            scaled.append(self.scale(pos, meta_p, bandit_w, regime, corr_pen))

        return pd.Series(scaled, index=positions.index)

    def get_regime_scales(self) -> Dict[str, float]:
        """Get current regime scaling configuration."""
        return self.regime_scales.copy()

    def update_regime_scale(self, regime: str, scale: float):
        """
        Update regime scaling factor.

        Args:
            regime: Regime name
            scale: New scaling factor
        """
        self.regime_scales[regime] = scale

    def __repr__(self) -> str:
        """String representation of PositionScaler configuration."""
        return (
            f"PositionScaler("
            f"meta_range={self.meta_confidence_range}, "
            f"bandit_min={self.bandit_min_scale}, "
            f"max_pos={self.max_position}x)"
        )
