"""
PRADO9_EVO Unified Adaptive Engine

Integrates all evolutionary modules:
- Module AR: Adaptive Retraining Engine
- Module X2: Forward-Looking Volatility Engine
- Module Y2: Adaptive Confidence Scaling
- Module MC2: Monte Carlo Robustness Engine
- Module CR2: Enhanced Crisis Detection

This engine provides a single interface for backtesting with all
adaptive components enabled.

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Import all adaptive modules
try:
    from ..features.adaptive_retraining import AdaptiveRetrainingEngine
except ImportError:
    AdaptiveRetrainingEngine = None

try:
    from ..features.forward_vol_engine import ForwardVolatilityEngine
except ImportError:
    ForwardVolatilityEngine = None

try:
    from ..features.adaptive_confidence import AdaptiveConfidenceScaler
except ImportError:
    AdaptiveConfidenceScaler = None

try:
    from ..backtest.monte_carlo_mc2 import MC2Engine
except ImportError:
    MC2Engine = None

try:
    from ..backtest.crisis_stress_cr2 import (
        MultiCrisisDetector,
        SyntheticCrisisGenerator,
        CrisisType
    )
except ImportError:
    MultiCrisisDetector = None
    SyntheticCrisisGenerator = None
    CrisisType = None


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class UnifiedAdaptiveConfig:
    """Configuration for Unified Adaptive Engine."""

    # Module AR: Adaptive Retraining
    enable_adaptive_retraining: bool = True
    ar_regime_threshold: float = 2.5
    ar_min_retrain_gap: int = 20
    ar_force_retrain_period: int = 252

    # Module X2: Forward-Looking Volatility
    enable_forward_vol: bool = True
    x2_forward_window: int = 20
    x2_ema_span: int = 10
    x2_vol_floor: float = 0.05
    x2_vol_cap: float = 1.0

    # Module Y2: Adaptive Confidence
    enable_adaptive_confidence: bool = True
    y2_base_confidence: float = 0.5
    y2_regime_weight: float = 0.3
    y2_vol_weight: float = 0.2
    y2_momentum_weight: float = 0.2
    y2_drawdown_weight: float = 0.3

    # Module MC2: Monte Carlo Robustness
    enable_mc2_validation: bool = False  # Optional (expensive)
    mc2_n_simulations: int = 1000
    mc2_block_size: int = 20

    # Module CR2: Crisis Detection
    enable_crisis_detection: bool = True
    cr2_vol_threshold: float = 2.0
    cr2_min_crisis_duration: int = 20

    # General
    random_seed: int = 42


# ============================================================================
# UNIFIED ADAPTIVE ENGINE
# ============================================================================

class UnifiedAdaptiveEngine:
    """
    Unified Adaptive Engine integrating AR, X2, Y2, MC2, CR2.

    This engine orchestrates all adaptive components to provide:
    1. Adaptive retraining based on regime changes (AR)
    2. Forward-looking volatility forecasts (X2)
    3. Adaptive confidence scaling (Y2)
    4. Monte Carlo robustness validation (MC2)
    5. Crisis detection and stress testing (CR2)

    Example:
        >>> config = UnifiedAdaptiveConfig(enable_mc2_validation=False)
        >>> engine = UnifiedAdaptiveEngine(config=config)
        >>> result = engine.run_adaptive_backtest(symbol='QQQ', df=data)
    """

    def __init__(self, config: Optional[UnifiedAdaptiveConfig] = None):
        """
        Initialize Unified Adaptive Engine.

        Args:
            config: Configuration for all modules (optional)
        """
        self.config = config or UnifiedAdaptiveConfig()

        # Initialize module components
        self._init_modules()

        # Tracking
        self.retraining_history: List[Dict] = []
        self.crisis_periods: List[Dict] = []
        self.confidence_history: List[float] = []

    def _init_modules(self):
        """Initialize all adaptive module components."""

        # Module AR: Adaptive Retraining
        if self.config.enable_adaptive_retraining and AdaptiveRetrainingEngine is not None:
            self.ar_engine = AdaptiveRetrainingEngine(
                regime_threshold=self.config.ar_regime_threshold,
                min_retrain_gap=self.config.ar_min_retrain_gap,
                force_retrain_period=self.config.ar_force_retrain_period
            )
        else:
            self.ar_engine = None

        # Module X2: Forward-Looking Volatility
        if self.config.enable_forward_vol and ForwardVolatilityEngine is not None:
            self.x2_engine = ForwardVolatilityEngine(
                forward_window=self.config.x2_forward_window,
                ema_span=self.config.x2_ema_span,
                vol_floor=self.config.x2_vol_floor,
                vol_cap=self.config.x2_vol_cap
            )
        else:
            self.x2_engine = None

        # Module Y2: Adaptive Confidence
        if self.config.enable_adaptive_confidence and AdaptiveConfidenceScaler is not None:
            self.y2_engine = AdaptiveConfidenceScaler(
                base_confidence=self.config.y2_base_confidence,
                regime_weight=self.config.y2_regime_weight,
                vol_weight=self.config.y2_vol_weight,
                momentum_weight=self.config.y2_momentum_weight,
                drawdown_weight=self.config.y2_drawdown_weight
            )
        else:
            self.y2_engine = None

        # Module MC2: Monte Carlo Robustness
        if self.config.enable_mc2_validation and MC2Engine is not None:
            self.mc2_engine = MC2Engine(seed=self.config.random_seed)
        else:
            self.mc2_engine = None

        # Module CR2: Crisis Detection
        if self.config.enable_crisis_detection and MultiCrisisDetector is not None:
            self.cr2_detector = MultiCrisisDetector(
                vol_threshold_multiplier=self.config.cr2_vol_threshold,
                min_crisis_duration=self.config.cr2_min_crisis_duration
            )
        else:
            self.cr2_detector = None

    def run_adaptive_backtest(
        self,
        symbol: str,
        df: pd.DataFrame,
        strategy_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run backtest with all adaptive components enabled.

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            strategy_func: Strategy function (optional, uses default if None)

        Returns:
            Comprehensive results dict with all module outputs
        """
        results = {
            'symbol': symbol,
            'start_date': str(df.index[0]),
            'end_date': str(df.index[-1]),
            'bars': len(df),
            'modules_enabled': self._get_enabled_modules(),
        }

        # 1. Module CR2: Detect crisis periods first
        if self.cr2_detector is not None:
            crisis_periods = self.cr2_detector.detect_crises(df)
            results['crisis_detection'] = {
                'num_crises': len(crisis_periods),
                'crises': [
                    {
                        'name': c.name,
                        'type': c.crisis_type.value,
                        'start_date': str(c.start_date),
                        'end_date': str(c.end_date),
                        'duration_days': c.duration_days,
                        'max_drawdown': c.max_drawdown,
                        'vol_multiplier': c.vol_multiplier,
                        'confidence': c.match_confidence
                    }
                    for c in crisis_periods
                ]
            }
            self.crisis_periods = crisis_periods

        # 2. Module X2: Compute forward-looking volatility
        if self.x2_engine is not None:
            forward_vol = self.x2_engine.compute_forward_volatility(df)
            results['forward_volatility'] = {
                'mean': float(forward_vol.mean()),
                'std': float(forward_vol.std()),
                'min': float(forward_vol.min()),
                'max': float(forward_vol.max()),
                'current': float(forward_vol.iloc[-1])
            }
        else:
            forward_vol = None

        # 3. Module AR: Determine retraining points
        if self.ar_engine is not None:
            retrain_points = self.ar_engine.detect_regime_changes(df)
            results['adaptive_retraining'] = {
                'num_retrains': len(retrain_points),
                'retrain_dates': [str(df.index[idx]) for idx in retrain_points],
                'avg_gap_days': int(np.mean(np.diff(retrain_points))) if len(retrain_points) > 1 else 0
            }
            self.retraining_history = retrain_points
        else:
            retrain_points = []

        # 4. Module Y2: Compute adaptive confidence scores
        if self.y2_engine is not None:
            confidence_scores = self.y2_engine.compute_adaptive_confidence(
                df=df,
                crisis_periods=self.crisis_periods if self.cr2_detector else None,
                forward_vol=forward_vol
            )
            results['adaptive_confidence'] = {
                'mean': float(confidence_scores.mean()),
                'std': float(confidence_scores.std()),
                'min': float(confidence_scores.min()),
                'max': float(confidence_scores.max()),
                'current': float(confidence_scores.iloc[-1])
            }
            self.confidence_history = confidence_scores
        else:
            confidence_scores = None

        # 5. Run backtest with adaptive components
        backtest_result = self._run_backtest_with_adaptations(
            df=df,
            forward_vol=forward_vol,
            confidence_scores=confidence_scores,
            retrain_points=retrain_points,
            strategy_func=strategy_func
        )

        results['backtest'] = backtest_result

        # 6. Module MC2: Optional robustness validation
        if self.mc2_engine is not None and self.config.enable_mc2_validation:
            mc2_result = self._run_mc2_validation(df, backtest_result)
            results['mc2_validation'] = mc2_result

        return results

    def _run_backtest_with_adaptations(
        self,
        df: pd.DataFrame,
        forward_vol: Optional[pd.Series],
        confidence_scores: Optional[pd.Series],
        retrain_points: List[int],
        strategy_func: Optional[callable]
    ) -> Dict[str, Any]:
        """
        Run backtest incorporating all adaptive components.

        This is a simplified implementation. In production, this would:
        1. Use forward_vol for position sizing (vol targeting)
        2. Use confidence_scores for signal filtering
        3. Use retrain_points to retrain model parameters
        4. Apply crisis-aware risk management
        """
        # Simplified backtest (placeholder)
        # In production, this would call the actual strategy with adaptations

        returns = df['close'].pct_change().dropna()

        # Apply confidence-based position sizing if available
        if confidence_scores is not None:
            # Scale positions by confidence
            positions = confidence_scores[1:].values  # Align with returns
        else:
            positions = np.ones(len(returns))

        # Apply forward-vol based risk management if available
        if forward_vol is not None:
            # Target volatility: scale positions inversely to forecast vol
            target_vol = 0.15  # 15% annualized
            realized_vol = forward_vol[1:].values
            vol_scale = np.minimum(target_vol / (realized_vol + 1e-6), 2.0)
            positions = positions * vol_scale

        # Compute strategy returns
        strategy_returns = returns.values * positions[:len(returns)]

        # Compute metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = strategy_returns.mean() / (strategy_returns.std() + 1e-6) * np.sqrt(252)

        cumulative = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'num_bars': len(strategy_returns),
            'adaptations_applied': {
                'forward_vol_sizing': forward_vol is not None,
                'confidence_scaling': confidence_scores is not None,
                'regime_retraining': len(retrain_points) > 0
            }
        }

    def _run_mc2_validation(
        self,
        df: pd.DataFrame,
        backtest_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run MC2 robustness validation."""
        returns = df['close'].pct_change().dropna()

        # Block bootstrap validation
        bootstrap_result = self.mc2_engine.block_bootstrap.run(
            returns=returns,
            block_size=self.config.mc2_block_size,
            n_sim=self.config.mc2_n_simulations
        )

        return {
            'actual_sharpe': bootstrap_result.actual_sharpe,
            'mc_sharpe_mean': bootstrap_result.mc_sharpe_mean,
            'skill_percentile': bootstrap_result.skill_percentile,
            'p_value': bootstrap_result.p_value,
            'significant': bootstrap_result.significant
        }

    def _get_enabled_modules(self) -> List[str]:
        """Get list of enabled modules."""
        modules = []
        if self.ar_engine is not None:
            modules.append('AR')
        if self.x2_engine is not None:
            modules.append('X2')
        if self.y2_engine is not None:
            modules.append('Y2')
        if self.mc2_engine is not None:
            modules.append('MC2')
        if self.cr2_detector is not None:
            modules.append('CR2')
        return modules

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of engine state and history."""
        return {
            'config': {
                'ar_enabled': self.ar_engine is not None,
                'x2_enabled': self.x2_engine is not None,
                'y2_enabled': self.y2_engine is not None,
                'mc2_enabled': self.mc2_engine is not None,
                'cr2_enabled': self.cr2_detector is not None,
            },
            'history': {
                'num_retrainings': len(self.retraining_history),
                'num_crises_detected': len(self.crisis_periods),
                'confidence_tracking': len(self.confidence_history) > 0
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_unified_backtest(
    symbol: str,
    df: pd.DataFrame,
    enable_all_modules: bool = True,
    enable_mc2: bool = False,
    config: Optional[UnifiedAdaptiveConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to run unified adaptive backtest.

    Args:
        symbol: Trading symbol
        df: OHLCV DataFrame
        enable_all_modules: Enable AR/X2/Y2/CR2 (default: True)
        enable_mc2: Enable MC2 validation (expensive, default: False)
        config: Custom configuration (optional)

    Returns:
        Comprehensive results dict

    Example:
        >>> result = run_unified_backtest('QQQ', df)
        >>> print(f"Sharpe: {result['backtest']['sharpe_ratio']:.2f}")
        >>> print(f"Modules: {result['modules_enabled']}")
    """
    if config is None:
        config = UnifiedAdaptiveConfig(
            enable_adaptive_retraining=enable_all_modules,
            enable_forward_vol=enable_all_modules,
            enable_adaptive_confidence=enable_all_modules,
            enable_crisis_detection=enable_all_modules,
            enable_mc2_validation=enable_mc2
        )

    engine = UnifiedAdaptiveEngine(config=config)
    return engine.run_adaptive_backtest(symbol=symbol, df=df)


# ============================================================================
# VERSION INFO
# ============================================================================

UNIFIED_ENGINE_VERSION = '1.0.0'
INTEGRATED_MODULES = ['AR', 'X2', 'Y2', 'MC2', 'CR2']

def get_version_info() -> Dict[str, Any]:
    """Get version information for unified engine."""
    return {
        'version': UNIFIED_ENGINE_VERSION,
        'modules': INTEGRATED_MODULES,
        'build_date': '2025-01-18',
        'components': {
            'AR': 'Adaptive Retraining Engine',
            'X2': 'Forward-Looking Volatility Engine',
            'Y2': 'Adaptive Confidence Scaling',
            'MC2': 'Monte Carlo Robustness Engine',
            'CR2': 'Enhanced Crisis Detection'
        }
    }
