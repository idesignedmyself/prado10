"""
PRADO9_EVO Strategy Registry

Centralized registry for all trading strategies with adapters to unify
different strategy signatures into a common interface.

Architecture:
- Volatility strategies (Module V): 5 strategies
- Breakout strategies (Module B2): 4 strategies
- Base strategies (signal_engine): 2 strategies
- Total: 11 strategies

All strategies are adapted to return StrategyResult objects compatible
with LiveSignalEngine.
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable

# Import strategy classes
from ..volatility.vol_strategies import VolatilityStrategies, VolatilitySignal
from ..trend.breakout_strategies import BreakoutStrategies, BreakoutSignal
from ..live.signal_engine import StrategyResult


# ============================================================================
# STRATEGY ADAPTERS
# ============================================================================

class StrategyAdapter:
    """
    Adapts different strategy signatures to unified StrategyResult format.

    Handles conversion from:
    - Class methods (VolatilityStrategies, BreakoutStrategies)
    - Dict-based features to DataFrame-based features
    - VolatilitySignal/BreakoutSignal to StrategyResult
    """

    def __init__(self):
        self.vol_strategies = VolatilityStrategies()
        self.breakout_strategies = BreakoutStrategies()

    def _df_to_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Convert DataFrame to feature dictionary.

        Args:
            df: OHLCV DataFrame with features

        Returns:
            Dict of feature name -> value
        """
        if df is None or df.empty:
            return {
                'volatility': 0.015,
                'momentum': 0.0,
                'close': 100.0,
                'prev_close': 100.0,
            }

        latest = df.iloc[-1]
        features = {}

        # Extract available features
        for col in df.columns:
            try:
                features[col.lower()] = float(latest[col])
            except (ValueError, TypeError):
                pass

        # Ensure required features exist with defaults
        features.setdefault('volatility', 0.015)
        features.setdefault('momentum', 0.0)
        features.setdefault('close', 100.0)
        features.setdefault('prev_close', features.get('close', 100.0))

        # Compute derived features if missing
        if 'donchian_high' not in features:
            features['donchian_high'] = features['close'] * 1.02
        if 'donchian_low' not in features:
            features['donchian_low'] = features['close'] * 0.98
        if 'range_upper' not in features:
            features['range_upper'] = features['close'] * 1.015
        if 'range_lower' not in features:
            features['range_lower'] = features['close'] * 0.985
        if 'atr' not in features:
            features['atr'] = features['close'] * 0.02
        if 'momentum_change' not in features:
            features['momentum_change'] = 0.0

        return features

    def _detect_regime(self, df: pd.DataFrame) -> str:
        """
        Simple regime detection from DataFrame.

        Args:
            df: DataFrame with features

        Returns:
            Regime string
        """
        if df is None or df.empty:
            return 'NORMAL'

        latest = df.iloc[-1]
        volatility = latest.get('volatility', 0.015)

        if volatility > 0.03:
            return 'HIGH_VOL'
        elif volatility < 0.01:
            return 'LOW_VOL'
        else:
            return 'NORMAL'

    def _signal_to_result(
        self,
        signal,
        strategy_name: str
    ) -> StrategyResult:
        """
        Convert VolatilitySignal or BreakoutSignal to StrategyResult.

        Args:
            signal: VolatilitySignal or BreakoutSignal object
            strategy_name: Strategy name override

        Returns:
            StrategyResult compatible with LiveSignalEngine
        """
        return StrategyResult(
            strategy_name=strategy_name,
            regime=signal.regime,
            horizon=signal.horizon,
            side=signal.side,
            probability=signal.probability,
            forecast_return=signal.forecast_return,
            volatility_forecast=signal.volatility_forecast,
            metadata={
                'meta_probability': signal.meta_probability,
                'bandit_weight': signal.bandit_weight,
                'uniqueness': signal.uniqueness,
                'correlation_penalty': signal.correlation_penalty,
            }
        )

    # ========================================================================
    # VOLATILITY STRATEGY ADAPTERS
    # ========================================================================

    def vol_breakout(self, df: pd.DataFrame) -> StrategyResult:
        """Volatility Breakout - Trades volatility expansions"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.vol_strategies.vol_breakout(features, regime, '5d')
        return self._signal_to_result(signal, 'vol_breakout')

    def vol_spike_fade(self, df: pd.DataFrame) -> StrategyResult:
        """Volatility Spike Fade - Fades extreme volatility spikes"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.vol_strategies.vol_spike_fade(features, regime, '5d')
        return self._signal_to_result(signal, 'vol_spike_fade')

    def vol_compression(self, df: pd.DataFrame) -> StrategyResult:
        """Volatility Compression - Anticipates breakout after compression"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.vol_strategies.vol_compression(features, regime, '5d')
        return self._signal_to_result(signal, 'vol_compression')

    def vol_mean_revert(self, df: pd.DataFrame) -> StrategyResult:
        """Volatility Mean Reversion - Volatility returns to average"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.vol_strategies.vol_mean_revert(features, regime, '5d')
        return self._signal_to_result(signal, 'vol_mean_revert')

    def trend_breakout_vol(self, df: pd.DataFrame) -> StrategyResult:
        """Trend Breakout (from vol module) - Breakouts from consolidation"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.vol_strategies.trend_breakout(features, regime, '5d')
        return self._signal_to_result(signal, 'trend_breakout')

    # ========================================================================
    # BREAKOUT STRATEGY ADAPTERS
    # ========================================================================

    def donchian_breakout(self, df: pd.DataFrame) -> StrategyResult:
        """Donchian Channel Breakout - Classic turtle trader"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.breakout_strategies.donchian_breakout(features, regime, '5d')
        return self._signal_to_result(signal, 'donchian_breakout')

    def range_breakout(self, df: pd.DataFrame) -> StrategyResult:
        """Range Breakout - Consolidation range breakouts"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.breakout_strategies.range_breakout(features, regime, '5d')
        return self._signal_to_result(signal, 'range_breakout')

    def atr_breakout(self, df: pd.DataFrame) -> StrategyResult:
        """ATR Breakout - ATR-based volatility breakouts"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.breakout_strategies.atr_breakout(features, regime, '5d')
        return self._signal_to_result(signal, 'atr_breakout')

    def momentum_surge(self, df: pd.DataFrame) -> StrategyResult:
        """Momentum Surge - Momentum acceleration detection"""
        features = self._df_to_features(df)
        regime = self._detect_regime(df)
        signal = self.breakout_strategies.momentum_surge(features, regime, '5d')
        return self._signal_to_result(signal, 'momentum_surge')


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

def build_strategy_registry() -> Dict[str, Callable]:
    """
    Build the complete strategy registry.

    Returns:
        Dict mapping strategy_name -> strategy_function
    """
    # Import base strategies from signal_engine
    from ..live.signal_engine import momentum_strategy, mean_reversion_strategy

    # Create adapter instance
    adapter = StrategyAdapter()

    # Build registry
    registry = {
        # Base strategies (2)
        'momentum': momentum_strategy,
        'mean_reversion': mean_reversion_strategy,

        # Volatility strategies (5)
        'vol_breakout': adapter.vol_breakout,
        'vol_spike_fade': adapter.vol_spike_fade,
        'vol_compression': adapter.vol_compression,
        'vol_mean_revert': adapter.vol_mean_revert,
        'trend_breakout': adapter.trend_breakout_vol,

        # Breakout strategies (4)
        'donchian_breakout': adapter.donchian_breakout,
        'range_breakout': adapter.range_breakout,
        'atr_breakout': adapter.atr_breakout,
        'momentum_surge': adapter.momentum_surge,
    }

    return registry


# Export registry builder
STRATEGY_REGISTRY = build_strategy_registry()

__all__ = [
    'STRATEGY_REGISTRY',
    'build_strategy_registry',
    'StrategyAdapter',
]
