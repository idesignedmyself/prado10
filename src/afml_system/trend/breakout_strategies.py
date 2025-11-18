"""
PRADO9_EVO Module B2 — Trend Breakout Engine

Dedicated breakout strategies optimized for TRENDING and HIGH_VOL regimes.

These strategies are distinct from momentum and focus on identifying
breakouts from consolidation ranges, which are fundamental patterns
in quantitative portfolios.

Strategies:
1. donchian_breakout - Donchian channel breakouts (classic turtle trader)
2. range_breakout - Range/consolidation breakouts
3. atr_breakout - ATR-based volatility breakouts (future)

Active Regimes:
- TRENDING: Strong directional moves
- HIGH_VOL: Volatility expansion environments
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class BreakoutSignal:
    """
    Signal dataclass for breakout strategies.

    Compatible with StrategySignal from Module G (Allocator).
    """
    strategy_name: str
    regime: str
    horizon: str
    side: int  # 1 = long, -1 = short, 0 = neutral
    probability: float  # Probability of success
    meta_probability: float  # Meta-learner probability
    forecast_return: float  # Expected return
    volatility_forecast: float  # Expected volatility
    bandit_weight: float  # Bandit algorithm weight
    uniqueness: float  # Strategy uniqueness score
    correlation_penalty: float  # Correlation penalty


class BreakoutStrategies:
    """
    Breakout trading strategies for trending markets.

    Each strategy identifies breakouts from consolidation periods
    and generates signals based on price action relative to key levels.

    Example:
        breakout_strats = BreakoutStrategies()
        signal = breakout_strats.donchian_breakout(features, "TRENDING", "1D")
    """

    def donchian_breakout(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> BreakoutSignal:
        """
        Donchian Channel Breakout Strategy.

        Classic turtle trader approach - trades breakouts of N-period
        high/low channels. Active in TRENDING regime.

        Logic:
        - Long if price breaks above N-period high
        - Short if price breaks below N-period low
        - Neutral if within channel

        Args:
            features: Feature dictionary with 'close', 'donchian_high', 'donchian_low'
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability (default 0.5)

        Returns:
            BreakoutSignal with strategy parameters
        """
        close = features.get("close", 100.0)
        donchian_high = features.get("donchian_high", close * 1.02)
        donchian_low = features.get("donchian_low", close * 0.98)
        volatility = features.get("volatility", 0.015)

        # Detect breakout
        high_break = close > donchian_high
        low_break = close < donchian_low

        if high_break:
            side = 1
            probability = 0.62
            forecast_return = 0.012
        elif low_break:
            side = -1
            probability = 0.62
            forecast_return = -0.012
        else:
            side = 0
            probability = 0.50
            forecast_return = 0.0

        return BreakoutSignal(
            strategy_name="donchian_breakout",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=volatility,
            bandit_weight=0.5,
            uniqueness=0.7,
            correlation_penalty=0.15
        )

    def range_breakout(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> BreakoutSignal:
        """
        Range/Consolidation Breakout Strategy.

        Identifies breakouts from recent trading ranges.
        Active in TRENDING and HIGH_VOL regimes.

        Logic:
        - Long if price breaks above range upper bound
        - Short if price breaks below range lower bound
        - Neutral if within range

        Args:
            features: Feature dictionary with 'close', 'range_upper', 'range_lower'
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability

        Returns:
            BreakoutSignal with strategy parameters
        """
        close = features.get("close", 100.0)
        range_upper = features.get("range_upper", close * 1.015)
        range_lower = features.get("range_lower", close * 0.985)
        volatility = features.get("volatility", 0.015)

        # Detect range breakout
        upper_break = close > range_upper
        lower_break = close < range_lower

        if upper_break:
            side = 1
            probability = 0.60
            forecast_return = 0.015
        elif lower_break:
            side = -1
            probability = 0.60
            forecast_return = -0.015
        else:
            side = 0
            probability = 0.50
            forecast_return = 0.0

        return BreakoutSignal(
            strategy_name="range_breakout",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=volatility,
            bandit_weight=0.45,
            uniqueness=0.65,
            correlation_penalty=0.12
        )

    def atr_breakout(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> BreakoutSignal:
        """
        ATR-Based Volatility Breakout Strategy.

        Uses Average True Range (ATR) to identify significant moves
        that exceed normal volatility. Active in HIGH_VOL regime.

        Logic:
        - Long if move > threshold * ATR above recent level
        - Short if move > threshold * ATR below recent level
        - Neutral otherwise

        Args:
            features: Feature dictionary with 'close', 'atr', 'prev_close'
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability

        Returns:
            BreakoutSignal with strategy parameters
        """
        close = features.get("close", 100.0)
        prev_close = features.get("prev_close", close)
        atr = features.get("atr", close * 0.02)  # Default 2% ATR
        volatility = features.get("volatility", 0.015)

        # Calculate move in ATR units
        move = close - prev_close
        atr_threshold = 1.5  # 1.5x ATR for breakout

        if move > atr * atr_threshold:
            side = 1
            probability = 0.63
            forecast_return = 0.018
        elif move < -atr * atr_threshold:
            side = -1
            probability = 0.63
            forecast_return = -0.018
        else:
            side = 0
            probability = 0.50
            forecast_return = 0.0

        return BreakoutSignal(
            strategy_name="atr_breakout",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=volatility,
            bandit_weight=0.55,
            uniqueness=0.75,
            correlation_penalty=0.18
        )

    def momentum_surge(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> BreakoutSignal:
        """
        Momentum Surge Strategy.

        Identifies sudden acceleration in momentum that indicates
        a breakout is occurring. Active in TRENDING regime.

        Logic:
        - Long if momentum > threshold and accelerating
        - Short if momentum < -threshold and accelerating
        - Neutral otherwise

        Args:
            features: Feature dictionary with 'momentum', 'momentum_change'
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability

        Returns:
            BreakoutSignal with strategy parameters
        """
        momentum = features.get("momentum", 0.0)
        momentum_change = features.get("momentum_change", 0.0)
        volatility = features.get("volatility", 0.015)

        # Surge threshold
        surge_threshold = 0.015
        acceleration_threshold = 0.005

        # Detect momentum surge
        bullish_surge = momentum > surge_threshold and momentum_change > acceleration_threshold
        bearish_surge = momentum < -surge_threshold and momentum_change < -acceleration_threshold

        if bullish_surge:
            side = 1
            probability = 0.64
            forecast_return = 0.020
        elif bearish_surge:
            side = -1
            probability = 0.64
            forecast_return = -0.020
        else:
            side = 0
            probability = 0.50
            forecast_return = 0.0

        return BreakoutSignal(
            strategy_name="momentum_surge",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=volatility,
            bandit_weight=0.60,
            uniqueness=0.68,
            correlation_penalty=0.20
        )
