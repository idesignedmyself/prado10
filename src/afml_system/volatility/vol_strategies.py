"""
PRADO9_EVO Module V — Volatility Strategy Engine

Volatility-based trading strategies that adapt to changing market conditions.

Strategies:
1. vol_breakout - Trade volatility expansions
2. vol_spike_fade - Fade extreme volatility spikes
3. vol_compression - Trade volatility compressions (anticipate breakout)
4. vol_mean_revert - Mean reversion on volatility itself

These strategies are designed to work with Module R (Regime Selector)
to activate only in appropriate market regimes.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class VolatilitySignal:
    """
    Signal dataclass for volatility-based strategies.

    Compatible with StrategySignal from Module G (Allocator).
    """
    strategy_name: str
    regime: str
    horizon: str
    side: int  # 1 = long, -1 = short
    probability: float  # Probability of success
    meta_probability: float  # Meta-learner probability
    forecast_return: float  # Expected return
    volatility_forecast: float  # Expected volatility
    bandit_weight: float  # Bandit algorithm weight
    uniqueness: float  # Strategy uniqueness score
    correlation_penalty: float  # Correlation penalty


class VolatilityStrategies:
    """
    Volatility-based trading strategies.

    Each strategy generates signals based on volatility conditions
    and is designed to activate in specific market regimes.

    Example:
        vol_strats = VolatilityStrategies()
        signal = vol_strats.vol_breakout(features, "HIGH_VOL", "1D")
    """

    def vol_breakout(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> VolatilitySignal:
        """
        Volatility Breakout Strategy.

        Trades in the direction of volatility expansion.
        Active in HIGH_VOL regime.

        Logic:
        - Long if volatility is expanding (vol > threshold)
        - Short if volatility is contracting

        Args:
            features: Feature dictionary with 'volatility' key
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability (default 0.5)

        Returns:
            VolatilitySignal with strategy parameters
        """
        vol = features.get("volatility", 0.015)

        # Breakout threshold: 2% annualized volatility
        side = 1 if vol > 0.02 else -1

        # Higher probability when volatility is clearly expanding
        probability = 0.60 if vol > 0.025 else 0.55

        # Forecast return scales with volatility magnitude
        forecast_return = vol * 0.02 if side == 1 else vol * -0.01

        return VolatilitySignal(
            strategy_name="vol_breakout",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=vol,
            bandit_weight=0.4,
            uniqueness=0.6,
            correlation_penalty=0.2
        )

    def vol_spike_fade(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> VolatilitySignal:
        """
        Volatility Spike Fade Strategy.

        Fades extreme volatility spikes (mean reversion on vol).
        Active in HIGH_VOL regime.

        Logic:
        - Short when volatility spikes (anticipate contraction)
        - Long when volatility is normal (stable)

        Args:
            features: Feature dictionary with 'volatility' key
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability

        Returns:
            VolatilitySignal with strategy parameters
        """
        vol = features.get("volatility", 0.015)

        # Fade threshold: 3% annualized volatility
        side = -1 if vol > 0.03 else 1

        # Higher probability on extreme spikes
        probability = 0.58 if vol > 0.04 else 0.55

        # Negative return when fading (contrarian)
        forecast_return = vol * -0.015 if side == -1 else 0.005

        return VolatilitySignal(
            strategy_name="vol_spike_fade",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=vol,
            bandit_weight=0.5,
            uniqueness=0.7,
            correlation_penalty=0.2
        )

    def vol_compression(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> VolatilitySignal:
        """
        Volatility Compression Strategy.

        Anticipates breakout after volatility compression.
        Active in LOW_VOL regime.

        Logic:
        - Long when volatility is compressed (anticipate expansion)
        - Short when volatility is normal/high

        Args:
            features: Feature dictionary with 'volatility' key
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability

        Returns:
            VolatilitySignal with strategy parameters
        """
        vol = features.get("volatility", 0.015)

        # Compression threshold: 1.2% annualized volatility
        side = 1 if vol < 0.012 else -1

        # Higher probability on extreme compression
        probability = 0.60 if vol < 0.010 else 0.58

        # Anticipate return from compression release
        forecast_return = 0.012 if side == 1 else -0.005

        return VolatilitySignal(
            strategy_name="vol_compression",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=vol,
            bandit_weight=0.3,
            uniqueness=0.8,
            correlation_penalty=0.1
        )

    def vol_mean_revert(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> VolatilitySignal:
        """
        Volatility Mean Reversion Strategy.

        Trades mean reversion of volatility to long-term average.
        Active in MEAN_REVERTING regime.

        Logic:
        - Short when volatility is above average (anticipate decline)
        - Long when volatility is below average (anticipate rise)

        Args:
            features: Feature dictionary with 'volatility' key
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability

        Returns:
            VolatilitySignal with strategy parameters
        """
        vol = features.get("volatility", 0.015)

        # Mean reversion threshold: 2% annualized volatility
        # Assume long-term average is ~1.5%
        side = -1 if vol > 0.02 else 1

        # Probability higher when far from mean
        probability = 0.59 if abs(vol - 0.015) > 0.01 else 0.57

        # Return from mean reversion
        forecast_return = 0.008 if side == 1 else 0.006

        return VolatilitySignal(
            strategy_name="vol_mean_revert",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=vol,
            bandit_weight=0.45,
            uniqueness=0.7,
            correlation_penalty=0.15
        )

    def trend_breakout(
        self,
        features: Dict[str, float],
        regime: str,
        horizon: str,
        meta_probability: float = 0.5
    ) -> VolatilitySignal:
        """
        Trend Breakout Strategy.

        Trades breakouts from consolidation ranges.
        Active in TRENDING regime.

        Logic:
        - Long when momentum is positive and strong
        - Short when momentum is negative and strong

        Args:
            features: Feature dictionary with 'momentum' key
            regime: Current market regime
            horizon: Time horizon
            meta_probability: Meta-learner probability

        Returns:
            VolatilitySignal with strategy parameters
        """
        momentum = features.get("momentum", 0.0)
        vol = features.get("volatility", 0.015)

        # Breakout threshold: momentum > threshold
        side = 1 if momentum > 0.01 else -1 if momentum < -0.01 else 0

        # Higher probability on strong trends
        probability = 0.62 if abs(momentum) > 0.02 else 0.58

        # Return proportional to momentum
        forecast_return = momentum * 0.5

        return VolatilitySignal(
            strategy_name="trend_breakout",
            regime=regime,
            horizon=horizon,
            side=side,
            probability=probability,
            meta_probability=meta_probability,
            forecast_return=forecast_return,
            volatility_forecast=vol,
            bandit_weight=0.55,
            uniqueness=0.65,
            correlation_penalty=0.25
        )
