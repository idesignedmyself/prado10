"""
PRADO9_EVO Module R — Regime-Based Strategy Selector

Dynamically selects which strategies to activate based on market regime.

Market Regimes:
- HIGH_VOL: High volatility environment
- LOW_VOL: Low volatility environment
- TRENDING: Strong directional trend
- MEAN_REVERTING: Range-bound oscillation
- NORMAL: Balanced market conditions

Strategy Types:
- momentum: Trend-following momentum
- mean_reversion: Mean reversion strategies
- vol_breakout: Volatility breakout (future)
- vol_compression: Volatility compression (future)
- vol_spike_fade: Volatility spike fade (future)
- vol_mean_revert: Volatility mean reversion (future)
- trend_breakout: Trend breakout (future)
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class RegimeStrategyConfig:
    """Configuration for regime-specific strategy selection."""
    regime: str
    enabled_strategies: List[str]


# Default regime-to-strategy mapping
DEFAULT_REGIME_MAP: Dict[str, List[str]] = {
    "HIGH_VOL": ["vol_breakout", "vol_spike_fade", "atr_breakout", "range_breakout"],
    "LOW_VOL": ["vol_compression", "mean_reversion"],
    "TRENDING": ["momentum", "donchian_breakout", "momentum_surge", "range_breakout"],
    "MEAN_REVERTING": ["mean_reversion", "vol_mean_revert"],
    "NORMAL": ["momentum", "mean_reversion"]
}


class RegimeStrategySelector:
    """
    Selects active strategies based on detected market regime.

    This allows the system to dynamically adapt its strategy mix to
    current market conditions, improving performance and reducing risk.

    PATCH 4: Includes regime smoothing (hysteresis) to prevent rapid switching.

    Example:
        selector = RegimeStrategySelector()
        active_strategies = selector.select("TRENDING")
        # Returns: ["momentum", "trend_breakout"]
    """

    def __init__(
        self,
        regime_map: Dict[str, List[str]] = None,
        min_regime_duration: int = 5,
        confirmation_bars: int = 3,
        confidence_threshold: float = 0.55
    ):
        """
        Initialize the regime strategy selector.

        Args:
            regime_map: Custom regime-to-strategy mapping. If None, uses DEFAULT_REGIME_MAP.
            min_regime_duration: Minimum bars to stay in a regime (default: 5)
            confirmation_bars: Bars needed to confirm regime change (default: 3)
            confidence_threshold: Minimum confidence to switch regimes (default: 0.55)
        """
        self.regime_map = regime_map or DEFAULT_REGIME_MAP.copy()

        # PATCH 4: Regime smoothing parameters
        self.min_regime_duration = min_regime_duration
        self.confirmation_bars = confirmation_bars
        self.confidence_threshold = confidence_threshold

        # Regime state tracking
        self.current_regime = "NORMAL"
        self.regime_bar_count = 0
        self.pending_regime = None
        self.pending_regime_count = 0

    def regime_smoother(self, new_regime: str, confidence: float = 1.0) -> str:
        """
        PATCH 4: Apply regime smoothing with hysteresis.

        Prevents rapid regime switching by requiring:
        1. Minimum regime duration (5 bars default)
        2. Confirmation period (3 bars default)
        3. Confidence threshold (55% default)

        Args:
            new_regime: Newly detected regime
            confidence: Confidence in the new regime (0.0 to 1.0)

        Returns:
            Smoothed regime (may differ from new_regime)
        """
        # Ensure regime exists (fallback to NORMAL)
        if new_regime is None or new_regime not in self.regime_map:
            new_regime = "NORMAL"

        # Increment current regime bar count
        self.regime_bar_count += 1

        # If confidence too low, stay with current regime
        if confidence < self.confidence_threshold:
            self.pending_regime = None
            self.pending_regime_count = 0
            return self.current_regime

        # If same as current regime, reset pending and continue
        if new_regime == self.current_regime:
            self.pending_regime = None
            self.pending_regime_count = 0
            return self.current_regime

        # Minimum duration check: stay in current regime if duration not met
        if self.regime_bar_count < self.min_regime_duration:
            return self.current_regime

        # Confirmation logic: need N consecutive bars to confirm change
        if new_regime == self.pending_regime:
            # Same pending regime, increment counter
            self.pending_regime_count += 1
        else:
            # Different regime detected, start new pending
            self.pending_regime = new_regime
            self.pending_regime_count = 1

        # Switch regime if confirmation period met
        if self.pending_regime_count >= self.confirmation_bars:
            self.current_regime = new_regime
            self.regime_bar_count = 0
            self.pending_regime = None
            self.pending_regime_count = 0

        return self.current_regime

    def select(self, regime: str, confidence: float = 1.0, use_smoothing: bool = True) -> List[str]:
        """
        Select active strategies for the given regime.

        Args:
            regime: Current market regime (HIGH_VOL, LOW_VOL, TRENDING, MEAN_REVERTING, NORMAL)
            confidence: Confidence in the regime (0.0 to 1.0, default: 1.0)
            use_smoothing: Whether to apply regime smoothing (default: True)

        Returns:
            List of strategy names to activate
        """
        # PATCH 4: Apply regime smoothing if enabled
        if use_smoothing:
            regime = self.regime_smoother(regime, confidence)

        return self.regime_map.get(regime, self.regime_map["NORMAL"])

    def update_regime_map(self, regime: str, strategies: List[str]):
        """
        Update the strategy mapping for a specific regime.

        Args:
            regime: Regime name
            strategies: List of strategy names to activate for this regime
        """
        self.regime_map[regime] = strategies

    def get_regime_map(self) -> Dict[str, List[str]]:
        """Get the current regime-to-strategy mapping."""
        return self.regime_map.copy()
