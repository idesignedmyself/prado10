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
    "HIGH_VOL": ["vol_breakout", "vol_spike_fade"],
    "LOW_VOL": ["vol_compression", "mean_reversion"],
    "TRENDING": ["momentum", "trend_breakout"],
    "MEAN_REVERTING": ["mean_reversion", "vol_mean_revert"],
    "NORMAL": ["momentum", "mean_reversion"]
}


class RegimeStrategySelector:
    """
    Selects active strategies based on detected market regime.

    This allows the system to dynamically adapt its strategy mix to
    current market conditions, improving performance and reducing risk.

    Example:
        selector = RegimeStrategySelector()
        active_strategies = selector.select("TRENDING")
        # Returns: ["momentum", "trend_breakout"]
    """

    def __init__(self, regime_map: Dict[str, List[str]] = None):
        """
        Initialize the regime strategy selector.

        Args:
            regime_map: Custom regime-to-strategy mapping. If None, uses DEFAULT_REGIME_MAP.
        """
        self.regime_map = regime_map or DEFAULT_REGIME_MAP.copy()

    def select(self, regime: str) -> List[str]:
        """
        Select active strategies for the given regime.

        Args:
            regime: Current market regime (HIGH_VOL, LOW_VOL, TRENDING, MEAN_REVERTING, NORMAL)

        Returns:
            List of strategy names to activate
        """
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
