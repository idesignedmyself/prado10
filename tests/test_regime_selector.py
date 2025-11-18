"""
SWEEP R.1 — Unit Tests for RegimeStrategySelector

Tests for Module R: Regime-Based Strategy Selection
"""

import pytest
from afml_system.regime import RegimeStrategySelector, DEFAULT_REGIME_MAP


class TestRegimeStrategySelector:
    """Unit tests for RegimeStrategySelector."""

    def setup_method(self):
        """Setup test fixtures."""
        self.selector = RegimeStrategySelector()

    def test_default_regime_mappings(self):
        """Test 1: Validate all default regime mappings."""
        # HIGH_VOL regime
        strategies = self.selector.select("HIGH_VOL")
        assert "vol_breakout" in strategies
        assert "vol_spike_fade" in strategies
        assert len(strategies) == 2

        # LOW_VOL regime
        strategies = self.selector.select("LOW_VOL")
        assert "vol_compression" in strategies
        assert "mean_reversion" in strategies
        assert len(strategies) == 2

        # TRENDING regime
        strategies = self.selector.select("TRENDING")
        assert "momentum" in strategies
        assert "trend_breakout" in strategies
        assert len(strategies) == 2

        # MEAN_REVERTING regime
        strategies = self.selector.select("MEAN_REVERTING")
        assert "mean_reversion" in strategies
        assert "vol_mean_revert" in strategies
        assert len(strategies) == 2

        # NORMAL regime
        strategies = self.selector.select("NORMAL")
        assert "momentum" in strategies
        assert "mean_reversion" in strategies
        assert len(strategies) == 2

    def test_unknown_regime_fallback(self):
        """Test 2: Unknown regime falls back to NORMAL."""
        strategies = self.selector.select("UNKNOWN_REGIME")
        assert strategies == DEFAULT_REGIME_MAP["NORMAL"]
        assert "momentum" in strategies
        assert "mean_reversion" in strategies

    def test_custom_regime_mapping(self):
        """Test 3: Custom regime mappings can be set."""
        custom_map = {
            "CUSTOM_REGIME": ["strategy_a", "strategy_b"],
            "NORMAL": ["momentum", "mean_reversion"]
        }
        selector = RegimeStrategySelector(regime_map=custom_map)

        strategies = selector.select("CUSTOM_REGIME")
        assert strategies == ["strategy_a", "strategy_b"]

    def test_update_regime_map(self):
        """Test 4: Regime map can be updated dynamically."""
        self.selector.update_regime_map("HIGH_VOL", ["new_strategy"])

        strategies = self.selector.select("HIGH_VOL")
        assert strategies == ["new_strategy"]

    def test_get_regime_map(self):
        """Test 5: Can retrieve current regime map."""
        regime_map = self.selector.get_regime_map()

        assert "HIGH_VOL" in regime_map
        assert "LOW_VOL" in regime_map
        assert "TRENDING" in regime_map
        assert "MEAN_REVERTING" in regime_map
        assert "NORMAL" in regime_map

    def test_determinism(self):
        """Test 6: Same regime returns same strategies consistently."""
        selector1 = RegimeStrategySelector()
        selector2 = RegimeStrategySelector()

        for regime in ["HIGH_VOL", "LOW_VOL", "TRENDING", "MEAN_REVERTING", "NORMAL"]:
            strategies1 = selector1.select(regime)
            strategies2 = selector2.select(regime)
            assert strategies1 == strategies2

    def test_empty_strategy_list(self):
        """Test 7: Handle regime with empty strategy list."""
        custom_map = {
            "EMPTY_REGIME": [],
            "NORMAL": ["momentum", "mean_reversion"]
        }
        selector = RegimeStrategySelector(regime_map=custom_map)

        strategies = selector.select("EMPTY_REGIME")
        assert strategies == []

    def test_regime_map_isolation(self):
        """Test 8: Modifying returned list doesn't affect internal map."""
        strategies = self.selector.select("NORMAL")
        original_length = len(strategies)

        # Modify returned list
        strategies.append("new_strategy")

        # Get strategies again
        strategies_again = self.selector.select("NORMAL")
        assert len(strategies_again) == original_length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
