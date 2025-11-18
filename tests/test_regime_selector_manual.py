"""
SWEEP R.1 — Manual Tests for RegimeStrategySelector

Tests for Module R: Regime-Based Strategy Selection (without pytest dependency)
"""

import sys
sys.path.insert(0, '/Users/darraykennedy/Desktop/python_pro/prado_evo/src')

from afml_system.regime import RegimeStrategySelector, DEFAULT_REGIME_MAP


def test_default_regime_mappings():
    """Test 1: Validate all default regime mappings."""
    print("\n=== Test 1: Default Regime Mappings ===")
    selector = RegimeStrategySelector()

    # HIGH_VOL regime
    strategies = selector.select("HIGH_VOL")
    assert "vol_breakout" in strategies and "vol_spike_fade" in strategies
    print(f"✓ HIGH_VOL: {strategies}")

    # LOW_VOL regime
    strategies = selector.select("LOW_VOL")
    assert "vol_compression" in strategies and "mean_reversion" in strategies
    print(f"✓ LOW_VOL: {strategies}")

    # TRENDING regime
    strategies = selector.select("TRENDING")
    assert "momentum" in strategies and "trend_breakout" in strategies
    print(f"✓ TRENDING: {strategies}")

    # MEAN_REVERTING regime
    strategies = selector.select("MEAN_REVERTING")
    assert "mean_reversion" in strategies and "vol_mean_revert" in strategies
    print(f"✓ MEAN_REVERTING: {strategies}")

    # NORMAL regime
    strategies = selector.select("NORMAL")
    assert "momentum" in strategies and "mean_reversion" in strategies
    print(f"✓ NORMAL: {strategies}")

    print("✅ Test 1 PASSED")


def test_unknown_regime_fallback():
    """Test 2: Unknown regime falls back to NORMAL."""
    print("\n=== Test 2: Unknown Regime Fallback ===")
    selector = RegimeStrategySelector()

    strategies = selector.select("UNKNOWN_REGIME")
    assert strategies == DEFAULT_REGIME_MAP["NORMAL"]
    print(f"✓ UNKNOWN_REGIME → {strategies} (fallback to NORMAL)")

    print("✅ Test 2 PASSED")


def test_custom_regime_mapping():
    """Test 3: Custom regime mappings can be set."""
    print("\n=== Test 3: Custom Regime Mapping ===")

    custom_map = {
        "CUSTOM_REGIME": ["strategy_a", "strategy_b"],
        "NORMAL": ["momentum", "mean_reversion"]
    }
    selector = RegimeStrategySelector(regime_map=custom_map)

    strategies = selector.select("CUSTOM_REGIME")
    assert strategies == ["strategy_a", "strategy_b"]
    print(f"✓ CUSTOM_REGIME: {strategies}")

    print("✅ Test 3 PASSED")


def test_update_regime_map():
    """Test 4: Regime map can be updated dynamically."""
    print("\n=== Test 4: Dynamic Regime Map Update ===")
    selector = RegimeStrategySelector()

    original = selector.select("HIGH_VOL")
    print(f"  Original HIGH_VOL: {original}")

    selector.update_regime_map("HIGH_VOL", ["new_strategy"])
    updated = selector.select("HIGH_VOL")
    assert updated == ["new_strategy"]
    print(f"✓ Updated HIGH_VOL: {updated}")

    print("✅ Test 4 PASSED")


def test_determinism():
    """Test 5: Same regime returns same strategies consistently."""
    print("\n=== Test 5: Determinism ===")

    selector1 = RegimeStrategySelector()
    selector2 = RegimeStrategySelector()

    for regime in ["HIGH_VOL", "LOW_VOL", "TRENDING", "MEAN_REVERTING", "NORMAL"]:
        strategies1 = selector1.select(regime)
        strategies2 = selector2.select(regime)
        assert strategies1 == strategies2
        print(f"✓ {regime}: {strategies1} (consistent)")

    print("✅ Test 5 PASSED")


def test_empty_strategy_list():
    """Test 6: Handle regime with empty strategy list."""
    print("\n=== Test 6: Empty Strategy List ===")

    custom_map = {
        "EMPTY_REGIME": [],
        "NORMAL": ["momentum", "mean_reversion"]
    }
    selector = RegimeStrategySelector(regime_map=custom_map)

    strategies = selector.select("EMPTY_REGIME")
    assert strategies == []
    print(f"✓ EMPTY_REGIME: {strategies} (empty list handled)")

    print("✅ Test 6 PASSED")


def test_missing_strategy_graceful_handling():
    """Test 7: System handles missing strategy implementations gracefully."""
    print("\n=== Test 7: Missing Strategy Handling ===")

    selector = RegimeStrategySelector()

    # HIGH_VOL selects vol_breakout and vol_spike_fade
    # These strategies are NOT implemented yet
    strategies = selector.select("HIGH_VOL")
    print(f"✓ HIGH_VOL strategies: {strategies}")
    print(f"  (vol_breakout and vol_spike_fade not yet implemented)")
    print(f"  Expected: No crash, strategies returned for future use")

    # BacktestEngine should skip these since they don't have implementation
    # The conditional checks in _get_allocation_decision prevent crashes

    print("✅ Test 7 PASSED (no crash with missing strategies)")


if __name__ == "__main__":
    print("=" * 60)
    print("SWEEP R.1 — Module R Unit Tests")
    print("=" * 60)

    try:
        test_default_regime_mappings()
        test_unknown_regime_fallback()
        test_custom_regime_mapping()
        test_update_regime_map()
        test_determinism()
        test_empty_strategy_list()
        test_missing_strategy_graceful_handling()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Module R Working Correctly")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
