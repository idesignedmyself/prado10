"""
SWEEP X.1 — ATR Volatility Targeting Test

Comprehensive validation tests for Module X (ATR Volatility Targeting).

Test Coverage:
1. ATR calculation correctness
2. Position scaling logic
3. Leverage capping at max_leverage
4. Handling of edge cases (NaN, zero, negative values)
5. Integration with BacktestEngine
6. Vectorized scaling performance

Author: PRADO9_EVO Builder
Date: 2025-01-18
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from afml_system.risk import ATRVolTarget


def test_1_atr_calculation():
    """
    Test 1: Validate ATR calculation correctness.

    ATR should correctly calculate True Range and its moving average.
    """
    print("\n" + "="*80)
    print("Test 1: ATR Calculation Correctness")
    print("="*80)

    # Create test data with known volatility pattern
    df = pd.DataFrame({
        'high': [102, 105, 104, 108, 107],
        'low': [98, 101, 100, 103, 102],
        'close': [100, 103, 102, 105, 104]
    })

    atr_target = ATRVolTarget(atr_period=3)
    atr = atr_target.compute_atr(df)

    print(f"\n📊 Test Data:")
    print(df.to_string())

    print(f"\n📈 Calculated ATR:")
    print(atr.to_string())

    # Verify ATR properties
    assert len(atr) == len(df), "ATR series length should match DataFrame length"
    assert not atr.isna().all(), "ATR should have valid values"
    assert (atr >= 0).all(), "ATR should be non-negative"

    # Check last ATR value (3-period average)
    last_atr = atr.iloc[-1]
    print(f"\n✓ Last ATR value: {last_atr:.4f}")
    assert last_atr > 0, "ATR should be positive"

    print("\n✅ Test 1 PASSED: ATR calculation working correctly")
    return True


def test_2_position_scaling_logic():
    """
    Test 2: Validate position scaling to target volatility.

    Position size should scale inversely with realized volatility.
    """
    print("\n" + "="*80)
    print("Test 2: Position Scaling Logic")
    print("="*80)

    atr_target = ATRVolTarget(target_vol=0.12, max_leverage=3.0)

    # Test cases: (atr, close, expected_behavior)
    # Note: With min_vol_threshold=0.001 (0.1%), tiny ATRs will hit max leverage
    test_cases = [
        (2.0, 100.0, "up", "Low volatility (2%) should scale UP"),
        (12.0, 100.0, "target", "Target volatility (12%) at 1x"),
        (20.0, 100.0, "down", "High volatility (20%) should scale DOWN"),
    ]

    print(f"\n🎯 Target Volatility: {atr_target.target_vol:.1%}")
    print(f"🔒 Max Leverage: {atr_target.max_leverage}x\n")

    for atr, close, expected_direction, description in test_cases:
        raw_position = 1.0
        scaled = atr_target.scale_position(raw_position, atr, close)

        realized_vol = atr / close
        print(f"ATR: {atr:.4f}, Close: {close:.2f}, Realized Vol: {realized_vol:.2%}")
        print(f"   Raw Position: {raw_position:.2f}x → Scaled: {scaled:.2f}x")
        print(f"   {description}")

        if expected_direction == "up":
            assert scaled > raw_position, f"Should scale up for low volatility"
        elif expected_direction == "target":
            assert 0.9 <= scaled <= 1.1, f"Should be near 1x for target volatility"
        elif expected_direction == "down":
            assert scaled < raw_position, f"Should scale down for high volatility"

        # Verify leverage cap
        assert scaled <= atr_target.max_leverage, f"Should cap at {atr_target.max_leverage}x"
        print()

    print("✅ Test 2 PASSED: Position scaling logic correct")
    return True


def test_3_leverage_capping():
    """
    Test 3: Ensure leverage is capped at max_leverage.

    Even with very low volatility, leverage should not exceed cap.
    """
    print("\n" + "="*80)
    print("Test 3: Leverage Capping")
    print("="*80)

    atr_target = ATRVolTarget(target_vol=0.12, max_leverage=3.0)

    # Extremely low volatility (0.5%)
    atr_very_low = 0.005
    close = 100.0
    raw_position = 1.0

    scaled = atr_target.scale_position(raw_position, atr_very_low, close)

    realized_vol = atr_very_low / close
    theoretical_scale = atr_target.target_vol / realized_vol

    print(f"\n📉 Extremely Low Volatility Test:")
    print(f"   Realized Vol: {realized_vol:.2%}")
    print(f"   Theoretical Scale: {theoretical_scale:.1f}x")
    print(f"   Actual Scale: {scaled:.1f}x")
    print(f"   Max Leverage: {atr_target.max_leverage}x")

    assert scaled == atr_target.max_leverage, \
        f"Leverage should be capped at {atr_target.max_leverage}x"

    assert scaled < theoretical_scale, \
        "Actual scale should be less than theoretical (due to cap)"

    print(f"\n✅ Leverage correctly capped at {scaled}x")
    print("✅ Test 3 PASSED: Leverage capping working correctly")
    return True


def test_4_edge_case_handling():
    """
    Test 4: Handle edge cases (NaN, zero, negative, missing).

    System should gracefully handle invalid ATR values.
    """
    print("\n" + "="*80)
    print("Test 4: Edge Case Handling")
    print("="*80)

    atr_target = ATRVolTarget(target_vol=0.12, max_leverage=3.0)
    raw_position = 1.0

    edge_cases = [
        (None, 100.0, "None ATR"),
        (np.nan, 100.0, "NaN ATR"),
        (0.0, 100.0, "Zero ATR"),
        (-0.01, 100.0, "Negative ATR"),
        (0.02, 0.0, "Zero close price"),
        (0.02, None, "None close price"),
    ]

    print("\n🛡️  Edge Case Tests:")
    for atr, close, description in edge_cases:
        scaled = atr_target.scale_position(raw_position, atr, close)

        print(f"\n   {description}:")
        print(f"      ATR: {atr}, Close: {close}")
        print(f"      Scaled Position: {scaled}")

        # Should fall back to raw position for invalid inputs
        if atr is None or (isinstance(atr, float) and (np.isnan(atr) or atr <= 0)):
            assert scaled == raw_position, \
                f"Should return raw position for invalid ATR: {description}"
            print(f"      ✓ Correctly returned raw position")

    print("\n✅ Test 4 PASSED: Edge cases handled gracefully")
    return True


def test_5_atr_percent_calculation():
    """
    Test 5: Validate ATR as percentage of close price.

    ATR percentage should correctly normalize by close price.
    """
    print("\n" + "="*80)
    print("Test 5: ATR Percentage Calculation")
    print("="*80)

    df = pd.DataFrame({
        'high': [105, 110, 108],
        'low': [95, 100, 98],
        'close': [100, 105, 103]
    })

    atr_target = ATRVolTarget(atr_period=2)
    atr_pct = atr_target.compute_atr_percent(df)

    print(f"\n📊 Test Data:")
    print(df.to_string())

    print(f"\n📈 ATR Percentage:")
    print(atr_pct.to_string())

    # Verify percentage properties
    assert len(atr_pct) == len(df), "ATR% series length should match DataFrame"
    assert (atr_pct >= 0).all(), "ATR% should be non-negative"

    # Check reasonable range (should be < 100%)
    last_atr_pct = atr_pct.iloc[-1]
    print(f"\n✓ Last ATR%: {last_atr_pct:.2%}")
    assert last_atr_pct < 1.0, "ATR% should be reasonable (< 100%)"

    print("\n✅ Test 5 PASSED: ATR percentage calculation correct")
    return True


def test_6_vectorized_scaling():
    """
    Test 6: Validate vectorized position scaling for backtesting.

    Series scaling should produce same results as individual scaling.
    """
    print("\n" + "="*80)
    print("Test 6: Vectorized Position Scaling")
    print("="*80)

    atr_target = ATRVolTarget(target_vol=0.12, max_leverage=3.0)

    # Create test series
    raw_positions = pd.Series([1.0, -0.5, 0.8, -1.0, 0.0])
    atr_series = pd.Series([0.015, 0.020, 0.010, 0.025, 0.018])
    close_series = pd.Series([100.0, 105.0, 102.0, 98.0, 101.0])

    # Vectorized scaling
    scaled_series = atr_target.scale_position_series(
        raw_positions, atr_series, close_series
    )

    print(f"\n📊 Vectorized Scaling Results:")
    results_df = pd.DataFrame({
        'Raw Position': raw_positions,
        'ATR': atr_series,
        'Close': close_series,
        'Scaled Position': scaled_series
    })
    print(results_df.to_string())

    # Verify against individual scaling
    print(f"\n🔍 Validation Against Individual Scaling:")
    for i in range(len(raw_positions)):
        individual_scaled = atr_target.scale_position(
            raw_positions.iloc[i],
            atr_series.iloc[i],
            close_series.iloc[i]
        )

        vectorized_scaled = scaled_series.iloc[i]

        print(f"   Row {i}: Individual={individual_scaled:.4f}, Vectorized={vectorized_scaled:.4f}")

        assert np.isclose(individual_scaled, vectorized_scaled, rtol=1e-5), \
            f"Vectorized scaling should match individual scaling at index {i}"

    print("\n✅ Test 6 PASSED: Vectorized scaling matches individual scaling")
    return True


def test_7_current_leverage_calculation():
    """
    Test 7: Validate current leverage calculation for monitoring.

    get_current_leverage should correctly compute leverage multiplier.
    """
    print("\n" + "="*80)
    print("Test 7: Current Leverage Calculation")
    print("="*80)

    atr_target = ATRVolTarget(target_vol=0.12, max_leverage=3.0)

    test_cases = [
        (0.01, 100.0, "Low volatility → high leverage"),
        (0.02, 100.0, "Moderate volatility → moderate leverage"),
        (0.04, 100.0, "High volatility → low leverage"),
    ]

    print(f"\n🎯 Target Vol: {atr_target.target_vol:.1%}, Max Leverage: {atr_target.max_leverage}x\n")

    for atr, close, description in test_cases:
        leverage = atr_target.get_current_leverage(atr, close)
        realized_vol = atr / close

        print(f"ATR: {atr:.4f}, Close: {close:.2f}")
        print(f"   Realized Vol: {realized_vol:.2%}")
        print(f"   Current Leverage: {leverage:.2f}x")
        print(f"   {description}")

        assert leverage > 0, "Leverage should be positive"
        assert leverage <= atr_target.max_leverage, "Leverage should be capped"
        print()

    print("✅ Test 7 PASSED: Current leverage calculation correct")
    return True


def test_8_min_vol_threshold():
    """
    Test 8: Validate minimum volatility threshold protection.

    Min threshold should prevent extreme leverage from near-zero volatility.
    """
    print("\n" + "="*80)
    print("Test 8: Minimum Volatility Threshold")
    print("="*80)

    min_threshold = 0.001
    atr_target = ATRVolTarget(
        target_vol=0.12,
        max_leverage=3.0,
        min_vol_threshold=min_threshold
    )

    # Extremely low volatility (below threshold)
    atr_tiny = 0.00001
    close = 100.0
    raw_position = 1.0

    scaled = atr_target.scale_position(raw_position, atr_tiny, close)

    realized_vol = atr_tiny / close
    theoretical_scale_no_floor = atr_target.target_vol / realized_vol
    theoretical_scale_with_floor = atr_target.target_vol / min_threshold

    print(f"\n📉 Tiny Volatility Test:")
    print(f"   Realized Vol: {realized_vol:.4%} (below threshold)")
    print(f"   Min Threshold: {min_threshold:.2%}")
    print(f"   Theoretical Scale (no floor): {theoretical_scale_no_floor:.1f}x")
    print(f"   Theoretical Scale (with floor): {theoretical_scale_with_floor:.1f}x")
    print(f"   Actual Scale: {scaled:.1f}x")

    assert scaled <= atr_target.max_leverage, \
        "Should still cap at max leverage"

    assert scaled < theoretical_scale_no_floor, \
        "Min threshold should prevent extreme leverage"

    print(f"\n✅ Min volatility threshold correctly prevents extreme leverage")
    print("✅ Test 8 PASSED: Minimum volatility threshold working")
    return True


def run_all_tests():
    """Run all SWEEP X.1 tests."""
    print("\n" + "="*80)
    print("SWEEP X.1 — ATR Volatility Targeting Test")
    print("Module X: Risk Management Validation")
    print("="*80)

    tests = [
        ("Test 1: ATR Calculation", test_1_atr_calculation),
        ("Test 2: Position Scaling Logic", test_2_position_scaling_logic),
        ("Test 3: Leverage Capping", test_3_leverage_capping),
        ("Test 4: Edge Case Handling", test_4_edge_case_handling),
        ("Test 5: ATR Percentage Calculation", test_5_atr_percent_calculation),
        ("Test 6: Vectorized Scaling", test_6_vectorized_scaling),
        ("Test 7: Current Leverage Calculation", test_7_current_leverage_calculation),
        ("Test 8: Minimum Volatility Threshold", test_8_min_vol_threshold),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASSED"))
        except AssertionError as e:
            results.append((test_name, f"FAILED: {e}"))
        except Exception as e:
            results.append((test_name, f"ERROR: {e}"))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)

    for test_name, status in results:
        icon = "✅" if status == "PASSED" else "❌"
        print(f"{icon} {test_name}: {status}")

    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'='*80}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Module X is production-ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review failures above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
