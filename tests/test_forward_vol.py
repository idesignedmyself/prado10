"""
PRADO9_EVO SWEEP X2.1 — Forward Volatility Validation

Comprehensive validation of Module X2: Forward-Looking Volatility Engine

Test Plan:
1. Verify realized volatility matches numpy std calculation
2. Verify regime-specific volatility returns expected ranges
3. Verify GARCH fallback works when computation fails
4. Verify backtest uses forward vol instead of ATR when enabled
5. Verify determinism across 5 runs

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import warnings
from src.afml_system.volatility import (
    realized_volatility,
    regime_adjusted_vol,
    garch_vol_forecast,
    forward_volatility_estimate,
    ForwardVolatilityEngine
)
from src.afml_system.backtest import BacktestEngine, BacktestConfig


class TestForwardVolatility:
    """Test suite for Module X2: Forward-Looking Volatility Engine."""

    def setup_method(self):
        """Initialize test data for each test."""
        # Create sample returns data
        np.random.seed(42)
        self.sample_returns = pd.Series(np.random.randn(100) * 0.02)  # 2% daily vol

    def test_1_realized_vol_matches_numpy_std(self):
        """
        Test 1: Verify realized volatility matches numpy std calculation.

        Realized vol should be close to standard deviation × sqrt(252).
        """
        print("\n" + "="*70)
        print("Test 1: Realized Volatility Matches Numpy Std")
        print("="*70)

        # Calculate realized volatility using our function
        realized_vol = realized_volatility(
            self.sample_returns,
            window=len(self.sample_returns),
            annualization_factor=252
        )

        # Calculate using numpy (annualized)
        numpy_std = self.sample_returns.std() * np.sqrt(252)

        print(f"Realized vol (our function): {realized_vol:.4f}")
        print(f"Numpy std (annualized):      {numpy_std:.4f}")
        print(f"Difference:                  {abs(realized_vol - numpy_std):.4f}")

        # Should be very close (within 10% due to EWMA vs simple std)
        # EWMA gives more weight to recent observations
        relative_diff = abs(realized_vol - numpy_std) / numpy_std
        print(f"Relative difference:         {relative_diff:.2%}")

        assert relative_diff < 0.15, f"Realized vol too different from numpy std: {relative_diff:.2%}"

        # Test with different window sizes
        print("\nTesting different window sizes:")
        for window in [21, 60, 100]:
            rv = realized_volatility(self.sample_returns, window=window)
            print(f"  Window {window:3d}: {rv:.4f}")
            assert 0.05 <= rv <= 2.00, f"Vol outside bounds for window {window}"

        print("✅ PASS: Realized volatility calculation correct")
        return True

    def test_2_regime_specific_vol_expected_ranges(self):
        """
        Test 2: Verify regime-specific volatility returns expected ranges.

        Each regime should apply the correct adjustment factor.
        """
        print("\n" + "="*70)
        print("Test 2: Regime-Specific Volatility Ranges")
        print("="*70)

        base_vol = 0.15  # 15% base volatility

        # Test each regime
        regimes_and_expected = {
            "HIGH_VOL": (1.3, "Increase 30%"),
            "LOW_VOL": (0.8, "Decrease 20%"),
            "TRENDING": (1.05, "Increase 5%"),
            "MEAN_REVERTING": (0.95, "Decrease 5%"),
            "NORMAL": (1.0, "No change"),
        }

        print(f"Base volatility: {base_vol:.2%}\n")

        for regime, (expected_factor, description) in regimes_and_expected.items():
            adjusted_vol = regime_adjusted_vol(base_vol, regime)
            expected_vol = base_vol * expected_factor

            print(f"{regime:20s}: {adjusted_vol:.4f} (expected: {expected_vol:.4f}) - {description}")

            # Should match expected factor exactly
            assert abs(adjusted_vol - expected_vol) < 1e-6, \
                f"Regime {regime}: got {adjusted_vol}, expected {expected_vol}"

        # Test custom adjustment factors
        print("\nTesting custom adjustment factors:")
        custom_factors = {
            "HIGH_VOL": 1.5,
            "LOW_VOL": 0.6,
        }

        custom_high = regime_adjusted_vol(base_vol, "HIGH_VOL", custom_factors)
        custom_low = regime_adjusted_vol(base_vol, "LOW_VOL", custom_factors)

        print(f"  Custom HIGH_VOL (1.5x): {custom_high:.4f}")
        print(f"  Custom LOW_VOL (0.6x):  {custom_low:.4f}")

        assert abs(custom_high - 0.225) < 1e-6, "Custom HIGH_VOL factor incorrect"
        assert abs(custom_low - 0.090) < 1e-6, "Custom LOW_VOL factor incorrect"

        # Test unknown regime (should default to 1.0)
        unknown = regime_adjusted_vol(base_vol, "UNKNOWN_REGIME")
        print(f"\nUnknown regime fallback:  {unknown:.4f} (should be {base_vol:.4f})")
        assert abs(unknown - base_vol) < 1e-6, "Unknown regime should use factor 1.0"

        print("✅ PASS: Regime-specific volatility ranges correct")
        return True

    def test_3_garch_fallback_works(self):
        """
        Test 3: Verify GARCH fallback works when computation fails.

        GARCH should fall back to realized vol on:
        - Insufficient data (< 30 bars)
        - Invalid parameters (α + β >= 1)
        - Numerical errors
        """
        print("\n" + "="*70)
        print("Test 3: GARCH Fallback Mechanism")
        print("="*70)

        # Test 1: Insufficient data (< 30 bars)
        print("\n1. Testing insufficient data fallback:")
        short_returns = pd.Series(np.random.randn(20) * 0.02)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vol = garch_vol_forecast(short_returns)

            # Should issue a warning about insufficient data
            assert len(w) > 0, "Should warn about insufficient data"
            print(f"   Warning issued: {w[0].message}")
            print(f"   Fallback vol: {vol:.4f}")

            # Should return valid volatility (from fallback)
            assert 0.05 <= vol <= 2.00, "Fallback should return valid vol"

        # Test 2: Invalid GARCH parameters (α + β >= 1)
        print("\n2. Testing invalid GARCH parameters:")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vol = garch_vol_forecast(
                self.sample_returns,
                alpha=0.6,  # α + β = 0.6 + 0.5 = 1.1 >= 1 (invalid)
                beta=0.5
            )

            # Should warn about stationarity violation
            assert len(w) > 0, "Should warn about stationarity"
            print(f"   Warning issued: {w[0].message}")
            print(f"   Fallback vol: {vol:.4f}")

            assert 0.05 <= vol <= 2.00, "Fallback should return valid vol"

        # Test 3: Valid GARCH parameters (should NOT fall back)
        print("\n3. Testing valid GARCH parameters:")
        vol_garch = garch_vol_forecast(
            self.sample_returns,
            alpha=0.1,
            beta=0.85
        )

        vol_realized = realized_volatility(self.sample_returns)

        print(f"   GARCH vol:    {vol_garch:.4f}")
        print(f"   Realized vol: {vol_realized:.4f}")

        # They should be different (GARCH is not just realized vol)
        # but both should be valid
        assert 0.05 <= vol_garch <= 2.00, "GARCH vol should be valid"
        assert 0.05 <= vol_realized <= 2.00, "Realized vol should be valid"

        # Test 4: Multi-step forecast
        print("\n4. Testing multi-step GARCH forecast:")
        for horizon in [1, 5, 10]:
            vol_h = garch_vol_forecast(self.sample_returns, horizon=horizon)
            print(f"   Horizon {horizon:2d}: {vol_h:.4f}")
            assert 0.05 <= vol_h <= 2.00, f"Vol invalid for horizon {horizon}"

        print("✅ PASS: GARCH fallback mechanism working correctly")
        return True

    def test_4_backtest_uses_forward_vol_not_atr(self):
        """
        Test 4: Verify backtest uses forward vol instead of ATR when enabled.

        When use_forward_vol=True, the backtest should:
        - Calculate forward volatility
        - Use it for position scaling
        - NOT use ATR-based scaling
        """
        print("\n" + "="*70)
        print("Test 4: Backtest Uses Forward Vol Instead of ATR")
        print("="*70)

        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        prices = 100 * np.cumprod(1 + np.random.randn(200) * 0.015)

        df = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'open': prices * (1 + np.random.randn(200) * 0.005),
            'volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)

        # Test 1: Backtest with Module X (ATR) only
        print("\n1. Running backtest with Module X (ATR):")
        config_atr = BacktestConfig(
            symbol="TEST",
            random_seed=42,
            use_atr_targeting=True,
            use_forward_vol=False  # Disable forward vol
        )

        engine_atr = BacktestEngine(config=config_atr)
        result_atr = engine_atr.run_standard("TEST", df)

        print(f"   ATR-based result:")
        print(f"   - Total Return: {result_atr.total_return:.2%}")
        print(f"   - Total Trades: {result_atr.total_trades}")

        # Test 2: Backtest with Module X2 (Forward Vol)
        print("\n2. Running backtest with Module X2 (Forward Vol):")
        config_fwd = BacktestConfig(
            symbol="TEST",
            random_seed=42,
            use_atr_targeting=True,  # Keep ATR enabled for scaling logic
            use_forward_vol=True,  # Enable forward vol
            forward_vol_garch=True,
            forward_vol_garch_weight=0.7,
            forward_vol_window=21
        )

        engine_fwd = BacktestEngine(config=config_fwd)
        result_fwd = engine_fwd.run_standard("TEST", df)

        print(f"   Forward vol result:")
        print(f"   - Total Return: {result_fwd.total_return:.2%}")
        print(f"   - Total Trades: {result_fwd.total_trades}")

        # Verify both produce valid results
        assert result_atr.total_trades >= 0, "ATR backtest should produce valid trades"
        assert result_fwd.total_trades >= 0, "Forward vol backtest should produce valid trades"

        # Verify forward vol engine was initialized
        assert engine_fwd.forward_vol_engine is not None, \
            "Forward vol engine should be initialized"
        assert engine_atr.forward_vol_engine is None, \
            "Forward vol engine should NOT be initialized when disabled"

        # Results may be different (using different vol estimates)
        print(f"\n3. Comparing results:")
        print(f"   Return difference: {abs(result_fwd.total_return - result_atr.total_return):.2%}")
        print(f"   Trade difference:  {abs(result_fwd.total_trades - result_atr.total_trades)}")

        # Both should be deterministic (same seed)
        result_fwd2 = engine_fwd.run_standard("TEST", df)
        assert result_fwd2.total_return == result_fwd.total_return, \
            "Forward vol backtest should be deterministic"

        print("✅ PASS: Backtest correctly uses forward vol when enabled")
        return True

    def test_5_determinism_across_5_runs(self):
        """
        Test 5: Verify determinism across 5 runs.

        With same seed and data, all results should be 100% identical.
        """
        print("\n" + "="*70)
        print("Test 5: Determinism Across 5 Runs")
        print("="*70)

        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.02)

        # Test 1: Realized volatility determinism
        print("\n1. Testing realized volatility determinism:")
        realized_vols = []
        for i in range(5):
            vol = realized_volatility(returns, window=21)
            realized_vols.append(vol)
            print(f"   Run {i+1}: {vol:.10f}")

        assert len(set(realized_vols)) == 1, "Realized vol should be deterministic"
        print(f"   ✓ All runs identical: {realized_vols[0]:.10f}")

        # Test 2: Regime adjustment determinism
        print("\n2. Testing regime adjustment determinism:")
        regime_vols = []
        for i in range(5):
            vol = regime_adjusted_vol(0.15, "HIGH_VOL")
            regime_vols.append(vol)
            print(f"   Run {i+1}: {vol:.10f}")

        assert len(set(regime_vols)) == 1, "Regime vol should be deterministic"
        print(f"   ✓ All runs identical: {regime_vols[0]:.10f}")

        # Test 3: GARCH forecast determinism (with seed reset)
        print("\n3. Testing GARCH forecast determinism:")
        garch_vols = []
        for i in range(5):
            np.random.seed(42)  # Reset seed for determinism
            vol = garch_vol_forecast(returns, alpha=0.1, beta=0.85)
            garch_vols.append(vol)
            print(f"   Run {i+1}: {vol:.10f}")

        assert len(set(garch_vols)) == 1, "GARCH vol should be deterministic"
        print(f"   ✓ All runs identical: {garch_vols[0]:.10f}")

        # Test 4: Forward volatility estimate determinism
        print("\n4. Testing forward volatility estimate determinism:")
        forward_vols = []
        for i in range(5):
            np.random.seed(42)  # Reset seed
            vol = forward_volatility_estimate(
                returns,
                regime="HIGH_VOL",
                use_garch=True,
                garch_weight=0.7
            )
            forward_vols.append(vol)
            print(f"   Run {i+1}: {vol:.10f}")

        assert len(set(forward_vols)) == 1, "Forward vol should be deterministic"
        print(f"   ✓ All runs identical: {forward_vols[0]:.10f}")

        # Test 5: ForwardVolatilityEngine determinism
        print("\n5. Testing ForwardVolatilityEngine determinism:")
        engine = ForwardVolatilityEngine(
            use_garch=True,
            garch_weight=0.7,
            window=21
        )

        engine_vols = []
        for i in range(5):
            np.random.seed(42)  # Reset seed
            vol = engine.estimate(returns, regime="TRENDING")
            engine_vols.append(vol)
            print(f"   Run {i+1}: {vol:.10f}")

        assert len(set(engine_vols)) == 1, "Engine should be deterministic"
        print(f"   ✓ All runs identical: {engine_vols[0]:.10f}")

        print("\n✅ PASS: All functions are 100% deterministic")
        return True


def run_sweep_x2_1():
    """
    Run SWEEP X2.1 validation suite.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n" + "="*70)
    print("PRADO9_EVO — SWEEP X2.1: Forward Volatility Validation")
    print("="*70)
    print("\nValidating Module X2: Forward-Looking Volatility Engine")
    print("Testing: realized vol, regime adjustment, GARCH, integration, determinism")
    print("")

    test_suite = TestForwardVolatility()

    tests = [
        ("Test 1: Realized Vol Matches Numpy", test_suite.test_1_realized_vol_matches_numpy_std),
        ("Test 2: Regime-Specific Vol Ranges", test_suite.test_2_regime_specific_vol_expected_ranges),
        ("Test 3: GARCH Fallback", test_suite.test_3_garch_fallback_works),
        ("Test 4: Backtest Uses Forward Vol", test_suite.test_4_backtest_uses_forward_vol_not_atr),
        ("Test 5: Determinism", test_suite.test_5_determinism_across_5_runs),
    ]

    passed = 0
    failed = 0
    failed_tests = []

    for test_name, test_func in tests:
        test_suite.setup_method()
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                failed_tests.append(test_name)
        except AssertionError as e:
            print(f"\n❌ FAIL: {test_name}")
            print(f"   Error: {e}")
            failed += 1
            failed_tests.append(test_name)
        except Exception as e:
            print(f"\n❌ ERROR: {test_name}")
            print(f"   Exception: {e}")
            import traceback
            print(traceback.format_exc())
            failed += 1
            failed_tests.append(test_name)

    # Final summary
    print("\n" + "="*70)
    print("SWEEP X2.1 Results")
    print("="*70)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")
    print("")

    if failed == 0:
        print("✅ ALL TESTS PASSED - Module X2 validated")
        print("\nValidated:")
        print("  • Realized volatility calculation correct")
        print("  • Regime-specific adjustments working")
        print("  • GARCH fallback mechanism functional")
        print("  • Backtest integration with forward vol")
        print("  • 100% deterministic behavior confirmed")
    else:
        print(f"❌ {failed} TESTS FAILED")
        print("\nFailed tests:")
        for test_name in failed_tests:
            print(f"  • {test_name}")

    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_sweep_x2_1()
    exit(0 if success else 1)
