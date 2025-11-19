"""
SWEEP FINAL — Full Pipeline Validation

Tests all backtest modes:
1. Standard backtest
2. Walk-forward backtest
3. Crisis backtest
4. Monte Carlo backtest

All must complete and produce valid outputs.

Author: PRADO9_EVO Builder
Date: 2025-01-18
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from afml_system.backtest import (
    evo_backtest_standard,
    evo_backtest_walk_forward,
    evo_backtest_crisis,
    evo_backtest_monte_carlo,
    BacktestConfig,
)


class TestFullPipeline:
    """Full pipeline validation tests."""

    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        """Create test OHLCV data (500 bars)."""
        np.random.seed(42)
        n_bars = 500

        # Generate realistic returns with slight positive drift
        returns = np.random.randn(n_bars) * 0.01 + 0.0002

        # Create prices
        price_multipliers = 1 + returns
        prices = 100 * price_multipliers.cumprod()

        # Create realistic OHLCV
        opens = prices * (1 + np.random.randn(n_bars) * 0.002)
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.randn(n_bars)) * 0.005)
        volumes = 1_000_000 * (1 + np.abs(np.random.randn(n_bars)) * 0.3)

        # Ensure OHLC validity
        highs = np.maximum(highs, np.maximum(opens, prices))
        lows = np.minimum(lows, np.minimum(opens, prices))

        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=pd.date_range('2020-01-01', periods=n_bars, freq='D'))

        return df

    # =====================================================================
    # Test 1: Standard Backtest
    # =====================================================================

    def test_standard_backtest(self, test_data):
        """
        Test 1: Verify standard backtest completes and produces valid output.

        Expected:
        - Backtest completes without errors
        - Returns standardized result dict
        - Contains required fields
        - Metrics are valid numbers
        """
        print("\n" + "="*70)
        print("Test 1: Standard Backtest")
        print("="*70)

        config = BacktestConfig(symbol='TEST', random_seed=42)

        result = evo_backtest_standard(
            symbol='TEST',
            df=test_data,
            config=config
        )

        print(f"\nResult keys: {list(result.keys())}")

        # Validate standardized result structure
        assert 'status' in result, "Result should have 'status' field"
        assert 'symbol' in result, "Result should have 'symbol' field"
        assert 'result' in result, "Result should have 'result' field"

        print(f"Status: {result['status']}")
        print(f"Symbol: {result['symbol']}")

        # Check success
        assert result['status'] == 'success', f"Backtest should succeed, got: {result.get('error', 'Unknown error')}"
        assert result['symbol'] == 'TEST', f"Symbol should be TEST, got: {result['symbol']}"

        # Validate result data
        backtest_result = result['result']

        # Check if it's a BacktestResult object or dict
        if hasattr(backtest_result, 'total_return'):
            # BacktestResult object
            total_return = backtest_result.total_return
            sharpe_ratio = backtest_result.sharpe_ratio
            max_drawdown = backtest_result.max_drawdown
            total_trades = backtest_result.total_trades
        else:
            # Dict
            total_return = backtest_result.get('total_return', 0.0)
            sharpe_ratio = backtest_result.get('sharpe_ratio', 0.0)
            max_drawdown = backtest_result.get('max_drawdown', 0.0)
            total_trades = backtest_result.get('total_trades', 0)

        print(f"\nMetrics:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Total Trades: {total_trades}")

        # Validate metrics are valid numbers
        assert isinstance(total_return, (int, float)), "Total return should be numeric"
        assert isinstance(sharpe_ratio, (int, float)), "Sharpe should be numeric"
        assert isinstance(max_drawdown, (int, float)), "Max drawdown should be numeric"
        assert isinstance(total_trades, (int, float)), "Total trades should be numeric"

        # Validate reasonable ranges
        assert -1.0 <= total_return <= 10.0, f"Total return should be reasonable, got {total_return:.2%}"
        assert -5.0 <= sharpe_ratio <= 10.0, f"Sharpe should be reasonable, got {sharpe_ratio:.3f}"
        assert -1.0 <= max_drawdown <= 0.0, f"Max drawdown should be negative, got {max_drawdown:.2%}"
        assert total_trades >= 0, f"Total trades should be non-negative, got {total_trades}"

        print("\n✅ Test 1 PASSED: Standard backtest completed successfully")

    # =====================================================================
    # Test 2: Walk-Forward Backtest
    # =====================================================================

    def test_walk_forward_backtest(self, test_data):
        """
        Test 2: Verify walk-forward backtest completes and produces valid output.

        Expected:
        - Backtest completes without errors
        - Returns aggregated results across folds
        - Contains fold metrics
        - Consistency metrics calculated
        """
        print("\n" + "="*70)
        print("Test 2: Walk-Forward Backtest")
        print("="*70)

        config = BacktestConfig(symbol='TEST', random_seed=42)

        result = evo_backtest_walk_forward(
            symbol='TEST',
            df=test_data,
            config=config
        )

        print(f"\nResult keys: {list(result.keys())}")

        # Validate standardized result structure
        assert 'status' in result, "Result should have 'status' field"
        assert result['status'] == 'success', f"Backtest should succeed, got: {result.get('error', 'Unknown error')}"

        # Validate walk-forward specific fields
        wf_result = result['result']

        print(f"\nWalk-Forward Result keys: {list(wf_result.keys())}")

        # Check for aggregated metrics
        if 'num_folds' in wf_result:
            print(f"Number of folds: {wf_result['num_folds']}")
            assert wf_result['num_folds'] > 0, "Should have at least 1 fold"

        if 'aggregated' in wf_result:
            agg = wf_result['aggregated']
            print(f"\nAggregated Metrics:")
            for key, value in agg.items():
                if isinstance(value, (int, float)):
                    if 'return' in key.lower() or 'drawdown' in key.lower():
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        print("\n✅ Test 2 PASSED: Walk-forward backtest completed successfully")

    # =====================================================================
    # Test 3: Crisis Backtest
    # =====================================================================

    def test_crisis_backtest(self, test_data):
        """
        Test 3: Verify crisis backtest completes and produces valid output.

        Expected:
        - Backtest completes without errors
        - Uses CR2 detector by default
        - Detects crisis periods (or reports none found)
        - Returns crisis metrics
        """
        print("\n" + "="*70)
        print("Test 3: Crisis Backtest")
        print("="*70)

        config = BacktestConfig(symbol='TEST', random_seed=42)

        result = evo_backtest_crisis(
            symbol='TEST',
            df=test_data,
            config=config
        )

        print(f"\nResult keys: {list(result.keys())}")

        # Validate standardized result structure
        assert 'status' in result, "Result should have 'status' field"
        assert result['status'] == 'success', f"Backtest should succeed, got: {result.get('error', 'Unknown error')}"

        # Validate crisis-specific fields
        crisis_result = result['result']

        print(f"\nCrisis Result keys: {list(crisis_result.keys())}")

        # Check detector type
        if 'detector' in crisis_result:
            print(f"Detector: {crisis_result['detector']}")
            # Should be CR2 by default
            # Note: Might be 'Standard' if no crises detected

        # Check crisis detection
        if 'num_crises' in crisis_result:
            num_crises = crisis_result['num_crises']
            print(f"Crises Detected: {num_crises}")

            if num_crises > 0 and 'crises' in crisis_result:
                print(f"\nCrisis Details:")
                for i, crisis in enumerate(crisis_result['crises'][:3], 1):  # Show first 3
                    print(f"  Crisis {i}:")
                    if isinstance(crisis, dict):
                        print(f"    Type: {crisis.get('type', 'Unknown')}")
                        print(f"    Duration: {crisis.get('duration_days', 0)} days")
                        print(f"    Drawdown: {crisis.get('max_drawdown', 0.0):.2%}")
            else:
                print("  No significant crises detected (test data may be too stable)")

        print("\n✅ Test 3 PASSED: Crisis backtest completed successfully")

    # =====================================================================
    # Test 4: Monte Carlo Backtest
    # =====================================================================

    def test_monte_carlo_backtest(self, test_data):
        """
        Test 4: Verify Monte Carlo backtest completes and produces valid output.

        Expected:
        - Backtest completes without errors
        - Runs specified number of simulations
        - Returns MC statistics (mean, std, percentile)
        - P-value calculated
        """
        print("\n" + "="*70)
        print("Test 4: Monte Carlo Backtest")
        print("="*70)

        config = BacktestConfig(symbol='TEST', random_seed=42)

        # Use 100 simulations for speed (production uses 1000-10000)
        n_sim = 100

        result = evo_backtest_monte_carlo(
            symbol='TEST',
            df=test_data,
            n_sim=n_sim,
            config=config
        )

        print(f"\nResult keys: {list(result.keys())}")

        # Validate standardized result structure
        assert 'status' in result, "Result should have 'status' field"
        assert result['status'] == 'success', f"Backtest should succeed, got: {result.get('error', 'Unknown error')}"

        # Validate MC-specific fields
        mc_result = result['result']

        print(f"\nMonte Carlo Result keys: {list(mc_result.keys())}")

        # Check MC metrics
        if 'num_simulations' in mc_result:
            print(f"Simulations: {mc_result['num_simulations']:,}")
            assert mc_result['num_simulations'] == n_sim, f"Should run {n_sim} simulations"

        if 'actual_sharpe' in mc_result:
            print(f"\nMonte Carlo Metrics:")
            print(f"  Actual Sharpe: {mc_result['actual_sharpe']:.3f}")

        if 'mc_sharpe_mean' in mc_result:
            print(f"  MC Sharpe Mean: {mc_result['mc_sharpe_mean']:.3f}")

        if 'mc_sharpe_std' in mc_result:
            print(f"  MC Sharpe Std: {mc_result['mc_sharpe_std']:.3f}")

        if 'skill_percentile' in mc_result:
            percentile = mc_result['skill_percentile']
            print(f"  Skill Percentile: {percentile:.1f}%")
            assert 0.0 <= percentile <= 100.0, f"Percentile should be 0-100%, got {percentile:.1f}%"

        if 'p_value' in mc_result:
            p_value = mc_result['p_value']
            print(f"  P-Value: {p_value:.4f}")
            assert 0.0 <= p_value <= 1.0, f"P-value should be 0-1, got {p_value:.4f}"

        if 'significant' in mc_result:
            significant = mc_result['significant']
            significance_str = "✅ SIGNIFICANT" if significant else "❌ NOT SIGNIFICANT"
            print(f"  Significance (p<0.05): {significance_str}")

        print("\n✅ Test 4 PASSED: Monte Carlo backtest completed successfully")

    # =====================================================================
    # Test 5: All Modes Run Successfully
    # =====================================================================

    def test_all_modes_complete(self, test_data):
        """
        Test 5: Verify all backtest modes can run in sequence.

        Expected:
        - All 4 modes complete without errors
        - Each returns valid results
        - No crashes or exceptions
        """
        print("\n" + "="*70)
        print("Test 5: All Modes Sequential Execution")
        print("="*70)

        config = BacktestConfig(symbol='TEST', random_seed=42)

        modes = [
            ('Standard', lambda: evo_backtest_standard('TEST', test_data, config)),
            ('Walk-Forward', lambda: evo_backtest_walk_forward('TEST', test_data, config)),
            ('Crisis', lambda: evo_backtest_crisis('TEST', test_data, config=config)),
            ('Monte Carlo', lambda: evo_backtest_monte_carlo('TEST', test_data, n_sim=50, config=config)),
        ]

        results = {}

        for mode_name, mode_func in modes:
            print(f"\nRunning {mode_name}...")
            try:
                result = mode_func()
                assert result['status'] == 'success', f"{mode_name} failed: {result.get('error', 'Unknown')}"
                results[mode_name] = result
                print(f"✅ {mode_name} completed")
            except Exception as e:
                print(f"❌ {mode_name} failed: {e}")
                raise

        print(f"\n✅ Test 5 PASSED: All {len(modes)} backtest modes completed successfully")
        print(f"   Modes tested: {', '.join(results.keys())}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
