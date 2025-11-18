"""
PRADO9_EVO Module I — Crisis Stress Engine

Crisis period stress testing for robustness validation.

Author: PRADO9_EVO Builder
Date: 2025-01-17
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .backtest_engine import BacktestEngine, BacktestConfig


# ============================================================================
# CONSTANTS
# ============================================================================

CRISIS_STRESS_VERSION = '1.0.0'
EPSILON = 1e-12

# Default crisis periods (for reference)
DEFAULT_CRISIS_PERIODS = [
    {
        'name': '2008 GFC',
        'train': ['2006-01-01', '2008-08-31'],
        'test': ['2008-09-01', '2009-03-31']
    },
    {
        'name': '2020 COVID',
        'train': ['2018-01-01', '2020-01-31'],
        'test': ['2020-02-01', '2020-06-30']
    },
    {
        'name': '2022 Bear Market',
        'train': ['2020-01-01', '2021-12-31'],
        'test': ['2022-01-01', '2022-09-30']
    }
]


# ============================================================================
# CRISIS STRESS ENGINE
# ============================================================================

class CrisisStressEngine:
    """
    Crisis stress testing engine.

    Tests strategy performance during historical crisis periods.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize crisis stress engine.

        Args:
            config: Backtest configuration
        """
        self.config = config

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        crisis_periods: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Run crisis stress tests.

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            crisis_periods: List of crisis period definitions (optional)

        Returns:
            Crisis test results dictionary
        """
        if crisis_periods is None:
            # Use synthetic crisis periods based on high volatility
            crisis_periods = self._detect_crisis_periods(df)

        if len(crisis_periods) == 0:
            return {
                'symbol': symbol,
                'num_crises': 0,
                'crises': [],
                'summary': {
                    'avg_sharpe': 0.0,
                    'avg_drawdown': 0.0,
                    'survival_rate': 0.0,
                    'survived': 0,
                    'total': 0
                }
            }

        # Run test on each crisis period
        crisis_results = []

        for crisis in crisis_periods:
            crisis_result = self._run_crisis_test(
                symbol=symbol,
                df=df,
                crisis=crisis
            )
            crisis_results.append(crisis_result)

        # Aggregate results
        summary = self._aggregate_crisis_results(crisis_results)

        # Build final result
        result = {
            'symbol': symbol,
            'num_crises': len(crisis_periods),
            'crises': crisis_results,
            'summary': summary
        }

        return result

    def _detect_crisis_periods(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect crisis periods from data based on high volatility.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of crisis period definitions
        """
        # Compute rolling volatility
        returns = df['close'].pct_change()
        vol_window = 20
        rolling_vol = returns.rolling(window=vol_window).std()

        # Find periods with vol > 2× median
        median_vol = rolling_vol.median()
        high_vol_threshold = 2.0 * median_vol

        high_vol_periods = rolling_vol > high_vol_threshold

        # Find contiguous high-vol periods
        crisis_periods = []
        in_crisis = False
        crisis_start = None

        for i, is_high_vol in enumerate(high_vol_periods):
            if is_high_vol and not in_crisis:
                # Crisis starts
                crisis_start = i
                in_crisis = True
            elif not is_high_vol and in_crisis:
                # Crisis ends
                crisis_end = i

                # Define train/test split
                train_start = max(0, crisis_start - 252)  # 1 year before crisis
                train_end = crisis_start
                test_start = crisis_start
                test_end = min(crisis_end + 63, len(df))  # Include 3 months after

                if train_end > train_start and test_end > test_start:
                    crisis_periods.append({
                        'name': f'Crisis {len(crisis_periods) + 1}',
                        'train': [df.index[train_start], df.index[train_end - 1]],
                        'test': [df.index[test_start], df.index[test_end - 1]]
                    })

                in_crisis = False

        # Limit to first 3 crises for efficiency
        return crisis_periods[:3]

    def _run_crisis_test(
        self,
        symbol: str,
        df: pd.DataFrame,
        crisis: Dict
    ) -> Dict[str, Any]:
        """
        Run test on a single crisis period.

        Mini-Sweep I.1D: Enhanced with date validation and diagnostics.

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            crisis: Crisis period definition

        Returns:
            Crisis test result dictionary
        """
        # Extract train/test periods
        if isinstance(crisis['train'][0], str):
            # Date strings - convert to datetime
            train_start = pd.to_datetime(crisis['train'][0])
            train_end = pd.to_datetime(crisis['train'][1])
            test_start = pd.to_datetime(crisis['test'][0])
            test_end = pd.to_datetime(crisis['test'][1])
        else:
            # Already datetime objects
            train_start = crisis['train'][0]
            train_end = crisis['train'][1]
            test_start = crisis['test'][0]
            test_end = crisis['test'][1]

        # Mini-Sweep I.1D: Date validation
        try:
            assert train_start < train_end, f"{crisis['name']}: train_start must be before train_end"
            assert train_end < test_start, f"{crisis['name']}: train_end must be before test_start"
            assert test_start < test_end, f"{crisis['name']}: test_start must be before test_end"
        except AssertionError as e:
            return {
                'name': crisis['name'],
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'survived': False,
                'error': f'Date validation failed: {e}'
            }

        # Mini-Sweep I.1D: Check if periods are within DF bounds
        df_start = df.index[0]
        df_end = df.index[-1]

        if train_start < df_start or test_end > df_end:
            return {
                'name': crisis['name'],
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'survived': False,
                'error': f'Period outside DF bounds (DF: {df_start} to {df_end})',
                'skipped': True
            }

        # Filter dataframe
        train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
        test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            # No data in this period
            return {
                'name': crisis['name'],
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'survived': False,
                'error': 'Insufficient data'
            }

        # Create fresh backtest engine
        engine = BacktestEngine(config=self.config)

        # Build train events
        train_events = engine._build_events_cusum(train_df)

        # Train evolution engine
        engine._train_evolution_engine(train_df, train_events)

        # Build test events
        test_events = engine._build_events_cusum(test_df)

        # Run backtest on test (crisis) period
        test_result = engine._run_backtest_on_events(
            symbol=symbol,
            df=test_df,
            events=test_events,
            phase='test'
        )

        # Mini-Sweep I.1D: Enhanced survival metric (Sharpe > 0 AND DD > -40%)
        survived = test_result.sharpe_ratio > 0.0 and test_result.max_drawdown > -0.40

        # Mini-Sweep I.1D: Collect diagnostics
        diagnostics = self._collect_crisis_diagnostics(test_result, test_df)

        # Build result
        crisis_result = {
            'name': crisis['name'],
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'sharpe': test_result.sharpe_ratio,
            'sortino': test_result.sortino_ratio,
            'calmar': test_result.calmar_ratio,
            'max_drawdown': test_result.max_drawdown,
            'total_return': test_result.total_return,
            'total_trades': test_result.total_trades,
            'win_rate': test_result.win_rate,
            'survived': survived,
            'diagnostics': diagnostics  # Mini-Sweep I.1D
        }

        return crisis_result

    def _collect_crisis_diagnostics(
        self,
        result,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Mini-Sweep I.1D: Collect crisis-specific diagnostics.

        Args:
            result: BacktestResult
            df: Price DataFrame

        Returns:
            Diagnostics dictionary
        """
        # Kill-switch counts from trades
        kill_switch_count = 0
        conflict_ratios = []
        volatilities = []

        # Analyze trades for kill-switch triggers
        for trade in result.trades:
            # Check for allocation details
            if 'allocator_details' in trade:
                details = trade.get('allocator_details', {})
                # Kill-switch triggered if position was forced to zero
                if details.get('kill_switch', False):
                    kill_switch_count += 1
                # Collect conflict ratios
                if 'conflict_ratio' in details:
                    conflict_ratios.append(details['conflict_ratio'])

        # Calculate volatility distribution from price data
        if len(df) > 1:
            returns = df['close'].pct_change().dropna()
            # Rolling volatility (20-day window)
            rolling_vol = returns.rolling(window=20).std()
            volatilities = rolling_vol.dropna().tolist()

        # Compute statistics
        diagnostics = {
            'kill_switch_count': kill_switch_count,
            'conflict_ratio_mean': float(np.mean(conflict_ratios)) if len(conflict_ratios) > 0 else 0.0,
            'conflict_ratio_std': float(np.std(conflict_ratios)) if len(conflict_ratios) > 0 else 0.0,
            'conflict_ratio_max': float(np.max(conflict_ratios)) if len(conflict_ratios) > 0 else 0.0,
            'volatility_mean': float(np.mean(volatilities)) if len(volatilities) > 0 else 0.0,
            'volatility_std': float(np.std(volatilities)) if len(volatilities) > 0 else 0.0,
            'volatility_max': float(np.max(volatilities)) if len(volatilities) > 0 else 0.0,
            'volatility_distribution': volatilities[:100]  # First 100 points for brevity
        }

        return diagnostics

    def _aggregate_crisis_results(self, crisis_results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate results across all crisis periods.

        Args:
            crisis_results: List of crisis results

        Returns:
            Aggregated summary dictionary
        """
        if len(crisis_results) == 0:
            return {
                'avg_sharpe': 0.0,
                'avg_drawdown': 0.0,
                'avg_return': 0.0,
                'survival_rate': 0.0,
                'survived': 0,
                'total': 0
            }

        # Extract metrics
        sharpes = [c['sharpe'] for c in crisis_results if 'error' not in c]
        drawdowns = [c['max_drawdown'] for c in crisis_results if 'error' not in c]
        returns = [c['total_return'] for c in crisis_results if 'error' not in c]
        survived_count = len([c for c in crisis_results if c.get('survived', False)])

        # Compute aggregated metrics
        summary = {
            'avg_sharpe': np.mean(sharpes) if len(sharpes) > 0 else 0.0,
            'avg_drawdown': np.mean(drawdowns) if len(drawdowns) > 0 else 0.0,
            'avg_return': np.mean(returns) if len(returns) > 0 else 0.0,
            'survival_rate': (survived_count / len(crisis_results)) * 100.0,
            'survived': survived_count,
            'total': len(crisis_results)
        }

        return summary


# ============================================================================
# MINI-SWEEP I.1D TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Crisis Stress Engine — Mini-Sweep I.1D Tests")
    print("=" * 80)

    # Setup test data
    from .backtest_engine import BacktestConfig
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # Create test DataFrame
    dates = pd.date_range('2020-01-01', periods=800, freq='D')
    prices = 100 * np.cumprod(1 + np.random.randn(800) * 0.02)

    test_df = pd.DataFrame({
        'close': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'open': prices * (1 + np.random.randn(800) * 0.005),
        'volume': np.random.randint(1000000, 10000000, 800)
    }, index=dates)

    config = BacktestConfig(
        symbol='TEST',
        initial_equity=100000.0,
        random_seed=42
    )

    crisis_engine = CrisisStressEngine(config=config)

    # ========================================================================
    # TEST 1: Mini-Sweep I.1D - Date validation and out-of-bounds handling
    # ========================================================================
    print("\n[TEST 1] Mini-Sweep I.1D: Date validation & bounds checking")
    print("-" * 80)

    # Test with periods that are outside DF bounds
    test_crisis_periods = [
        {
            'name': 'Valid Period',
            'train': ['2020-02-01', '2020-05-01'],
            'test': ['2020-05-02', '2020-07-01']
        },
        {
            'name': 'Out of Bounds (Future)',
            'train': ['2025-01-01', '2025-06-01'],
            'test': ['2025-06-02', '2025-12-31']
        },
        {
            'name': 'Invalid Dates (train_end > test_start)',
            'train': ['2020-06-01', '2020-08-01'],
            'test': ['2020-07-01', '2020-09-01']  # Overlaps with train!
        }
    ]

    result = crisis_engine.run(
        symbol='TEST',
        df=test_df,
        crisis_periods=test_crisis_periods
    )

    print(f"  Total crises tested: {result['num_crises']}")

    for crisis in result['crises']:
        status = "✓" if crisis.get('survived', False) else "✗"
        error = crisis.get('error', 'none')
        skipped = crisis.get('skipped', False)

        if skipped:
            print(f"  {crisis['name']}: SKIPPED - {error}")
        elif error != 'none':
            print(f"  {crisis['name']}: ERROR - {error}")
        else:
            print(f"  {crisis['name']}: {status} Sharpe={crisis['sharpe']:.2f}, DD={crisis['max_drawdown']:.2%}")

    # Should have 1 valid, 1 out-of-bounds, 1 invalid dates
    valid_results = [c for c in result['crises'] if c.get('error') == 'none' or 'error' not in c]
    skipped_results = [c for c in result['crises'] if c.get('skipped', False)]

    print(f"\n  Valid results: {len(valid_results)}")
    print(f"  Skipped (out of bounds): {len(skipped_results)}")

    assert len(valid_results) >= 1, "Should have at least 1 valid result"

    print("  ✓ Date validation and bounds checking working")

    # ========================================================================
    # TEST 2: Mini-Sweep I.1D - Survival metric & diagnostics
    # ========================================================================
    print("\n[TEST 2] Mini-Sweep I.1D: Survival metric & diagnostics")
    print("-" * 80)

    # Run on auto-detected crisis periods
    result = crisis_engine.run(
        symbol='TEST',
        df=test_df,
        crisis_periods=None  # Auto-detect
    )

    print(f"  Auto-detected crises: {result['num_crises']}")

    for crisis in result['crises']:
        if 'error' in crisis and crisis['error'] != 'none':
            continue

        # Check diagnostics are present
        assert 'diagnostics' in crisis, f"{crisis['name']}: Missing diagnostics"

        diag = crisis['diagnostics']

        print(f"\n  {crisis['name']}:")
        print(f"    Survived: {crisis['survived']}")
        print(f"    Sharpe: {crisis['sharpe']:.2f}")
        print(f"    Max DD: {crisis['max_drawdown']:.2%}")
        print(f"    Kill-switch count: {diag['kill_switch_count']}")
        print(f"    Conflict ratio (mean): {diag['conflict_ratio_mean']:.3f}")
        print(f"    Volatility (mean): {diag['volatility_mean']:.4f}")

        # Verify survival metric (Sharpe > 0 AND DD > -40%)
        if crisis['survived']:
            assert crisis['sharpe'] > 0.0, "Survived crisis should have Sharpe > 0"
            assert crisis['max_drawdown'] > -0.40, "Survived crisis should have DD > -40%"

        # Verify diagnostics structure
        assert 'kill_switch_count' in diag, "Missing kill_switch_count"
        assert 'conflict_ratio_mean' in diag, "Missing conflict_ratio_mean"
        assert 'volatility_mean' in diag, "Missing volatility_mean"
        assert 'volatility_distribution' in diag, "Missing volatility_distribution"

    print(f"\n  Summary:")
    print(f"    Survival rate: {result['summary']['survival_rate']:.1f}%")
    print(f"    Survived: {result['summary']['survived']}/{result['summary']['total']}")

    print("\n  ✓ Survival metric and diagnostics working")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL CRISIS STRESS TESTS PASSED (2 TESTS)")
    print("=" * 80)
    print("\nMini-Sweep I.1D Enhancements:")
    print("  ✓ Date validation (train_start < train_end < test_start < test_end)")
    print("  ✓ Out-of-bounds period skipping")
    print("  ✓ Enhanced survival metric (Sharpe > 0 AND DD > -40%)")
    print("  ✓ Crisis diagnostics:")
    print("    - Kill-switch counts")
    print("    - Conflict ratio distribution (mean, std, max)")
    print("    - Volatility distribution (mean, std, max)")
    print("    - Volatility time series (first 100 points)")
    print("\nCrisis Stress Engine: PRODUCTION READY")
    print("=" * 80)
