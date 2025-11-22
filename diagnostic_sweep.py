#!/usr/bin/env python3
"""
PRADO9_EVO Diagnostic Sweep - ML + Allocator + Fusion Audit
============================================================

Purpose: Forensic testing to identify dead branches and verify all ML parameters
         actually influence final trading decisions.

This suite tests 8 critical pathways to determine:
1. Does ML influence final decisions?
2. Is the confidence threshold used?
3. Is meta-labeling used?
4. Do horizon choices matter?
5. Do regime choices matter?
6. Do ML weights matter?
7. Do rule signals override ML when rules_priority is active?
8. Does the allocator use ML fusion output?
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any

from src.afml_system.backtest import BacktestConfig, evo_backtest_standard

print("=" * 80)
print("PRADO9_EVO DIAGNOSTIC SWEEP")
print("=" * 80)
print()

# Load test data
print("Loading QQQ data...")
data = yf.download('QQQ', start='2020-01-01', end='2025-11-21', progress=False)
cols = data.columns
if isinstance(cols[0], tuple):
    data.columns = [str(col[0]).lower() for col in cols]
else:
    data.columns = [str(col).lower() for col in cols]
print(f"Loaded {len(data)} bars\n")

results = {}

# -------------------------------------------------------------
# TEST 1 — ML Disabled vs ML Enabled (Baseline Check)
# -------------------------------------------------------------
def test_1_ml_on_off():
    """Verify ML actually changes results when enabled"""
    print("TEST 1: ML Disabled vs ML Enabled")
    print("-" * 80)

    # Run without ML
    config_off = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=False
    )
    response_off = evo_backtest_standard('QQQ', data, config=config_off)
    result_off = response_off['result']

    # Run with ML
    config_on = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True
    )
    response_on = evo_backtest_standard('QQQ', data, config=config_on)
    result_on = response_on['result']

    # Compare
    sharpe_diff = result_on.sharpe_ratio - result_off.sharpe_ratio
    trades_diff = result_on.total_trades - result_off.total_trades
    return_diff = result_on.total_return - result_off.total_return

    print(f"  ML OFF: Sharpe={result_off.sharpe_ratio:.3f}, Trades={result_off.total_trades}, Return={result_off.total_return:.2%}")
    print(f"  ML ON:  Sharpe={result_on.sharpe_ratio:.3f}, Trades={result_on.total_trades}, Return={result_on.total_return:.2%}")
    print(f"  DELTA:  Sharpe={sharpe_diff:+.3f}, Trades={trades_diff:+d}, Return={return_diff:+.2%}")

    # Verdict
    if abs(sharpe_diff) > 0.01 or abs(trades_diff) > 0:
        print("  ✅ PASS - ML changes results")
        verdict = "PASS"
    else:
        print("  ❌ FAIL - ML has no effect (dead branch)")
        verdict = "FAIL"

    print()
    return {
        'verdict': verdict,
        'ml_off': {'sharpe': result_off.sharpe_ratio, 'trades': result_off.total_trades},
        'ml_on': {'sharpe': result_on.sharpe_ratio, 'trades': result_on.total_trades},
        'delta': {'sharpe': sharpe_diff, 'trades': trades_diff}
    }


# -------------------------------------------------------------
# TEST 2 — ML Weight Sensitivity (0.15 vs 0.45)
# -------------------------------------------------------------
def test_2_ml_weight():
    """Verify ml_weight parameter actually influences results"""
    print("TEST 2: ML Weight Sensitivity")
    print("-" * 80)

    # Low ML weight
    config_low = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_weight=0.15
    )
    response_low = evo_backtest_standard('QQQ', data, config=config_low)
    result_low = response_low['result']

    # High ML weight
    config_high = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_weight=0.45
    )
    response_high = evo_backtest_standard('QQQ', data, config=config_high)
    result_high = response_high['result']

    # Compare
    sharpe_diff = result_high.sharpe_ratio - result_low.sharpe_ratio
    trades_diff = result_high.total_trades - result_low.total_trades

    print(f"  ML Weight 0.15: Sharpe={result_low.sharpe_ratio:.3f}, Trades={result_low.total_trades}")
    print(f"  ML Weight 0.45: Sharpe={result_high.sharpe_ratio:.3f}, Trades={result_high.total_trades}")
    print(f"  DELTA:          Sharpe={sharpe_diff:+.3f}, Trades={trades_diff:+d}")

    # Verdict
    if abs(sharpe_diff) > 0.01 or abs(trades_diff) > 0:
        print("  ✅ PASS - ml_weight parameter is active")
        verdict = "PASS"
    else:
        print("  ❌ FAIL - ml_weight has no effect (parameter not used)")
        verdict = "FAIL"

    print()
    return {
        'verdict': verdict,
        'weight_0.15': {'sharpe': result_low.sharpe_ratio, 'trades': result_low.total_trades},
        'weight_0.45': {'sharpe': result_high.sharpe_ratio, 'trades': result_high.total_trades},
        'delta': {'sharpe': sharpe_diff, 'trades': trades_diff}
    }


# -------------------------------------------------------------
# TEST 3 — Confidence Threshold Sensitivity (0.03 vs 0.10)
# -------------------------------------------------------------
def test_3_conf_threshold():
    """Verify ml_conf_threshold parameter filters ML contributions"""
    print("TEST 3: Confidence Threshold Sensitivity")
    print("-" * 80)

    # Low threshold (more ML signals)
    config_low = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_conf_threshold=0.03
    )
    response_low = evo_backtest_standard('QQQ', data, config=config_low)
    result_low = response_low['result']

    # High threshold (fewer ML signals)
    config_high = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_conf_threshold=0.10
    )
    response_high = evo_backtest_standard('QQQ', data, config=config_high)
    result_high = response_high['result']

    # Compare
    sharpe_diff = result_high.sharpe_ratio - result_low.sharpe_ratio
    trades_diff = result_high.total_trades - result_low.total_trades

    print(f"  Threshold 0.03: Sharpe={result_low.sharpe_ratio:.3f}, Trades={result_low.total_trades}")
    print(f"  Threshold 0.10: Sharpe={result_high.sharpe_ratio:.3f}, Trades={result_high.total_trades}")
    print(f"  DELTA:          Sharpe={sharpe_diff:+.3f}, Trades={trades_diff:+d}")

    # Verdict
    if abs(sharpe_diff) > 0.01 or abs(trades_diff) > 0:
        print("  ✅ PASS - ml_conf_threshold parameter is active")
        verdict = "PASS"
    else:
        print("  ❌ FAIL - ml_conf_threshold has no effect (threshold not enforced)")
        verdict = "FAIL"

    print()
    return {
        'verdict': verdict,
        'threshold_0.03': {'sharpe': result_low.sharpe_ratio, 'trades': result_low.total_trades},
        'threshold_0.10': {'sharpe': result_high.sharpe_ratio, 'trades': result_high.total_trades},
        'delta': {'sharpe': sharpe_diff, 'trades': trades_diff}
    }


# -------------------------------------------------------------
# TEST 4 — Meta-Labeling Mode (rules_priority vs ml_priority)
# -------------------------------------------------------------
def test_4_meta_labeling():
    """Verify meta-labeling mode changes signal fusion behavior"""
    print("TEST 4: Meta-Labeling Mode Sensitivity")
    print("-" * 80)

    # Rules priority (strong rules protected)
    config_rules = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_meta_mode='rules_priority'
    )
    response_rules = evo_backtest_standard('QQQ', data, config=config_rules)
    result_rules = response_rules['result']

    # ML priority (ML can override)
    config_ml = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_meta_mode='ml_priority'
    )
    response_ml = evo_backtest_standard('QQQ', data, config=config_ml)
    result_ml = response_ml['result']

    # Compare
    sharpe_diff = result_ml.sharpe_ratio - result_rules.sharpe_ratio
    trades_diff = result_ml.total_trades - result_rules.total_trades

    print(f"  rules_priority: Sharpe={result_rules.sharpe_ratio:.3f}, Trades={result_rules.total_trades}")
    print(f"  ml_priority:    Sharpe={result_ml.sharpe_ratio:.3f}, Trades={result_ml.total_trades}")
    print(f"  DELTA:          Sharpe={sharpe_diff:+.3f}, Trades={trades_diff:+d}")

    # Verdict
    if abs(sharpe_diff) > 0.01 or abs(trades_diff) > 0:
        print("  ✅ PASS - ml_meta_mode parameter is active")
        verdict = "PASS"
    else:
        print("  ❌ FAIL - ml_meta_mode has no effect (meta-labeling not implemented)")
        verdict = "FAIL"

    print()
    return {
        'verdict': verdict,
        'rules_priority': {'sharpe': result_rules.sharpe_ratio, 'trades': result_rules.total_trades},
        'ml_priority': {'sharpe': result_ml.sharpe_ratio, 'trades': result_ml.total_trades},
        'delta': {'sharpe': sharpe_diff, 'trades': trades_diff}
    }


# -------------------------------------------------------------
# TEST 5 — Horizon Mode (1d vs 10d vs adaptive)
# -------------------------------------------------------------
def test_5_horizon_mode():
    """Verify horizon mode selection influences predictions"""
    print("TEST 5: Horizon Mode Sensitivity")
    print("-" * 80)

    # 1-day horizon
    config_1d = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_horizon_mode='1d'
    )
    response_1d = evo_backtest_standard('QQQ', data, config=config_1d)
    result_1d = response_1d['result']

    # 10-day horizon
    config_10d = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_horizon_mode='10d'
    )
    response_10d = evo_backtest_standard('QQQ', data, config=config_10d)
    result_10d = response_10d['result']

    # Adaptive horizon
    config_adaptive = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_horizon_mode='adaptive'
    )
    response_adaptive = evo_backtest_standard('QQQ', data, config=config_adaptive)
    result_adaptive = response_adaptive['result']

    # Compare
    sharpe_diff_1d_10d = result_10d.sharpe_ratio - result_1d.sharpe_ratio
    sharpe_diff_adaptive = result_adaptive.sharpe_ratio - result_1d.sharpe_ratio
    trades_diff_1d_10d = result_10d.total_trades - result_1d.total_trades
    trades_diff_adaptive = result_adaptive.total_trades - result_1d.total_trades

    print(f"  Horizon 1d:       Sharpe={result_1d.sharpe_ratio:.3f}, Trades={result_1d.total_trades}")
    print(f"  Horizon 10d:      Sharpe={result_10d.sharpe_ratio:.3f}, Trades={result_10d.total_trades}")
    print(f"  Horizon adaptive: Sharpe={result_adaptive.sharpe_ratio:.3f}, Trades={result_adaptive.total_trades}")
    print(f"  DELTA (1d→10d):   Sharpe={sharpe_diff_1d_10d:+.3f}, Trades={trades_diff_1d_10d:+d}")
    print(f"  DELTA (1d→adapt): Sharpe={sharpe_diff_adaptive:+.3f}, Trades={trades_diff_adaptive:+d}")

    # Verdict
    if abs(sharpe_diff_1d_10d) > 0.01 or abs(trades_diff_1d_10d) > 0:
        print("  ✅ PASS - ml_horizon_mode parameter is active")
        verdict = "PASS"
    else:
        print("  ❌ FAIL - ml_horizon_mode has no effect (horizon selection not working)")
        verdict = "FAIL"

    print()
    return {
        'verdict': verdict,
        'horizon_1d': {'sharpe': result_1d.sharpe_ratio, 'trades': result_1d.total_trades},
        'horizon_10d': {'sharpe': result_10d.sharpe_ratio, 'trades': result_10d.total_trades},
        'horizon_adaptive': {'sharpe': result_adaptive.sharpe_ratio, 'trades': result_adaptive.total_trades},
        'delta_1d_10d': {'sharpe': sharpe_diff_1d_10d, 'trades': trades_diff_1d_10d},
        'delta_adaptive': {'sharpe': sharpe_diff_adaptive, 'trades': trades_diff_adaptive}
    }


# -------------------------------------------------------------
# TEST 6 — Extreme ML Weight (0.0 vs 0.99)
# -------------------------------------------------------------
def test_6_extreme_weights():
    """Test extreme ML weights to force maximum differentiation"""
    print("TEST 6: Extreme ML Weights (0.0 vs 0.99)")
    print("-" * 80)

    # ML weight = 0 (pure rules)
    config_zero = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_weight=0.0
    )
    response_zero = evo_backtest_standard('QQQ', data, config=config_zero)
    result_zero = response_zero['result']

    # ML weight = 0.99 (almost pure ML)
    config_max = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_weight=0.99
    )
    response_max = evo_backtest_standard('QQQ', data, config=config_max)
    result_max = response_max['result']

    # Compare
    sharpe_diff = result_max.sharpe_ratio - result_zero.sharpe_ratio
    trades_diff = result_max.total_trades - result_zero.total_trades
    return_diff = result_max.total_return - result_zero.total_return

    print(f"  ML Weight 0.00: Sharpe={result_zero.sharpe_ratio:.3f}, Trades={result_zero.total_trades}, Return={result_zero.total_return:.2%}")
    print(f"  ML Weight 0.99: Sharpe={result_max.sharpe_ratio:.3f}, Trades={result_max.total_trades}, Return={result_max.total_return:.2%}")
    print(f"  DELTA:          Sharpe={sharpe_diff:+.3f}, Trades={trades_diff:+d}, Return={return_diff:+.2%}")

    # Verdict
    if abs(sharpe_diff) > 0.1 or abs(trades_diff) > 5:
        print("  ✅ PASS - Extreme weights produce major differences")
        verdict = "PASS"
    else:
        print("  ❌ FAIL - Even extreme weights produce identical results (fusion broken)")
        verdict = "FAIL"

    print()
    return {
        'verdict': verdict,
        'weight_0.00': {'sharpe': result_zero.sharpe_ratio, 'trades': result_zero.total_trades},
        'weight_0.99': {'sharpe': result_max.sharpe_ratio, 'trades': result_max.total_trades},
        'delta': {'sharpe': sharpe_diff, 'trades': trades_diff, 'return': return_diff}
    }


# -------------------------------------------------------------
# TEST 7 — Extreme Confidence Threshold (0.01 vs 0.50)
# -------------------------------------------------------------
def test_7_extreme_thresholds():
    """Test extreme confidence thresholds to verify filtering logic"""
    print("TEST 7: Extreme Confidence Thresholds (0.01 vs 0.50)")
    print("-" * 80)

    # Very low threshold (accept all ML signals)
    config_low = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_conf_threshold=0.01
    )
    response_low = evo_backtest_standard('QQQ', data, config=config_low)
    result_low = response_low['result']

    # Very high threshold (reject most ML signals)
    config_high = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_conf_threshold=0.50
    )
    response_high = evo_backtest_standard('QQQ', data, config=config_high)
    result_high = response_high['result']

    # Compare
    sharpe_diff = result_high.sharpe_ratio - result_low.sharpe_ratio
    trades_diff = result_high.total_trades - result_low.total_trades

    print(f"  Threshold 0.01: Sharpe={result_low.sharpe_ratio:.3f}, Trades={result_low.total_trades}")
    print(f"  Threshold 0.50: Sharpe={result_high.sharpe_ratio:.3f}, Trades={result_high.total_trades}")
    print(f"  DELTA:          Sharpe={sharpe_diff:+.3f}, Trades={trades_diff:+d}")

    # Verdict
    if abs(sharpe_diff) > 0.05 or abs(trades_diff) > 5:
        print("  ✅ PASS - Extreme thresholds produce major differences")
        verdict = "PASS"
    else:
        print("  ❌ FAIL - Even extreme thresholds produce identical results (threshold not enforced)")
        verdict = "FAIL"

    print()
    return {
        'verdict': verdict,
        'threshold_0.01': {'sharpe': result_low.sharpe_ratio, 'trades': result_low.total_trades},
        'threshold_0.50': {'sharpe': result_high.sharpe_ratio, 'trades': result_high.total_trades},
        'delta': {'sharpe': sharpe_diff, 'trades': trades_diff}
    }


# -------------------------------------------------------------
# TEST 8 — Full Parameter Matrix (detect correlation)
# -------------------------------------------------------------
def test_8_parameter_matrix():
    """Test multiple parameter combinations to detect correlations"""
    print("TEST 8: Parameter Matrix Test")
    print("-" * 80)

    configs = [
        {'ml_weight': 0.15, 'ml_conf_threshold': 0.03, 'ml_meta_mode': 'rules_priority', 'ml_horizon_mode': '1d'},
        {'ml_weight': 0.45, 'ml_conf_threshold': 0.03, 'ml_meta_mode': 'rules_priority', 'ml_horizon_mode': '1d'},
        {'ml_weight': 0.15, 'ml_conf_threshold': 0.10, 'ml_meta_mode': 'rules_priority', 'ml_horizon_mode': '1d'},
        {'ml_weight': 0.15, 'ml_conf_threshold': 0.03, 'ml_meta_mode': 'ml_priority', 'ml_horizon_mode': '1d'},
        {'ml_weight': 0.15, 'ml_conf_threshold': 0.03, 'ml_meta_mode': 'rules_priority', 'ml_horizon_mode': '10d'},
    ]

    matrix_results = []

    for i, params in enumerate(configs):
        config = BacktestConfig(
            symbol='QQQ',
            random_seed=42,
            enable_ml_fusion=True,
            **params
        )
        response = evo_backtest_standard('QQQ', data, config=config)
        result = response['result']

        matrix_results.append({
            'config': i + 1,
            'params': params,
            'sharpe': result.sharpe_ratio,
            'trades': result.total_trades,
            'return': result.total_return
        })

        print(f"  Config {i+1}: Sharpe={result.sharpe_ratio:.3f}, Trades={result.total_trades}")

    # Check if all results are identical
    sharpes = [r['sharpe'] for r in matrix_results]
    trades = [r['trades'] for r in matrix_results]

    sharpe_variance = np.var(sharpes)
    trade_variance = np.var(trades)

    print(f"\n  Sharpe Variance: {sharpe_variance:.6f}")
    print(f"  Trade Variance:  {trade_variance:.2f}")

    # Verdict
    if sharpe_variance < 0.001 and trade_variance < 1.0:
        print("  ❌ FAIL - All configurations produce identical results (parameters disconnected)")
        verdict = "FAIL"
    else:
        print("  ✅ PASS - Configurations produce varied results")
        verdict = "PASS"

    print()
    return {
        'verdict': verdict,
        'matrix': matrix_results,
        'sharpe_variance': sharpe_variance,
        'trade_variance': trade_variance
    }


# -------------------------------------------------------------
# RUN ALL TESTS
# -------------------------------------------------------------
print("\nRunning diagnostic suite...\n")

tests = [
    ("TEST 1: ML On/Off", test_1_ml_on_off),
    ("TEST 2: ML Weight", test_2_ml_weight),
    ("TEST 3: Confidence Threshold", test_3_conf_threshold),
    ("TEST 4: Meta-Labeling", test_4_meta_labeling),
    ("TEST 5: Horizon Mode", test_5_horizon_mode),
    ("TEST 6: Extreme Weights", test_6_extreme_weights),
    ("TEST 7: Extreme Thresholds", test_7_extreme_thresholds),
    ("TEST 8: Parameter Matrix", test_8_parameter_matrix),
]

for test_name, test_fn in tests:
    try:
        result = test_fn()
        results[test_name] = result
    except Exception as e:
        print(f"❌ {test_name} CRASHED: {str(e)}\n")
        results[test_name] = {'verdict': 'CRASH', 'error': str(e)}


# -------------------------------------------------------------
# FINAL VERDICT
# -------------------------------------------------------------
print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

pass_count = sum(1 for r in results.values() if r.get('verdict') == 'PASS')
fail_count = sum(1 for r in results.values() if r.get('verdict') == 'FAIL')
crash_count = sum(1 for r in results.values() if r.get('verdict') == 'CRASH')

print(f"Tests Passed:  {pass_count}/{len(tests)}")
print(f"Tests Failed:  {fail_count}/{len(tests)}")
print(f"Tests Crashed: {crash_count}/{len(tests)}")
print()

# Analysis
if pass_count == len(tests):
    print("✅ ALL TESTS PASSED - ML system fully functional with parameter sensitivity")
elif pass_count == 0:
    print("❌ ALL TESTS FAILED - ML system completely disconnected from decision pipeline")
    print("\nPOSSIBLE CAUSES:")
    print("  1. ML fusion not called in allocator")
    print("  2. ML predictions returning constant values")
    print("  3. Fusion logic bypassed")
    print("  4. Parameters not passed to fusion engine")
else:
    print(f"⚠️ PARTIAL FAILURE - {pass_count} tests passed, {fail_count} failed")
    print("\nFAILED TESTS:")
    for test_name, result in results.items():
        if result.get('verdict') == 'FAIL':
            print(f"  - {test_name}")

    print("\nPOSSIBLE CAUSES:")
    if results.get("TEST 1: ML On/Off", {}).get('verdict') == 'FAIL':
        print("  - ML system not integrated into backtest engine")
    if results.get("TEST 2: ML Weight", {}).get('verdict') == 'FAIL':
        print("  - ml_weight parameter not used in fusion")
    if results.get("TEST 3: Confidence Threshold", {}).get('verdict') == 'FAIL':
        print("  - ml_conf_threshold not enforced")
    if results.get("TEST 4: Meta-Labeling", {}).get('verdict') == 'FAIL':
        print("  - ml_meta_mode logic not implemented")
    if results.get("TEST 5: Horizon Mode", {}).get('verdict') == 'FAIL':
        print("  - ml_horizon_mode not selecting different models")
    if results.get("TEST 8: Parameter Matrix", {}).get('verdict') == 'FAIL':
        print("  - Models returning identical predictions regardless of configuration")

print()
print("=" * 80)
print("DIAGNOSTIC SWEEP COMPLETE")
print("=" * 80)
