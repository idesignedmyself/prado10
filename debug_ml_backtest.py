#!/usr/bin/env python3
"""
Debug script to trace ML backtest execution.
"""

import yfinance as yf
import sys

# Test imports first
try:
    from src.afml_system.backtest import BacktestConfig
    from src.afml_system.backtest import evo_backtest_standard
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Load minimal data
print("\n1. Loading data...")
data = yf.download('QQQ', start='2023-01-01', end='2023-12-31', progress=False)

# Fix columns
cols = data.columns
if isinstance(cols[0], tuple):
    data.columns = [str(col[0]).lower() for col in cols]
else:
    data.columns = [str(col).lower() for col in cols]

print(f"   Loaded {len(data)} bars")

# Test 1: WITHOUT ML
print("\n2. Running backtest WITHOUT ML...")
config_no_ml = BacktestConfig(
    symbol='QQQ',
    random_seed=42,
    enable_ml_fusion=False
)

try:
    result_no_ml = evo_backtest_standard('QQQ', data, config=config_no_ml)
    print(f"   Status: {result_no_ml['status']}")
    if result_no_ml['status'] == 'success':
        res = result_no_ml['result']
        print(f"   Trades: {res.total_trades}")
        print(f"   Return: {res.total_return:.2%}")
        print(f"   Sharpe: {res.sharpe_ratio:.3f}")
    else:
        print(f"   Error: {result_no_ml.get('error', 'Unknown')}")
except Exception as e:
    print(f"   Exception: {e}")
    import traceback
    traceback.print_exc()

# Test 2: WITH ML
print("\n3. Running backtest WITH ML...")
config_with_ml = BacktestConfig(
    symbol='QQQ',
    random_seed=42,
    enable_ml_fusion=True
)

try:
    result_with_ml = evo_backtest_standard('QQQ', data, config=config_with_ml)
    print(f"   Status: {result_with_ml['status']}")
    if result_with_ml['status'] == 'success':
        res = result_with_ml['result']
        print(f"   Trades: {res.total_trades}")
        print(f"   Return: {res.total_return:.2%}")
        print(f"   Sharpe: {res.sharpe_ratio:.3f}")
    else:
        print(f"   Error: {result_with_ml.get('error', 'Unknown')}")
except Exception as e:
    print(f"   Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Debug complete")
