#!/usr/bin/env python3
"""
Quick diagnostic script to test ML activation in backtest.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

from src.afml_system.backtest import BacktestConfig, BacktestEngine

# Load data
print("Loading QQQ data...")
data = yf.download('QQQ', start='2020-01-01', end='2025-11-21', progress=False)

# Fix columns
cols = data.columns
if isinstance(cols[0], tuple):
    data.columns = [str(col[0]).lower() for col in cols]
else:
    data.columns = [str(col).lower() for col in cols]

print(f"Loaded {len(data)} bars\n")

# Test 1: Backtest WITHOUT ML
print("="*60)
print("TEST 1: Backtest WITHOUT ML (baseline)")
print("="*60)

config_no_ml = BacktestConfig(
    symbol='QQQ',
    random_seed=42,
    enable_ml_fusion=False
)

engine_no_ml = BacktestEngine(config=config_no_ml)
print(f"ML models loaded: {engine_no_ml.ml_horizon_models is not None}")
print(f"ML fusion loaded: {engine_no_ml.ml_fusion is not None}")

# Test 2: Backtest WITH ML
print("\n" + "="*60)
print("TEST 2: Backtest WITH ML enabled")
print("="*60)

config_with_ml = BacktestConfig(
    symbol='QQQ',
    random_seed=42,
    enable_ml_fusion=True
)

engine_with_ml = BacktestEngine(config=config_with_ml)
print(f"ML models loaded: {engine_with_ml.ml_horizon_models is not None}")
print(f"ML fusion loaded: {engine_with_ml.ml_fusion is not None}")

if engine_with_ml.ml_horizon_models:
    print(f"Horizon models: {list(engine_with_ml.ml_horizon_models.keys())}")

# Test ML prediction on a sample window
print("\n" + "="*60)
print("TEST 3: ML Prediction on sample window")
print("="*60)

window = data.tail(100)
print(f"Window size: {len(window)} bars")

# Test horizon prediction
horizon = '1d'
ml_h_signal, ml_h_conf = engine_with_ml._get_ml_horizon_prediction(window, horizon)
print(f"\nHorizon '{horizon}' prediction:")
print(f"  Signal: {ml_h_signal} ({'+1=up, -1=down, 0=neutral'})")
print(f"  Confidence: {ml_h_conf:.3f}")

# Test regime prediction
regime = 'trend_up'
ml_r_signal, ml_r_conf = engine_with_ml._get_ml_regime_prediction(window, regime, horizon)
print(f"\nRegime '{regime}' prediction:")
print(f"  Signal: {ml_r_signal}")
print(f"  Confidence: {ml_r_conf:.3f}")

# Test fusion
from src.afml_system.ml import HybridMLFusion

fusion = HybridMLFusion()
rule_signal = 0.6  # Simulated rule-based signal

fused_signal, diagnostics = fusion.fuse(
    rule_signal=rule_signal,
    ml_horizon_signal=ml_h_signal,
    ml_regime_signal=ml_r_signal,
    ml_horizon_conf=ml_h_conf,
    ml_regime_conf=ml_r_conf,
    ml_weight=0.25
)

print(f"\nFusion test:")
print(f"  Rule signal: {rule_signal:.3f}")
print(f"  ML vote: {diagnostics['ml_vote']:.3f}")
print(f"  Fused signal: {fused_signal:.3f}")

print("\n" + "="*60)
print("All tests complete!")
print("="*60)
