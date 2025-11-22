#!/usr/bin/env python3
"""
ML Parameter Sweep - Complete Optimization

Systematically tests all ML fusion parameters:
1. Confidence thresholds: 0.55, 0.60, 0.65, 0.70
2. Meta-labeling modes: rules_priority, ml_priority, balanced_blend
3. Horizon modes: 1d, 3d, 5d, 10d, adaptive
4. ML weights: 0.15, 0.25, 0.35
"""

import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from src.afml_system.backtest import BacktestConfig, evo_backtest_standard


@dataclass
class SweepResult:
    config_name: str
    params: Dict
    total_return: float
    sharpe: float
    sortino: float
    max_dd: float
    total_trades: int
    win_rate: float

    def score(self) -> float:
        """Sharpe / sqrt(trades/50) - penalize overtrading"""
        trade_penalty = np.sqrt(max(1.0, self.total_trades / 50.0))
        return self.sharpe / trade_penalty


def load_data():
    """Load QQQ data"""
    data = yf.download('QQQ', start='2020-01-01', end='2025-11-21', progress=False)
    cols = data.columns
    if isinstance(cols[0], tuple):
        data.columns = [str(col[0]).lower() for col in cols]
    else:
        data.columns = [str(col).lower() for col in cols]
    return data


def run_backtest(params: Dict, data: pd.DataFrame) -> SweepResult:
    """Run backtest with given parameters"""
    config = BacktestConfig(
        symbol='QQQ',
        random_seed=42,
        enable_ml_fusion=True,
        ml_conf_threshold=params.get('conf_threshold', 0.05),
        ml_weight=params.get('ml_weight', 0.25),
        ml_meta_mode=params.get('meta_mode', 'balanced_blend'),
        ml_horizon_mode=params.get('horizon_mode', '1d')
    )

    response = evo_backtest_standard('QQQ', data, config=config)

    if response['status'] != 'success':
        raise RuntimeError(f"Backtest failed: {response.get('error')}")

    result = response['result']

    return SweepResult(
        config_name=str(params),
        params=params,
        total_return=result.total_return,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        max_dd=result.max_drawdown,
        total_trades=result.total_trades,
        win_rate=result.win_rate
    )


def main():
    """Run comprehensive parameter sweep"""
    print("="*80)
    print("ML PARAMETER SWEEP - PRADO9_EVO v1.2")
    print("="*80)

    data = load_data()
    print(f"Loaded {len(data)} bars\n")

    # Get baseline
    print("Running BASELINE (no ML)...")
    baseline_config = BacktestConfig(symbol='QQQ', random_seed=42, enable_ml_fusion=False)
    baseline_response = evo_backtest_standard('QQQ', data, config=baseline_config)
    baseline = baseline_response['result']
    print(f"Baseline: Return={baseline.total_return:.2%}, Sharpe={baseline.sharpe_ratio:.3f}, Trades={baseline.total_trades}\n")

    # SWEEP 1: Confidence Thresholds
    print("="*80)
    print("SWEEP 1: CONFIDENCE THRESHOLDS")
    print("="*80)

    conf_results = []
    for threshold in [0.03, 0.05, 0.08, 0.10]:
        params = {
            'conf_threshold': threshold,
            'ml_weight': 0.25,
            'meta_mode': 'balanced_blend',
            'horizon_mode': '1d'
        }
        print(f"Testing conf_threshold={threshold:.2f}...")
        result = run_backtest(params, data)
        conf_results.append(result)
        print(f"  Sharpe={result.sharpe:.3f}, Trades={result.total_trades}, Score={result.score():.3f}")

    best_conf = max(conf_results, key=lambda r: r.score())
    print(f"\nBest Confidence Threshold: {best_conf.params['conf_threshold']:.2f}")
    print(f"  Score: {best_conf.score():.3f}, Sharpe: {best_conf.sharpe:.3f}\n")

    # SWEEP 2: Meta-labeling Modes
    print("="*80)
    print("SWEEP 2: META-LABELING MODES")
    print("="*80)

    meta_results = []
    for mode in ['rules_priority', 'ml_priority', 'balanced_blend']:
        params = {
            'conf_threshold': best_conf.params['conf_threshold'],
            'ml_weight': 0.25,
            'meta_mode': mode,
            'horizon_mode': '1d'
        }
        print(f"Testing meta_mode={mode}...")
        result = run_backtest(params, data)
        meta_results.append(result)
        print(f"  Sharpe={result.sharpe:.3f}, Trades={result.total_trades}, Score={result.score():.3f}")

    best_meta = max(meta_results, key=lambda r: r.score())
    print(f"\nBest Meta-labeling Mode: {best_meta.params['meta_mode']}")
    print(f"  Score: {best_meta.score():.3f}, Sharpe: {best_meta.sharpe:.3f}\n")

    # SWEEP 3: Horizon Modes
    print("="*80)
    print("SWEEP 3: HORIZON MODES")
    print("="*80)

    horizon_results = []
    for horizon in ['1d', '3d', '5d', '10d', 'adaptive']:
        params = {
            'conf_threshold': best_conf.params['conf_threshold'],
            'ml_weight': 0.25,
            'meta_mode': best_meta.params['meta_mode'],
            'horizon_mode': horizon
        }
        print(f"Testing horizon_mode={horizon}...")
        result = run_backtest(params, data)
        horizon_results.append(result)
        print(f"  Sharpe={result.sharpe:.3f}, Trades={result.total_trades}, Score={result.score():.3f}")

    best_horizon = max(horizon_results, key=lambda r: r.score())
    print(f"\nBest Horizon Mode: {best_horizon.params['horizon_mode']}")
    print(f"  Score: {best_horizon.score():.3f}, Sharpe: {best_horizon.sharpe:.3f}\n")

    # SWEEP 4: ML Weights
    print("="*80)
    print("SWEEP 4: ML WEIGHTS")
    print("="*80)

    weight_results = []
    for weight in [0.15, 0.25, 0.35, 0.45]:
        params = {
            'conf_threshold': best_conf.params['conf_threshold'],
            'ml_weight': weight,
            'meta_mode': best_meta.params['meta_mode'],
            'horizon_mode': best_horizon.params['horizon_mode']
        }
        print(f"Testing ml_weight={weight:.2f}...")
        result = run_backtest(params, data)
        weight_results.append(result)
        print(f"  Sharpe={result.sharpe:.3f}, Trades={result.total_trades}, Score={result.score():.3f}")

    best_weight = max(weight_results, key=lambda r: r.score())
    print(f"\nBest ML Weight: {best_weight.params['ml_weight']:.2f}")
    print(f"  Score: {best_weight.score():.3f}, Sharpe: {best_weight.sharpe:.3f}\n")

    # FINAL RECOMMENDATION
    print("="*80)
    print("FINAL RECOMMENDED CONFIGURATION")
    print("="*80)
    print(f"\nml_conf_threshold = {best_conf.params['conf_threshold']:.2f}")
    print(f"ml_weight = {best_weight.params['ml_weight']:.2f}")
    print(f"ml_meta_mode = '{best_meta.params['meta_mode']}'")
    print(f"ml_horizon_mode = '{best_horizon.params['horizon_mode']}'")

    # Run final optimized backtest
    print("\n" + "="*80)
    print("FINAL VALIDATION")
    print("="*80)

    optimal_params = {
        'conf_threshold': best_conf.params['conf_threshold'],
        'ml_weight': best_weight.params['ml_weight'],
        'meta_mode': best_meta.params['meta_mode'],
        'horizon_mode': best_horizon.params['horizon_mode']
    }

    print("\nRunning optimized ML configuration...")
    optimized = run_backtest(optimal_params, data)

    print("\nBASELINE (No ML)")
    print(f"  Total Return:  {baseline.total_return:>8.2%}")
    print(f"  Sharpe Ratio:  {baseline.sharpe_ratio:>8.3f}")
    print(f"  Sortino Ratio: {baseline.sortino_ratio:>8.3f}")
    print(f"  Max Drawdown:  {baseline.max_drawdown:>8.2%}")
    print(f"  Total Trades:  {baseline.total_trades:>8d}")

    print("\nOPTIMIZED ML")
    print(f"  Total Return:  {optimized.total_return:>8.2%}")
    print(f"  Sharpe Ratio:  {optimized.sharpe:>8.3f}")
    print(f"  Sortino Ratio: {optimized.sortino:>8.3f}")
    print(f"  Max Drawdown:  {optimized.max_dd:>8.2%}")
    print(f"  Total Trades:  {optimized.total_trades:>8d}")

    sharpe_improvement = (optimized.sharpe / baseline.sharpe_ratio - 1) * 100
    print(f"\nSharpe Improvement: {sharpe_improvement:+.1f}%")

    print("\n" + "="*80)
    print("SWEEP COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
