#!/usr/bin/env python
"""
Quick script to test ML V2 backtest
"""

from afml_system.backtest import BacktestConfig, BacktestEngine

# ML V2 Configuration
config = BacktestConfig(
    symbol='QQQ',
    enable_ml_fusion=True,
    use_ml_features_v2=True,  # Enable V2
    seed=42
)

print("\n📊 Running ML V2 Backtest (24 features)...\n")

engine = BacktestEngine(config)
results = engine.run()

# Print results
print("\n" + "="*50)
print("ML V2 BACKTEST RESULTS")
print("="*50)
print(f"Total Return:  {results.total_return*100:.2f}%")
print(f"Sharpe Ratio:  {results.sharpe_ratio:.3f}")
print(f"Sortino Ratio: {results.sortino_ratio:.3f}")
print(f"Calmar Ratio:  {results.calmar_ratio:.3f}")
print(f"Max Drawdown:  {results.max_drawdown*100:.2f}%")
print(f"Win Rate:      {results.win_rate*100:.2f}%")
print(f"Profit Factor: {results.profit_factor:.2f}")
print(f"Total Trades:  {results.total_trades}")
print("="*50)
