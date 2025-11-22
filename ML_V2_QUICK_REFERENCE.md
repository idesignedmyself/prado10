# ML V2 Quick Reference Card

## Training Models

```bash
# Train V2 models (24 features)
prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024
prado train-ml-v2 SPY start 01 01 2015 end 12 31 2024

# Verify models
ls ~/.prado/models/QQQ/ml_v2/
```

## Using V2 Models

### Programmatic
```python
from afml_system.backtest import BacktestConfig, BacktestEngine

config = BacktestConfig(
    symbol='QQQ',
    enable_ml_fusion=True,
    use_ml_features_v2=True,  # Enable V2
    ml_weight=0.30,
    ml_horizon_mode='adaptive'
)

engine = BacktestEngine(config)
results = engine.run()
```

### V1 vs V2 Comparison
```python
# V1 (original, 9 features) - DEFAULT
config_v1 = BacktestConfig(
    symbol='QQQ',
    enable_ml_fusion=True,
    use_ml_features_v2=False  # or omit (default)
)

# V2 (enhanced, 24 features)
config_v2 = BacktestConfig(
    symbol='QQQ',
    enable_ml_fusion=True,
    use_ml_features_v2=True
)
```

## Features Overview

### V1 Features (9)
1. ret_1d, ret_3d, ret_5d - Returns
2. vol_20, vol_60, vol_ratio - Volatility
3. ma_ratio - Moving average ratio
4. dist_ma_20, dist_ma_50 - MA distance

### V2 Additional Features (15)

**Momentum (4)**
10. rsi_14 - RSI
11. roc_10 - Rate of Change
12. momentum_zscore_20 - Momentum z-score
13. stochastic_k - Stochastic

**Volatility (4)**
14. atr_14 - ATR
15. bb_width_20 - Bollinger width
16. hv_10 - Historical vol
17. vol_change_5 - Vol change

**Trend (4)**
18. trend_slope_20 - Trend 20d
19. trend_slope_50 - Trend 50d
20. macd_line - MACD line
21. macd_hist - MACD histogram

**Volume (3)**
22. vol_rel_20 - Relative volume
23. vol_accel_5 - Volume acceleration
24. obv_change - OBV change

## Models Trained

### Horizon Models (4)
- ml_horizon_1d_v2.pkl - Short-term (1 day)
- ml_horizon_3d_v2.pkl - Swing (3 days)
- ml_horizon_5d_v2.pkl - Medium (5 days)
- ml_horizon_10d_v2.pkl - Long-term (10 days)

### Regime Models (20 max)
- 5 regimes × 4 horizons
- trend_up, trend_down, choppy, high_vol, low_vol
- ml_regime_{regime}_{horizon}_v2.pkl

## File Locations

```
~/.prado/models/{symbol}/
├── ml_horizons/          # V1 models (9 features)
│   └── h_{horizon}.joblib
└── ml_v2/                # V2 models (24 features)
    ├── ml_horizon_{horizon}_v2.pkl
    ├── ml_regime_{regime}_{horizon}_v2.pkl
    └── training_metadata_v2.pkl
```

## Key Differences: V1 vs V2

| Aspect | V1 | V2 |
|--------|----|----|
| Features | 9 | 24 |
| Horizon Labels | Same for all | Different per horizon |
| Regime Labels | No | Yes (5 regimes) |
| Model Directory | ml_horizons/ | ml_v2/ |
| File Extension | .joblib | .pkl |
| Backward Compatible | Yes | Yes |
| Default | Yes | No (opt-in) |

## Troubleshooting

### Models Not Loading
```bash
# Check if models exist
ls ~/.prado/models/QQQ/ml_v2/

# Retrain if missing
prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024
```

### Import Errors
```bash
# Reinstall package
pip install -e .
```

### Check Config
```python
# In backtest_engine.py
config.use_ml_features_v2  # Should be True for V2
```

## Expected Improvements

**V1 Diagnostic Results**
- All parameters → Sharpe=1.993, Trades=88 (constant)
- No parameter sensitivity

**V2 Expected Results**
- Parameter sweep produces varied results
- ml_weight=0.15 ≠ ml_weight=0.45
- ml_horizon_mode='1d' ≠ ml_horizon_mode='10d'
- Models make different predictions

## Quick Commands

```bash
# Train V2
prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024

# Check models
ls -lh ~/.prado/models/QQQ/ml_v2/

# Count models
ls ~/.prado/models/QQQ/ml_v2/*.pkl | wc -l

# Check metadata
python -c "import joblib; print(joblib.load('~/.prado/models/QQQ/ml_v2/training_metadata_v2.pkl'))"
```

---

**Status**: COMPLETE ✅
**Date**: 2025-11-21
**Version**: PRADO9_EVO v1.3
