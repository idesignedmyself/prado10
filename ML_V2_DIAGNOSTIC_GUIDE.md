# ML V2 Diagnostic Suite - User Guide

**Purpose**: Validate ML V2 models after training

**Status**: Production Ready ✅

**Date**: 2025-11-21

---

## Overview

The ML V2 Diagnostic Suite provides comprehensive validation of your trained ML V2 models. It runs 9 different tests to ensure:

- Features are correctly built (24 features)
- Targets are properly constructed
- Models load successfully
- Predictions are valid and diverse
- V2 improvements are working as expected

---

## Usage

### Basic Command

```bash
prado ml-v2-diagnostic SYMBOL start MM DD YYYY end MM DD YYYY
```

### Examples

```bash
# Validate QQQ models
prado ml-v2-diagnostic QQQ start 01 01 2020 end 12 31 2024

# Validate SPY models
prado ml-v2-diagnostic SPY start 01 01 2015 end 12 31 2024
```

---

## Diagnostic Tests

### Test 1: Feature Integrity ✅

**What it checks**:
- Number of features (should be 24)
- Feature shape
- NaN count
- Feature names

**Pass Criteria**: Exactly 24 features built

**Example Output**:
```
- Expected Features: 24
- Actual Features: 24
- Shape: (1000, 24)
- NaN Count: 0
- Status: ✅ PASS
```

---

### Test 2: Target Integrity ✅

**What it checks**:
- Horizon labels built correctly
- Regime labels built correctly
- Label shapes
- Column names

**Pass Criteria**: Both label types created successfully

**Example Output**:
```
- Horizon Labels Shape: (1000, 4)
- Regime Labels Shape: (1000, 6)
- Horizon Columns: ['label_up_1d', 'label_up_3d', 'label_up_5d', 'label_up_10d']
- Status: ✅ PASS
```

---

### Test 3: Model Loading ✅

**What it checks**:
- All 4 horizon models load (1d, 3d, 5d, 10d)
- Sample of regime models load
- Model files exist in correct locations

**Pass Criteria**: Models successfully loaded from disk

**Example Output**:
```
Horizon Models:
- 1d: ✅ LOADED
- 3d: ✅ LOADED
- 5d: ✅ LOADED
- 10d: ✅ LOADED

Regime Models (Sample):
- trend_up×1d: ✅ EXISTS
- trend_up×5d: ✅ EXISTS
- trend_down×1d: ✅ EXISTS
- trend_down×5d: ✅ EXISTS
```

---

### Test 4: Horizon Predictions ✅

**What it checks**:
- Each horizon model generates predictions
- Predictions are valid (-1 or +1)
- Confidence values are reasonable (0 to 1)

**Pass Criteria**: All models predict successfully

**Example Output**:
```
- 1d: signal=+1, confidence=0.6234 ✅
- 3d: signal=-1, confidence=0.5823 ✅
- 5d: signal=+1, confidence=0.7145 ✅
- 10d: signal=+1, confidence=0.6891 ✅
```

---

### Test 5: Regime Predictions ✅

**What it checks**:
- Regime-specific models predict correctly
- Current regime is detected
- Regime models load and execute

**Pass Criteria**: Regime models generate valid predictions

**Example Output**:
```
Current Regime: trend_up

- 1d×trend_up: signal=+1, confidence=0.6534 ✅
- 3d×trend_up: signal=+1, confidence=0.6123 ✅
- 5d×trend_up: signal=+1, confidence=0.7234 ✅
- 10d×trend_up: signal=+1, confidence=0.6891 ✅
```

---

### Test 6: Confidence Distribution ✅

**What it checks**:
- Confidence values are diverse
- No constant predictions
- Confidence statistics (min, max, mean, std)

**Pass Criteria**: Confidence values show reasonable variance

**Example Output**:
```
- Min Confidence: 0.5234
- Max Confidence: 0.7891
- Mean Confidence: 0.6432
- Std Confidence: 0.0823
- Status: ✅ PASS
```

---

### Test 7: V1 vs V2 Comparison ✅

**What it checks**:
- V1 models exist (if available)
- V2 models exist
- Model counts
- Directory structure

**Pass Criteria**: V2 models present (V1 optional)

**Example Output**:
```
- V1 Directory: /Users/.../.prado/models/qqq/ml_horizons
- V1 Exists: ✅ YES
- V2 Directory: /Users/.../.prado/models/qqq/ml_v2
- V2 Exists: ✅ YES
- V1 Model Count: 24
- V2 Model Count: 24
```

---

### Test 8: SHAP Explainability ⚠️

**What it checks**:
- SHAP library available (optional)
- SHAP values can be computed
- Top 5 most important features

**Pass Criteria**: SHAP values computed (or gracefully skipped if unavailable)

**Example Output**:
```
- Status: ✅ PASS
- Top 5 Features:
  - rsi_14
  - macd_line
  - trend_slope_20
  - vol_20
  - ret_5d
```

**If SHAP unavailable**:
```
- Status: ⚠️ SHAP not installed
```

---

### Test 9: Prediction Consistency ✅

**What it checks**:
- Different horizons make different predictions
- Predictions are not constant across all models
- Model diversity is working

**Pass Criteria**: At least 2 different signals across horizons

**Example Output**:
```
- Models Loaded: 4
- Unique Signals: 2
- Prediction Variance: ✅ GOOD
  - 1d: +1 (conf=0.6234)
  - 3d: -1 (conf=0.5823)
  - 5d: +1 (conf=0.7145)
  - 10d: +1 (conf=0.6891)
```

**Bad Example** (V2 not working):
```
- Models Loaded: 4
- Unique Signals: 1
- Prediction Variance: ⚠️ ALL SAME
  - 1d: +1 (conf=0.6234)
  - 3d: +1 (conf=0.6234)
  - 5d: +1 (conf=0.6234)
  - 10d: +1 (conf=0.6234)
```

---

## Output

### Terminal Display

The diagnostic suite displays:
1. Configuration panel
2. Progress indicators for each step
3. Test execution messages
4. Summary table with all test results
5. Path to saved report

### Markdown Report

**Filename**: `ML_V2_DIAGNOSTIC_REPORT_{SYMBOL}.md`

**Location**: Current working directory

**Contains**:
- Full test results
- Detailed statistics
- Error messages (if any)
- Feature lists
- Model status

**Example**:
```markdown
# ML V2 Diagnostic Report

**Symbol**: QQQ
**Date**: 2025-11-21 14:30:45
**Data Range**: 2020-01-02 → 2024-12-31
**Total Bars**: 1258

## Test 1: Feature Integrity
- Expected Features: 24
- Actual Features: 24
...
```

---

## When to Run Diagnostics

### Required

1. **After initial training**: Validate V2 models work correctly
   ```bash
   prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024
   prado ml-v2-diagnostic QQQ start 01 01 2020 end 12 31 2024
   ```

2. **After retraining**: Verify models still function correctly

3. **Before production**: Ensure everything works before deploying V2

### Recommended

1. **Monthly**: Check model health with latest data
2. **After market regime changes**: Verify regime models adapt
3. **When performance degrades**: Diagnose potential issues

### Optional

1. **During development**: Test changes to feature/target builders
2. **Troubleshooting**: Debug prediction issues
3. **Comparison**: Validate V1 vs V2 differences

---

## Interpreting Results

### All Tests Pass ✅

**Status**: V2 models are healthy and ready for production

**Next Steps**:
- Enable V2 in backtest config: `use_ml_features_v2=True`
- Run backtests to compare V1 vs V2 performance
- Monitor live predictions

### Some Tests Fail ❌

**Common Issues**:

1. **Feature Integrity Fails**
   - Problem: Wrong number of features
   - Fix: Check `feature_builder_v2.py` implementation
   - Retrain models after fixing

2. **Model Loading Fails**
   - Problem: Models not found
   - Fix: Run `prado train-ml-v2` first
   - Check model directory: `~/.prado/models/{symbol}/ml_v2/`

3. **Predictions Error**
   - Problem: Model predictions crash
   - Fix: Check feature alignment
   - Verify data has required columns

4. **Prediction Consistency Fails**
   - Problem: All models predict same thing
   - Fix: This indicates V2 not working correctly
   - Check horizon-specific labels are different

### Some Tests Skipped ⚠️

**Normal Skips**:
- SHAP (if library not installed) - optional feature
- Regime models with insufficient data - expected behavior

**Concerning Skips**:
- Horizon models - should NOT skip
- All regime models - indicates training issue

---

## Integration with Workflow

### Complete ML V2 Workflow

```bash
# Step 1: Train V2 models
prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024

# Step 2: Validate models
prado ml-v2-diagnostic QQQ start 01 01 2020 end 12 31 2024

# Step 3: Review diagnostic report
cat ML_V2_DIAGNOSTIC_REPORT_QQQ.md

# Step 4: Enable V2 programmatically
python -c "
from afml_system.backtest import BacktestConfig, BacktestEngine
config = BacktestConfig(
    symbol='QQQ',
    enable_ml_fusion=True,
    use_ml_features_v2=True
)
engine = BacktestEngine(config)
results = engine.run()
print(f'Sharpe: {results.sharpe_ratio:.2f}')
"
```

---

## Troubleshooting

### Diagnostic Won't Run

**Error**: "ml-v2-diagnostic command not found"
**Fix**: Reinstall package
```bash
pip install -e .
```

**Error**: "No data retrieved"
**Fix**: Check symbol is valid and dates are correct

### Models Not Loading

**Error**: "❌ MISSING" for all models
**Fix**: Train models first
```bash
prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024
```

### Feature Count Wrong

**Error**: "Actual Features: 9" instead of 24
**Fix**: Check imports - may be using V1 feature builder
- Verify `feature_builder_v2.py` exists
- Check `ml_v2_diagnostic.py` imports `FeatureBuilderV2`

### SHAP Errors

**Error**: SHAP computation fails
**Fix**: Install SHAP (optional)
```bash
pip install shap
```

Or ignore - SHAP is optional for diagnostics

---

## Advanced Usage

### Programmatic Access

```python
from afml_system.ml.ml_v2_diagnostic import MLV2Diagnostic
from afml_system.regime.regime_detector import RegimeDetector
import yfinance as yf

# Load data
df = yf.download('QQQ', start='2020-01-01', end='2024-12-31')
df.columns = [c.lower() for c in df.columns]

# Detect regimes
detector = RegimeDetector()
regimes = detector.detect_regime_series(df)

# Run diagnostics
diagnostic = MLV2Diagnostic('QQQ')
results = diagnostic.run_full_diagnostic(df, regimes)

# Access results
print(f"Feature test: {results['features']['status']}")
print(f"Model loading: {results['model_loading']}")

# Save report
report_path = diagnostic.save_report('custom_report.md')
```

### Custom Diagnostics

You can extend `MLV2Diagnostic` class to add custom tests:

```python
class CustomDiagnostic(MLV2Diagnostic):
    def _test_custom_metric(self, df):
        # Your custom test here
        return {'status': 'PASS', 'value': 42}
```

---

## Summary

The ML V2 Diagnostic Suite is your validation tool for ensuring the ML V2 system is working correctly. Use it:

✅ After training models
✅ Before enabling V2 in production
✅ When troubleshooting issues
✅ To compare V1 vs V2

**Key Indicators of Healthy V2**:
- ✅ 24 features built
- ✅ All horizon models load
- ✅ Predictions vary across horizons
- ✅ Confidence shows reasonable distribution
- ✅ Regime models work for current regime

**If all tests pass**: V2 is ready for backtesting and production use.

---

**Version**: PRADO9_EVO v1.3
**Status**: Production Ready
**Date**: 2025-11-21
