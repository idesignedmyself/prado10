# ML V2 Diagnostic Report
**Symbol**: QQQ
**Date**: 2025-11-21 20:49:29
**Data Range**: 2020-01-02 → 2024-12-30
**Total Bars**: 1257

## Test 1: Feature Integrity
- **Expected Features**: 24
- **Actual Features**: 24
- **Shape**: (1197, 24)
- **NaN Count**: 0
- **Status**: ✅ PASS

## Test 2: Target Integrity
- **Horizon Labels Shape**: (1257, 4)
- **Regime Labels Shape**: (1257, 7)
- **Horizon Columns**: ['label_up_1d', 'label_up_3d', 'label_up_5d', 'label_up_10d']
- **Status**: ✅ PASS

## Test 3: Model Loading
### Horizon Models
- **1d**: ❌ MISSING
- **3d**: ❌ MISSING
- **5d**: ❌ MISSING
- **10d**: ❌ MISSING

### Regime Models (Sample)
- **trend_up×1d**: ⚠️ MISSING
- **trend_up×5d**: ⚠️ MISSING
- **trend_down×1d**: ⚠️ MISSING
- **trend_down×5d**: ⚠️ MISSING

## Test 4: Horizon Predictions
- **1d**: ⚠️ SKIPPED (model not found)
- **3d**: ⚠️ SKIPPED (model not found)
- **5d**: ⚠️ SKIPPED (model not found)
- **10d**: ⚠️ SKIPPED (model not found)

## Test 5: Regime Predictions
**Current Regime**: trend_up

- **1d×trend_up**: signal=+0, confidence=0.5000 ✅
- **3d×trend_up**: signal=+0, confidence=0.5000 ✅
- **5d×trend_up**: signal=+0, confidence=0.5000 ✅
- **10d×trend_up**: signal=+0, confidence=0.5000 ✅

## Test 6: Confidence Distribution
- **Status**: ⚠️ NO DATA

## Test 7: V1 vs V2 Comparison
- **V1 Directory**: /Users/darraykennedy/.prado/models/qqq/ml_horizons
- **V1 Exists**: ❌ NO
- **V2 Directory**: /Users/darraykennedy/.prado/models/qqq/ml_v2
- **V2 Exists**: ✅ YES
- **V2 Model Count**: 9

## Test 8: SHAP Explainability
- **Status**: ⚠️ SHAP not installed

## Test 9: Prediction Consistency
- **Status**: ⚠️ INSUFFICIENT DATA
