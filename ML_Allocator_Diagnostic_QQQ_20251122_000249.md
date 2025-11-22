# PRADO9_EVO ML + Allocator Diagnostic Report

**Symbol:** QQQ
**Period:** 2020-01-01 to 2024-12-31
**Total Bars:** 1257
**Generated:** 2025-11-22T00:02:49.828298

## Executive Summary

- **Tests Run:** 10
- **Tests Passed:** 1 ✅
- **Tests Warning:** 8 ⚠️
- **Tests Failed:** 1 ❌
- **Overall ML Influence Score:** 44.1%

## Bottlenecks Detected

- Test 1: ML Impact on Final Signal
- Test 4: ML Tanh Suppression
- Test 7: ML Independent Trade Creation

## Critical Fixes Recommended

1. Increase ML fusion strategy weight if independent trades desired
2. Allow ML signals to bypass tanh when confidence > 0.75
3. Currently ML acts more as a filter/amplifier than independent signal generator
4. Consider alternative normalization (softmax, L2)
5. Consider adding 'ml_priority' mode where ML can override rules
6. Fix backtest execution errors before proceeding

## Detailed Test Results

### Test 1: ML Impact on Final Signal

**Status:** FAIL
**ML Influence Score:** 0.0%
**Bottleneck Detected:** Yes

**Findings:**
- Error: evo_backtest_standard() got an unexpected keyword argument 'enable_ml_fusion'

**Recommendations:**
- Fix backtest execution errors before proceeding

---

### Test 2: ML vs Rule Priority

**Status:** WARNING
**ML Influence Score:** 50.0%
**Bottleneck Detected:** No

**Findings:**
- Test requires instrumented backtest to track rule vs ML signal strength
- Current implementation uses allocator weighting which blends signals

**Recommendations:**
- Add per-trade logging to track rule signal strength vs ML signal strength
- Measure correlation between ML confidence and final position size

---

### Test 3: ML Meta-Labeling Suppression

**Status:** WARNING
**ML Influence Score:** 50.0%
**Bottleneck Detected:** No

**Findings:**
- Meta-labeling is designed to filter weak signals
- Need to verify if suppression threshold is too aggressive

**Recommendations:**
- Test different meta-labeling thresholds (0.55, 0.60, 0.65)
- Measure suppression rate: (signals filtered / total signals)

---

### Test 4: ML Tanh Suppression

**Status:** WARNING
**ML Influence Score:** 26.1%
**Bottleneck Detected:** Yes

**Findings:**
- Tanh normalization maps (-inf, +inf) → (-1, +1)
- Signal strength preservation: 23.1%
- Strong signals (>2.0) are heavily compressed

**Recommendations:**
- Consider alternative normalization (softmax, L2)
- Allow ML signals to bypass tanh when confidence > 0.75

**Metrics:**
- avg_suppression_pct: 73.9334789840601

---

### Test 5: ML Volatility Scaling Suppression

**Status:** PASS
**ML Influence Score:** 75.0%
**Bottleneck Detected:** No

**Findings:**
- Volatility scaling reduces position size during high volatility
- ML signals should maintain relative strength after scaling

**Recommendations:**
- Volatility scaling affects all strategies equally
- ML relative influence should be preserved

---

### Test 6: ML Application Frequency

**Status:** WARNING
**ML Influence Score:** 50.0%
**Bottleneck Detected:** No

**Findings:**
- Requires per-trade logging to measure application rate
- Should track: ML signal present, ML signal used, ML signal dominant

**Recommendations:**
- Add trade-level ML usage tracking
- Measure: applied_count / total_trade_count

---

### Test 7: ML Independent Trade Creation

**Status:** WARNING
**ML Influence Score:** 40.0%
**Bottleneck Detected:** Yes

**Findings:**
- Current architecture: ML is a separate strategy in allocator
- ML can theoretically create independent trades
- Need to verify ML weight is sufficient for independent action

**Recommendations:**
- Increase ML fusion strategy weight if independent trades desired
- Currently ML acts more as a filter/amplifier than independent signal generator
- Consider adding 'ml_priority' mode where ML can override rules

---

### Test 8: Per-Trade ML Contribution

**Status:** WARNING
**ML Influence Score:** 50.0%
**Bottleneck Detected:** No

**Findings:**
- Per-trade analysis requires trade-level logging
- Should measure: ML signal strength, final position, ML attribution

**Recommendations:**
- Add per-trade ML contribution tracking
- Calculate Shapley values for ML vs rules

---

### Test 9: ML Confidence Distribution

**Status:** WARNING
**ML Influence Score:** 50.0%
**Bottleneck Detected:** No

**Findings:**
- ML models output probabilities [0, 1]
- High confidence (>0.7) should correlate with better outcomes
- Need actual prediction data to analyze distribution

**Recommendations:**
- Log ML confidence scores for each prediction
- Analyze: confidence histogram, confidence vs outcome correlation

---

### Test 10: Horizon Model Agreement

**Status:** WARNING
**ML Influence Score:** 50.0%
**Bottleneck Detected:** No

**Findings:**
- System has 4 horizon models: 1d, 3d, 5d, 10d
- Agreement strengthens signal, disagreement requires resolution
- Need horizon-specific prediction logs to measure agreement rate

**Recommendations:**
- Log predictions from all 4 horizon models
- Measure agreement rate: (same_direction_count / total_predictions)
- Analyze: how are disagreements weighted/resolved

---

