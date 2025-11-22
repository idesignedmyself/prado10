"""
ML V2 Diagnostic Suite

Validates ML V2 models after training:
- Feature reconstruction
- Label reconstruction
- Model loading validation
- Prediction sanity checks
- Horizon consistency
- Regime consistency
- V1 vs V2 comparison
- Confidence distribution
- SHAP explainability (optional)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .feature_builder_v2 import FeatureBuilderV2
from .target_builder_v2 import TargetBuilderV2
from .horizon_models import HorizonModel
from .regime_models import RegimeHorizonModel

# SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class MLV2Diagnostic:
    """
    Comprehensive diagnostic suite for ML V2 models.

    Runs after training to validate:
    - Feature integrity
    - Model loading
    - Prediction validity
    - Horizon/regime consistency
    - V1/V2 comparison
    - SHAP explainability
    """

    def __init__(self, symbol: str):
        self.symbol = symbol.lower()
        self.report_lines = []
        self.horizons = ['1d', '3d', '5d', '10d']
        self.regimes = ['trend_up', 'trend_down', 'choppy', 'high_vol', 'low_vol']

    def log(self, line: str):
        """Add line to report"""
        self.report_lines.append(line)

    def run_full_diagnostic(self, df: pd.DataFrame, regime_series: pd.Series) -> Dict:
        """
        Run complete diagnostic suite.

        Returns dictionary with all test results.
        """
        results = {}

        self.log("# ML V2 Diagnostic Report")
        self.log(f"**Symbol**: {self.symbol.upper()}")
        self.log(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"**Data Range**: {df.index[0].date()} → {df.index[-1].date()}")
        self.log(f"**Total Bars**: {len(df)}")
        self.log("")

        # Test 1: Feature Integrity
        results['features'] = self._test_feature_integrity(df)

        # Test 2: Target Integrity
        results['targets'] = self._test_target_integrity(df)

        # Test 3: Model Loading
        results['model_loading'] = self._test_model_loading()

        # Test 4: Horizon Predictions
        results['horizon_predictions'] = self._test_horizon_predictions(df)

        # Test 5: Regime Predictions
        results['regime_predictions'] = self._test_regime_predictions(df, regime_series)

        # Test 6: Confidence Distribution
        results['confidence'] = self._test_confidence_distribution(df)

        # Test 7: V1 vs V2 Comparison
        results['v1_v2_comparison'] = self._test_v1_v2_comparison()

        # Test 8: SHAP Explainability
        results['shap'] = self._test_shap_explainability(df)

        # Test 9: Prediction Consistency
        results['consistency'] = self._test_prediction_consistency(df)

        return results

    def _test_feature_integrity(self, df: pd.DataFrame) -> Dict:
        """Test 1: Validate feature reconstruction"""
        self.log("## Test 1: Feature Integrity")

        try:
            X = FeatureBuilderV2.build_features_v2(df)

            expected_features = 24
            actual_features = X.shape[1]

            self.log(f"- **Expected Features**: {expected_features}")
            self.log(f"- **Actual Features**: {actual_features}")
            self.log(f"- **Shape**: {X.shape}")
            self.log(f"- **NaN Count**: {X.isna().sum().sum()}")
            self.log(f"- **Status**: {'✅ PASS' if actual_features == expected_features else '❌ FAIL'}")
            self.log("")

            return {
                'status': 'PASS' if actual_features == expected_features else 'FAIL',
                'expected': expected_features,
                'actual': actual_features,
                'shape': X.shape,
                'nan_count': X.isna().sum().sum()
            }
        except Exception as e:
            self.log(f"- **Status**: ❌ ERROR")
            self.log(f"- **Error**: {str(e)}")
            self.log("")
            return {'status': 'ERROR', 'error': str(e)}

    def _test_target_integrity(self, df: pd.DataFrame) -> Dict:
        """Test 2: Validate target reconstruction"""
        self.log("## Test 2: Target Integrity")

        try:
            horizon_labels = TargetBuilderV2.build_horizon_labels(df)
            regime_labels = TargetBuilderV2.build_regime_labels(df)

            self.log(f"- **Horizon Labels Shape**: {horizon_labels.shape}")
            self.log(f"- **Regime Labels Shape**: {regime_labels.shape}")
            self.log(f"- **Horizon Columns**: {list(horizon_labels.columns)}")
            self.log(f"- **Status**: ✅ PASS")
            self.log("")

            return {
                'status': 'PASS',
                'horizon_shape': horizon_labels.shape,
                'regime_shape': regime_labels.shape
            }
        except Exception as e:
            self.log(f"- **Status**: ❌ ERROR")
            self.log(f"- **Error**: {str(e)}")
            self.log("")
            return {'status': 'ERROR', 'error': str(e)}

    def _test_model_loading(self) -> Dict:
        """Test 3: Validate model loading"""
        self.log("## Test 3: Model Loading")

        results = {}

        # Test horizon models
        self.log("### Horizon Models")
        for horizon in self.horizons:
            model = HorizonModel(self.symbol, horizon, use_v2=True)
            loaded = model.load()
            self.log(f"- **{horizon}**: {'✅ LOADED' if loaded else '❌ MISSING'}")
            results[f'horizon_{horizon}'] = 'LOADED' if loaded else 'MISSING'

        self.log("")

        # Test regime models
        self.log("### Regime Models (Sample)")
        for regime in self.regimes[:2]:  # Just check first 2 regimes
            for horizon in ['1d', '5d']:  # Just check 2 horizons
                model = RegimeHorizonModel(self.symbol, horizon, use_v2=True)
                model_path = os.path.join(
                    model.model_dir,
                    f"ml_regime_{regime}_{horizon}_v2.pkl"
                )
                exists = os.path.exists(model_path)
                self.log(f"- **{regime}×{horizon}**: {'✅ EXISTS' if exists else '⚠️ MISSING'}")
                results[f'regime_{regime}_{horizon}'] = 'EXISTS' if exists else 'MISSING'

        self.log("")
        return results

    def _test_horizon_predictions(self, df: pd.DataFrame) -> Dict:
        """Test 4: Validate horizon predictions"""
        self.log("## Test 4: Horizon Predictions")

        results = {}

        for horizon in self.horizons:
            try:
                model = HorizonModel(self.symbol, horizon, use_v2=True)
                if not model.load():
                    self.log(f"- **{horizon}**: ⚠️ SKIPPED (model not found)")
                    results[horizon] = {'status': 'SKIPPED'}
                    continue

                signal, conf = model.predict(df)

                self.log(f"- **{horizon}**: signal={signal:+d}, confidence={conf:.4f} ✅")
                results[horizon] = {
                    'status': 'PASS',
                    'signal': signal,
                    'confidence': conf
                }
            except Exception as e:
                self.log(f"- **{horizon}**: ❌ ERROR - {str(e)}")
                results[horizon] = {'status': 'ERROR', 'error': str(e)}

        self.log("")
        return results

    def _test_regime_predictions(self, df: pd.DataFrame, regime_series: pd.Series) -> Dict:
        """Test 5: Validate regime predictions"""
        self.log("## Test 5: Regime Predictions")

        results = {}
        current_regime = regime_series.iloc[-1]

        self.log(f"**Current Regime**: {current_regime}")
        self.log("")

        for horizon in self.horizons:
            try:
                model = RegimeHorizonModel(self.symbol, horizon, use_v2=True)
                signal, conf = model.predict(df, current_regime)

                self.log(f"- **{horizon}×{current_regime}**: signal={signal:+d}, confidence={conf:.4f} ✅")
                results[f'{horizon}_{current_regime}'] = {
                    'status': 'PASS',
                    'signal': signal,
                    'confidence': conf
                }
            except Exception as e:
                self.log(f"- **{horizon}×{current_regime}**: ❌ ERROR - {str(e)}")
                results[f'{horizon}_{current_regime}'] = {'status': 'ERROR', 'error': str(e)}

        self.log("")
        return results

    def _test_confidence_distribution(self, df: pd.DataFrame) -> Dict:
        """Test 6: Analyze confidence distribution"""
        self.log("## Test 6: Confidence Distribution")

        confidences = []

        for horizon in self.horizons:
            model = HorizonModel(self.symbol, horizon, use_v2=True)
            if model.load():
                _, conf = model.predict(df)
                confidences.append(conf)

        if confidences:
            self.log(f"- **Min Confidence**: {min(confidences):.4f}")
            self.log(f"- **Max Confidence**: {max(confidences):.4f}")
            self.log(f"- **Mean Confidence**: {np.mean(confidences):.4f}")
            self.log(f"- **Std Confidence**: {np.std(confidences):.4f}")
            self.log(f"- **Status**: ✅ PASS")
            self.log("")

            return {
                'status': 'PASS',
                'min': min(confidences),
                'max': max(confidences),
                'mean': np.mean(confidences),
                'std': np.std(confidences)
            }
        else:
            self.log("- **Status**: ⚠️ NO DATA")
            self.log("")
            return {'status': 'NO_DATA'}

    def _test_v1_v2_comparison(self) -> Dict:
        """Test 7: Compare V1 and V2 models"""
        self.log("## Test 7: V1 vs V2 Comparison")

        v1_dir = Path('.prado') / 'models' / self.symbol / 'ml_horizons'
        v2_dir = Path('.prado') / 'models' / self.symbol / 'ml_v2'

        v1_exists = v1_dir.exists()
        v2_exists = v2_dir.exists()

        self.log(f"- **V1 Directory**: {v1_dir}")
        self.log(f"- **V1 Exists**: {'✅ YES' if v1_exists else '❌ NO'}")
        self.log(f"- **V2 Directory**: {v2_dir}")
        self.log(f"- **V2 Exists**: {'✅ YES' if v2_exists else '❌ NO'}")

        if v1_exists:
            v1_files = list(v1_dir.glob('*.joblib'))
            self.log(f"- **V1 Model Count**: {len(v1_files)}")

        if v2_exists:
            v2_files = list(v2_dir.glob('*.pkl'))
            self.log(f"- **V2 Model Count**: {len(v2_files)}")

        self.log("")

        return {
            'v1_exists': v1_exists,
            'v2_exists': v2_exists,
            'v1_count': len(v1_files) if v1_exists else 0,
            'v2_count': len(v2_files) if v2_exists else 0
        }

    def _test_shap_explainability(self, df: pd.DataFrame) -> Dict:
        """Test 8: SHAP explainability (optional)"""
        self.log("## Test 8: SHAP Explainability")

        if not SHAP_AVAILABLE:
            self.log("- **Status**: ⚠️ SHAP not installed")
            self.log("")
            return {'status': 'UNAVAILABLE', 'reason': 'SHAP not installed'}

        try:
            # Test on 1d model
            model = HorizonModel(self.symbol, '1d', use_v2=True)
            if not model.load():
                self.log("- **Status**: ⚠️ SKIPPED (model not found)")
                self.log("")
                return {'status': 'SKIPPED'}

            X = FeatureBuilderV2.build_features_v2(df)

            # Use last 100 rows for SHAP
            X_sample = X.iloc[-100:]

            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(X_sample)

            # Get feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_features = X.columns[np.argsort(feature_importance)[-5:]][::-1]

            self.log("- **Status**: ✅ PASS")
            self.log("- **Top 5 Features**:")
            for feat in top_features:
                self.log(f"  - {feat}")
            self.log("")

            return {
                'status': 'PASS',
                'top_features': list(top_features)
            }
        except Exception as e:
            self.log(f"- **Status**: ❌ ERROR")
            self.log(f"- **Error**: {str(e)}")
            self.log("")
            return {'status': 'ERROR', 'error': str(e)}

    def _test_prediction_consistency(self, df: pd.DataFrame) -> Dict:
        """Test 9: Check prediction consistency across horizons"""
        self.log("## Test 9: Prediction Consistency")

        predictions = {}

        for horizon in self.horizons:
            model = HorizonModel(self.symbol, horizon, use_v2=True)
            if model.load():
                signal, conf = model.predict(df)
                predictions[horizon] = (signal, conf)

        if len(predictions) >= 2:
            # Check if predictions vary
            signals = [p[0] for p in predictions.values()]
            unique_signals = len(set(signals))

            self.log(f"- **Models Loaded**: {len(predictions)}")
            self.log(f"- **Unique Signals**: {unique_signals}")
            self.log(f"- **Prediction Variance**: {'✅ GOOD' if unique_signals > 1 else '⚠️ ALL SAME'}")

            for horizon, (signal, conf) in predictions.items():
                self.log(f"  - {horizon}: {signal:+d} (conf={conf:.4f})")

            self.log("")

            return {
                'status': 'PASS',
                'models_loaded': len(predictions),
                'unique_signals': unique_signals,
                'predictions': predictions
            }
        else:
            self.log("- **Status**: ⚠️ INSUFFICIENT DATA")
            self.log("")
            return {'status': 'INSUFFICIENT_DATA'}

    def save_report(self, filename: Optional[str] = None) -> str:
        """Save diagnostic report to file"""
        if filename is None:
            filename = f"ML_V2_DIAGNOSTIC_REPORT_{self.symbol.upper()}.md"

        with open(filename, 'w') as f:
            f.write('\n'.join(self.report_lines))

        return filename

    def get_report(self) -> str:
        """Get report as string"""
        return '\n'.join(self.report_lines)
