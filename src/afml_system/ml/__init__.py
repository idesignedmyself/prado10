"""
PRADO9_EVO ML Module

Machine Learning components for hybrid rule + ML fusion:
- Multi-horizon ML models (1d, 3d, 5d, 10d)
- Regime-specific ML wrappers
- Hybrid fusion engine
- Shared feature builder
- SHAP model explainability (optional)

Author: PRADO9_EVO Builder
Date: 2025-01-21
Version: 1.1.0
"""

from .horizon_models import HorizonModel, HORIZONS
from .regime_models import RegimeHorizonModel, REGIMES
from .hybrid_fusion import HybridMLFusion
from .feature_builder import FeatureBuilder
from .shap_explainer import SHAPExplainer, SHAP_AVAILABLE

__all__ = [
    'HorizonModel',
    'HORIZONS',
    'RegimeHorizonModel',
    'REGIMES',
    'HybridMLFusion',
    'FeatureBuilder',
    'SHAPExplainer',
    'SHAP_AVAILABLE',
]
