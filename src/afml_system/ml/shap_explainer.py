"""
SHAP Model Explainability

Provides model interpretation using SHAP (SHapley Additive exPlanations).
Helps understand which features drive ML predictions.

Optional module - gracefully degrades if SHAP not installed.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based model explainer for XGBoost models.

    Provides feature importance and per-prediction explanations.
    """

    def __init__(self):
        """Initialize SHAP explainer."""
        self.explainer = None
        self.feature_names = None

    def fit(self, model: Any, X: pd.DataFrame):
        """
        Fit SHAP explainer to model.

        Args:
            model: Trained XGBoost model
            X: Feature dataframe used for training
        """
        if not SHAP_AVAILABLE:
            return

        self.feature_names = X.columns.tolist()
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Explain a single prediction.

        Args:
            X: Single row dataframe (1 sample)

        Returns:
            Dict mapping feature -> SHAP value, or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return None

        shap_values = self.explainer.shap_values(X)

        # Extract SHAP values for the single prediction
        if isinstance(shap_values, list):
            # Binary classification - use class 1 SHAP values
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        # Map to feature names
        explanation = dict(zip(self.feature_names, shap_vals))

        return explanation

    def get_feature_importance(self, X: pd.DataFrame, top_k: int = 10) -> Optional[Dict[str, float]]:
        """
        Get global feature importance across dataset.

        Args:
            X: Feature dataframe
            top_k: Return top K features

        Returns:
            Dict mapping feature -> mean |SHAP value|, or None if unavailable
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return None

        shap_values = self.explainer.shap_values(X)

        # Extract SHAP values
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Class 1 for binary classification
        else:
            shap_vals = shap_values

        # Compute mean absolute SHAP value per feature
        mean_shap = np.abs(shap_vals).mean(axis=0)

        # Map to feature names
        importance = dict(zip(self.feature_names, mean_shap))

        # Sort and take top K
        sorted_importance = dict(sorted(importance.items(), key=lambda x: -x[1])[:top_k])

        return sorted_importance
