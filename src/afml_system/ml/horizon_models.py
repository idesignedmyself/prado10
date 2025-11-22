"""
Multi-Horizon ML Models

Provides ML models trained on different forecast horizons:
- 1d: Short-term direction
- 3d: Swing bias
- 5d: Medium-term trend
- 10d: Long-term trend horizon

Each model predicts directional probability using XGBoost.
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from typing import Dict, Tuple, Optional

from .feature_builder import FeatureBuilder
try:
    from .feature_builder_v2 import FeatureBuilderV2
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

HORIZONS = {
    "1d": 1,
    "3d": 3,
    "5d": 5,
    "10d": 10
}


class HorizonModel:
    """
    ML model for predicting directional moves over different horizons.

    Each horizon gets its own XGBoost model, trained on the same feature set.
    """

    def __init__(self, symbol: str, horizon_key: str, use_v2: bool = False):
        if horizon_key not in HORIZONS:
            raise ValueError(f"Invalid horizon: {horizon_key}")

        self.symbol = symbol.lower()
        self.horizon_key = horizon_key
        self.horizon = HORIZONS[horizon_key]
        self.use_v2 = use_v2

        # V2 models saved in separate directory
        if use_v2:
            self.model_dir = f".prado/models/{self.symbol}/ml_v2/"
            self.model_path = os.path.join(self.model_dir, f"ml_horizon_{horizon_key}_v2.pkl")
        else:
            self.model_dir = f".prado/models/{self.symbol}/ml_horizons/"
            self.model_path = os.path.join(self.model_dir, f"h_{horizon_key}.joblib")

        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------
    def train(self, df: pd.DataFrame):
        """
        Train XGBoost model for the horizon using FeatureBuilder.
        """
        X = FeatureBuilder.build_features(df)
        y = self._build_targets(df)

        if len(X) != len(y):
            m = min(len(X), len(y))
            X = X.iloc[-m:]
            y = y.iloc[-m:]

        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X, y)
        self.model = model
        self.save()

    # -----------------------------------------------------
    # TARGETS
    # -----------------------------------------------------
    def _build_targets(self, df: pd.DataFrame) -> pd.Series:
        """
        Class 1 = Up move over horizon
        Class 0 = Down move
        """
        df = df.copy()
        if "Close" in df.columns and "close" not in df.columns:
            df["close"] = df["Close"]

        future = np.log(df["close"].shift(-self.horizon) / df["close"])
        return (future > 0).astype(int).dropna()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------
    def predict(self, window: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict long/short + confidence from the latest window.
        Returns:
            signal ∈ {1, -1}
            confidence ∈ [0,1]
        """

        if self.model is None and not self.load():
            return 0, 0.5

        # Use v2 or v1 features based on model type
        if self.use_v2 and HAS_V2:
            X = FeatureBuilderV2.build_features_v2(window)
        else:
            X = FeatureBuilder.build_features(window)

        if X.empty:
            return 0, 0.5

        X_last = X.iloc[-1:]
        prob = float(self.model.predict_proba(X_last)[0, 1])
        signal = 1 if prob > 0.5 else -1

        return signal, prob

    # -----------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------
    def save(self):
        joblib.dump(self.model, self.model_path)

    def load(self) -> bool:
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False
