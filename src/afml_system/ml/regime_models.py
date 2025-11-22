"""
Regime-Specific ML Models

Provides ML models trained separately for each market regime:
- trend_up: Strong uptrend
- trend_down: Strong downtrend
- choppy: Range-bound/sideways
- high_vol: High volatility regime
- low_vol: Low volatility regime

Each regime gets its own set of horizon models for specialized predictions.
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from typing import Dict, Tuple

from .feature_builder import FeatureBuilder
try:
    from .feature_builder_v2 import FeatureBuilderV2
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

REGIMES = [
    "trend_up",
    "trend_down",
    "choppy",
    "high_vol",
    "low_vol"
]


class RegimeHorizonModel:
    """
    Maintains a separate ML model for each regime at a given horizon.

    Example model directory:
      .prado/models/qqq/ml_horizons/trend_up_1d.joblib
    """

    def __init__(self, symbol: str, horizon_key: str, use_v2: bool = False):
        self.symbol = symbol.lower()
        self.horizon_key = horizon_key
        self.use_v2 = use_v2

        # V2 models saved in separate directory
        if use_v2:
            self.model_dir = f".prado/models/{self.symbol}/ml_v2/"
        else:
            self.model_dir = f".prado/models/{self.symbol}/ml_horizons/"

        os.makedirs(self.model_dir, exist_ok=True)
        self.models: Dict[str, xgb.XGBClassifier] = {}

    # -----------------------------------------------------
    # TRAINING
    # -----------------------------------------------------
    def train_all(self, df: pd.DataFrame, regime_series: pd.Series):
        """
        Train one ML model per regime at this horizon.
        """

        # Shared features for all regimes
        X = FeatureBuilder.build_features(df)

        # Build targets with same index as features
        y = self._build_targets(df)

        # Align features, targets, and regime series
        common_idx = X.index.intersection(y.index).intersection(regime_series.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        regime_series = regime_series.loc[common_idx]

        for regime in REGIMES:
            idx = regime_series == regime
            if idx.sum() < 250:
                continue  # Not enough data, skip

            X_r = X.loc[idx]
            y_r = y.loc[idx]

            if len(X_r) < 50:
                continue

            model_path = os.path.join(
                self.model_dir,
                f"{regime}_{self.horizon_key}.joblib"
            )

            model = xgb.XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_r, y_r)
            joblib.dump(model, model_path)

    # -----------------------------------------------------
    # TARGETS (shared with horizon model)
    # -----------------------------------------------------
    def _build_targets(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        if "Close" in df.columns and "close" not in df.columns:
            df["close"] = df["Close"]

        h = {"1d": 1, "3d": 3, "5d": 5, "10d": 10}[self.horizon_key]
        future = np.log(df["close"].shift(-h) / df["close"])
        return (future > 0).astype(int).dropna()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------
    def predict(self, window: pd.DataFrame, regime: str) -> Tuple[int, float]:
        """
        Predict signal for (regime, horizon).
        """

        # V2 uses different naming convention
        if self.use_v2:
            model_path = os.path.join(
                self.model_dir,
                f"ml_regime_{regime}_{self.horizon_key}_v2.pkl"
            )
        else:
            model_path = os.path.join(
                self.model_dir,
                f"{regime}_{self.horizon_key}.joblib"
            )

        if not os.path.exists(model_path):
            return 0, 0.5

        model = joblib.load(model_path)

        # Use v2 or v1 features based on model type
        if self.use_v2 and HAS_V2:
            X = FeatureBuilderV2.build_features_v2(window)
        else:
            X = FeatureBuilder.build_features(window)
        if X.empty:
            return 0, 0.5

        X_last = X.iloc[-1:]
        prob = float(model.predict_proba(X_last)[0, 1])
        signal = 1 if prob > 0.5 else -1

        return signal, prob
