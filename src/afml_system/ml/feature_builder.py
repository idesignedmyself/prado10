"""
Shared Feature Builder for ML Models

Ensures consistent features between training and prediction.
"""

import numpy as np
import pandas as pd

class FeatureBuilder:
    """
    Builds ML features from OHLCV data.

    Features:
    - Log returns (1d, 3d, 5d)
    - Volatility structure (20d, 60d, ratio)
    - Trend structure (MA ratios)
    - Strength indicators (distance from MAs)
    """

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature set from price data.

        Args:
            df: DataFrame with 'close' or 'Close' column

        Returns:
            DataFrame with features
        """
        # Ensure lowercase close column
        if 'Close' in df.columns and 'close' not in df.columns:
            df = df.copy()
            df['close'] = df['Close']

        out = pd.DataFrame(index=df.index)

        # Log returns
        out["ret_1d"] = np.log(df["close"] / df["close"].shift(1))
        out["ret_3d"] = np.log(df["close"] / df["close"].shift(3))
        out["ret_5d"] = np.log(df["close"] / df["close"].shift(5))

        # Volatility structure
        out["vol_20"] = out["ret_1d"].rolling(20).std()
        out["vol_60"] = out["ret_1d"].rolling(60).std()
        out["vol_ratio"] = out["vol_20"] / out["vol_60"]

        # Trend structure
        ma50 = df["close"].rolling(50).mean()
        ma200 = df["close"].rolling(200).mean()
        out["ma_ratio"] = ma50 / ma200 - 1.0

        # Strength features
        out["dist_ma_20"] = df["close"] / df["close"].rolling(20).mean() - 1
        out["dist_ma_50"] = df["close"] / df["close"].rolling(50).mean() - 1

        # Clean
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out.dropna(inplace=True)

        return out
