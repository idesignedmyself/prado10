"""
Simple Regime Detector for ML Training

Classifies market conditions into 5 regimes:
- trend_up: Strong uptrend
- trend_down: Strong downtrend
- choppy: Range-bound/sideways
- high_vol: High volatility regime
- low_vol: Low volatility regime
"""

import numpy as np
import pandas as pd
from typing import Optional

class RegimeDetector:
    """
    Detects market regimes for ML model training.

    Uses simple heuristics:
    - Trend: MA50 vs MA200 slope
    - Volatility: Rolling 20-day vs 60-day std
    """

    def __init__(self):
        pass

    def detect_regime_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect regime for each bar.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series with regime labels
        """
        # Ensure lowercase column names
        df = df.copy()
        if 'Close' in df.columns:
            df['close'] = df['Close']

        # Calculate indicators
        ma50 = df['close'].rolling(50).mean()
        ma200 = df['close'].rolling(200).mean()

        returns = np.log(df['close'] / df['close'].shift(1))
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()

        # Trend detection
        trend_strength = (ma50 - ma200) / ma200

        # Volatility detection
        vol_ratio = vol_20 / vol_60

        # Classify regimes
        regime = pd.Series(index=df.index, dtype=str)

        for i in range(len(df)):
            if pd.isna(trend_strength.iloc[i]) or pd.isna(vol_ratio.iloc[i]):
                regime.iloc[i] = 'choppy'
                continue

            ts = trend_strength.iloc[i]
            vr = vol_ratio.iloc[i]

            # Volatility first
            if vr > 1.3:
                regime.iloc[i] = 'high_vol'
            elif vr < 0.7:
                regime.iloc[i] = 'low_vol'
            # Then trend
            elif ts > 0.05:
                regime.iloc[i] = 'trend_up'
            elif ts < -0.05:
                regime.iloc[i] = 'trend_down'
            else:
                regime.iloc[i] = 'choppy'

        return regime

    def detect_current_regime(self, df: pd.DataFrame) -> str:
        """Get regime for most recent bar."""
        regime_series = self.detect_regime_series(df)
        return regime_series.iloc[-1]
