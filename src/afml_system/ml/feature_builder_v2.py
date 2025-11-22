"""
Enhanced Feature Builder V2 for ML Models

ADDITIVE DESIGN - Does not replace feature_builder.py
Maintains backward compatibility while adding 15 new features.

Total Features: 24
- Original 9 features (from feature_builder.py)
- New 15 features (momentum, volatility, trend, volume indicators)

This expanded feature set creates model diversity across:
- Different horizons (1d, 3d, 5d, 10d)
- Different regimes (trend_up, trend_down, choppy, high_vol, low_vol)
"""

import numpy as np
import pandas as pd
from typing import Optional


class FeatureBuilderV2:
    """
    Enhanced feature builder with 24 ML features.

    Feature Categories:
    1. Returns (3): ret_1d, ret_3d, ret_5d
    2. Volatility Structure (3): vol_20, vol_60, vol_ratio
    3. Trend Structure (3): ma_ratio, dist_ma_20, dist_ma_50
    4. Momentum (4): rsi_14, roc_10, momentum_zscore_20, stochastic_k
    5. Volatility Expansion (4): atr_14, bb_width_20, hv_10, vol_change_5
    6. Trend/Slope (4): trend_slope_20, trend_slope_50, macd_line, macd_hist
    7. Volume (3): vol_rel_20, vol_accel_5, obv_change
    """

    @staticmethod
    def _safe_division(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """Safe division with inf/nan handling"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
        result.replace([np.inf, -np.inf], fill_value, inplace=True)
        result.fillna(fill_value, inplace=True)
        return result

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = FeatureBuilderV2._safe_division(gain, loss, fill_value=1.0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Stochastic Oscillator %K"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).min()
        stoch_k = FeatureBuilderV2._safe_division(
            (close - lowest_low),
            (highest_high - lowest_low),
            fill_value=50.0
        ) * 100
        return stoch_k

    @staticmethod
    def _compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Compute MACD line and histogram"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_hist

    @staticmethod
    def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build enhanced feature set with 24 features.

        Args:
            df: DataFrame with OHLCV columns (close, high, low, volume)

        Returns:
            DataFrame with 24 ML features
        """
        # Normalize column names
        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]

        # Ensure required columns exist
        if 'close' not in df.columns:
            if 'adj close' in df.columns:
                df['close'] = df['adj close']
            else:
                raise ValueError("DataFrame must have 'close' or 'adj close' column")

        # Create high/low/volume with fallbacks
        if 'high' not in df.columns:
            df['high'] = df['close'] * 1.01
        if 'low' not in df.columns:
            df['low'] = df['close'] * 0.99
        if 'volume' not in df.columns:
            df['volume'] = 1000000.0

        out = pd.DataFrame(index=df.index)

        # =====================================================================
        # ORIGINAL 9 FEATURES (from feature_builder.py)
        # =====================================================================

        # Returns (3)
        out["ret_1d"] = np.log(df["close"] / df["close"].shift(1))
        out["ret_3d"] = np.log(df["close"] / df["close"].shift(3))
        out["ret_5d"] = np.log(df["close"] / df["close"].shift(5))

        # Volatility structure (3)
        out["vol_20"] = out["ret_1d"].rolling(20).std()
        out["vol_60"] = out["ret_1d"].rolling(60).std()
        out["vol_ratio"] = FeatureBuilderV2._safe_division(out["vol_20"], out["vol_60"], fill_value=1.0)

        # Trend structure (3)
        ma50 = df["close"].rolling(50).mean()
        ma200 = df["close"].rolling(200).mean()
        out["ma_ratio"] = FeatureBuilderV2._safe_division(ma50, ma200, fill_value=1.0) - 1.0
        out["dist_ma_20"] = FeatureBuilderV2._safe_division(df["close"], df["close"].rolling(20).mean(), fill_value=1.0) - 1
        out["dist_ma_50"] = FeatureBuilderV2._safe_division(df["close"], ma50, fill_value=1.0) - 1

        # =====================================================================
        # NEW 15 FEATURES
        # =====================================================================

        # --- Momentum Features (4) ---
        out["rsi_14"] = FeatureBuilderV2._compute_rsi(df["close"], period=14)
        out["roc_10"] = FeatureBuilderV2._safe_division(
            df["close"] - df["close"].shift(10),
            df["close"].shift(10),
            fill_value=0.0
        ) * 100

        # Momentum z-score (standardized momentum)
        mom_20 = df["close"].diff(20)
        out["momentum_zscore_20"] = FeatureBuilderV2._safe_division(
            mom_20 - mom_20.rolling(60).mean(),
            mom_20.rolling(60).std(),
            fill_value=0.0
        )

        out["stochastic_k"] = FeatureBuilderV2._compute_stochastic(
            df["high"], df["low"], df["close"], period=14
        )

        # --- Volatility Expansion Features (4) ---
        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr_14"] = true_range.rolling(14).mean()

        # Bollinger Band Width
        bb_ma = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        out["bb_width_20"] = FeatureBuilderV2._safe_division(bb_std * 2, bb_ma, fill_value=0.0)

        # Historical Volatility (10-day)
        out["hv_10"] = np.log(df["close"] / df["close"].shift(1)).rolling(10).std() * np.sqrt(252)

        # Volatility Change
        out["vol_change_5"] = out["vol_20"] - out["vol_20"].shift(5)

        # --- Trend/Slope Features (4) ---
        # Linear regression slope of price
        def compute_slope(series, window):
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = series.iloc[i-window+1:i+1].values
                    x = np.arange(window)
                    if len(y) == window:
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
            return pd.Series(slopes, index=series.index)

        out["trend_slope_20"] = compute_slope(df["close"], 20)
        out["trend_slope_50"] = compute_slope(df["close"], 50)

        # MACD
        macd_line, macd_hist = FeatureBuilderV2._compute_macd(df["close"])
        out["macd_line"] = macd_line
        out["macd_hist"] = macd_hist

        # --- Volume Features (3) ---
        # Relative volume
        out["vol_rel_20"] = FeatureBuilderV2._safe_division(
            df["volume"],
            df["volume"].rolling(20).mean(),
            fill_value=1.0
        )

        # Volume acceleration
        vol_ma_5 = df["volume"].rolling(5).mean()
        vol_ma_20 = df["volume"].rolling(20).mean()
        out["vol_accel_5"] = FeatureBuilderV2._safe_division(vol_ma_5, vol_ma_20, fill_value=1.0) - 1

        # OBV change
        obv = FeatureBuilderV2._compute_obv(df["close"], df["volume"])
        out["obv_change"] = obv.pct_change(10).fillna(0)

        # =====================================================================
        # CLEANUP
        # =====================================================================
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out.dropna(inplace=True)

        return out

    @staticmethod
    def build_features(df: pd.DataFrame, use_v2: bool = False) -> pd.DataFrame:
        """
        Unified feature builder with backward compatibility.

        Args:
            df: OHLCV DataFrame
            use_v2: If True, use v2 features (24 features). If False, use v1 (9 features).

        Returns:
            Feature DataFrame
        """
        if use_v2:
            return FeatureBuilderV2.build_features_v2(df)
        else:
            # Fall back to original feature_builder.py logic
            from .feature_builder import FeatureBuilder
            return FeatureBuilder.build_features(df)


# Convenience function for backward compatibility
def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Build v2 features (24 features)"""
    return FeatureBuilderV2.build_features_v2(df)


__all__ = ['FeatureBuilderV2', 'build_features_v2']
