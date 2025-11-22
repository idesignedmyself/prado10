"""
Optimized Feature Builder V2 for ML Models - Enhanced Edition

FULLY BACKWARD COMPATIBLE - All 24 original V2 features preserved
Adds 18 optimized features for improved predictive power.

Total Features: 42
- Original 24 V2 features (unchanged)
- New 18 optimized features (interaction, microstructure, regime-sensitive)

Enhancement Categories:
1. Return-based enhancements (4 features)
2. Advanced volatility structure (4 features)
3. Microstructure features (3 features)
4. Cross-feature interactions (3 features)
5. Regime-sensitive transforms (2 features)
6. Polynomial/nonlinear terms (2 features)

Author: PRADO9_EVO v3.7.0
Date: 2025-11-21
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy import stats


class FeatureBuilderV2Optimized:
    """
    Optimized feature builder with 42 ML features.

    Feature Categories:

    ORIGINAL 24 FEATURES (V2 Baseline):
    1. Returns (3): ret_1d, ret_3d, ret_5d
    2. Volatility Structure (3): vol_20, vol_60, vol_ratio
    3. Trend Structure (3): ma_ratio, dist_ma_20, dist_ma_50
    4. Momentum (4): rsi_14, roc_10, momentum_zscore_20, stochastic_k
    5. Volatility Expansion (4): atr_14, bb_width_20, hv_10, vol_change_5
    6. Trend/Slope (4): trend_slope_20, trend_slope_50, macd_line, macd_hist
    7. Volume (3): vol_rel_20, vol_accel_5, obv_change

    NEW 18 OPTIMIZED FEATURES:
    8. Return Enhancements (4): ret_skew_20, ret_kurt_20, ret_autocorr_5, ret_range_ratio_20
    9. Advanced Volatility (4): parkinson_vol_20, garman_klass_vol_20, rogers_satchell_vol_20, vol_asymmetry_20
    10. Microstructure (3): hl_spread_norm_20, intraday_range_vol_20, close_position_20
    11. Cross-Feature Interactions (3): vol_momentum_cross, trend_vol_interaction, volume_price_corr_20
    12. Regime-Sensitive (2): vol_regime_ratio, momentum_regime_adj
    13. Nonlinear Terms (2): ret_3d_squared, vol_ratio_squared
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
        rs = FeatureBuilderV2Optimized._safe_division(gain, loss, fill_value=1.0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Stochastic Oscillator %K"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch_k = FeatureBuilderV2Optimized._safe_division(
            (close - lowest_low),
            (highest_high - lowest_low),
            fill_value=50.0
        ) * 100
        return stoch_k

    @staticmethod
    def _compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
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
    def _compute_slope(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling linear regression slope"""
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

    # =========================================================================
    # NEW OPTIMIZED FEATURE COMPUTATIONS
    # =========================================================================

    @staticmethod
    def _compute_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """
        Parkinson volatility estimator (range-based volatility).
        More efficient than close-to-close vol, uses high-low range.
        """
        hl_ratio = np.log(high / low)
        parkinson_var = (hl_ratio ** 2) / (4 * np.log(2))
        parkinson_vol = np.sqrt(parkinson_var.rolling(window).mean()) * np.sqrt(252)
        return parkinson_vol

    @staticmethod
    def _compute_garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series,
                                        close: pd.Series, window: int = 20) -> pd.Series:
        """
        Garman-Klass volatility estimator (OHLC-based).
        More robust than Parkinson, incorporates open and close.
        """
        hl = np.log(high / low)
        co = np.log(close / open_)

        gk_var = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)
        gk_vol = np.sqrt(gk_var.rolling(window).mean()) * np.sqrt(252)
        return gk_vol

    @staticmethod
    def _compute_rogers_satchell_volatility(open_: pd.Series, high: pd.Series, low: pd.Series,
                                           close: pd.Series, window: int = 20) -> pd.Series:
        """
        Rogers-Satchell volatility estimator.
        Handles trending markets better than Garman-Klass.
        """
        hc = np.log(high / close)
        ho = np.log(high / open_)
        lc = np.log(low / close)
        lo = np.log(low / open_)

        rs_var = (hc * ho) + (lc * lo)
        rs_vol = np.sqrt(rs_var.rolling(window).mean()) * np.sqrt(252)
        return rs_vol

    @staticmethod
    def _compute_volatility_asymmetry(returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Volatility asymmetry: ratio of downside to upside volatility.
        Captures regime-dependent volatility structure.
        """
        upside_vol = returns.where(returns > 0, 0).rolling(window).std()
        downside_vol = returns.where(returns < 0, 0).rolling(window).std()
        vol_asymmetry = FeatureBuilderV2Optimized._safe_division(
            downside_vol, upside_vol, fill_value=1.0
        )
        return vol_asymmetry

    @staticmethod
    def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build optimized feature set with 42 features.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with 42 ML features (24 original + 18 optimized)
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

        # Create OHLV with fallbacks
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df['close'] * 1.01
        if 'low' not in df.columns:
            df['low'] = df['close'] * 0.99
        if 'volume' not in df.columns:
            df['volume'] = 1000000.0

        out = pd.DataFrame(index=df.index)

        # =====================================================================
        # ORIGINAL 24 V2 FEATURES (UNCHANGED FOR BACKWARD COMPATIBILITY)
        # =====================================================================

        # Returns (3)
        out["ret_1d"] = np.log(df["close"] / df["close"].shift(1))
        out["ret_3d"] = np.log(df["close"] / df["close"].shift(3))
        out["ret_5d"] = np.log(df["close"] / df["close"].shift(5))

        # Volatility structure (3)
        out["vol_20"] = out["ret_1d"].rolling(20).std()
        out["vol_60"] = out["ret_1d"].rolling(60).std()
        out["vol_ratio"] = FeatureBuilderV2Optimized._safe_division(out["vol_20"], out["vol_60"], fill_value=1.0)

        # Trend structure (3)
        ma50 = df["close"].rolling(50).mean()
        ma200 = df["close"].rolling(200).mean()
        out["ma_ratio"] = FeatureBuilderV2Optimized._safe_division(ma50, ma200, fill_value=1.0) - 1.0
        out["dist_ma_20"] = FeatureBuilderV2Optimized._safe_division(df["close"], df["close"].rolling(20).mean(), fill_value=1.0) - 1
        out["dist_ma_50"] = FeatureBuilderV2Optimized._safe_division(df["close"], ma50, fill_value=1.0) - 1

        # Momentum Features (4)
        out["rsi_14"] = FeatureBuilderV2Optimized._compute_rsi(df["close"], period=14)
        out["roc_10"] = FeatureBuilderV2Optimized._safe_division(
            df["close"] - df["close"].shift(10),
            df["close"].shift(10),
            fill_value=0.0
        ) * 100

        mom_20 = df["close"].diff(20)
        out["momentum_zscore_20"] = FeatureBuilderV2Optimized._safe_division(
            mom_20 - mom_20.rolling(60).mean(),
            mom_20.rolling(60).std(),
            fill_value=0.0
        )

        out["stochastic_k"] = FeatureBuilderV2Optimized._compute_stochastic(
            df["high"], df["low"], df["close"], period=14
        )

        # Volatility Expansion Features (4)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr_14"] = true_range.rolling(14).mean()

        bb_ma = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        out["bb_width_20"] = FeatureBuilderV2Optimized._safe_division(bb_std * 2, bb_ma, fill_value=0.0)

        out["hv_10"] = np.log(df["close"] / df["close"].shift(1)).rolling(10).std() * np.sqrt(252)
        out["vol_change_5"] = out["vol_20"] - out["vol_20"].shift(5)

        # Trend/Slope Features (4)
        out["trend_slope_20"] = FeatureBuilderV2Optimized._compute_slope(df["close"], 20)
        out["trend_slope_50"] = FeatureBuilderV2Optimized._compute_slope(df["close"], 50)

        macd_line, macd_hist = FeatureBuilderV2Optimized._compute_macd(df["close"])
        out["macd_line"] = macd_line
        out["macd_hist"] = macd_hist

        # Volume Features (3)
        out["vol_rel_20"] = FeatureBuilderV2Optimized._safe_division(
            df["volume"],
            df["volume"].rolling(20).mean(),
            fill_value=1.0
        )

        vol_ma_5 = df["volume"].rolling(5).mean()
        vol_ma_20 = df["volume"].rolling(20).mean()
        out["vol_accel_5"] = FeatureBuilderV2Optimized._safe_division(vol_ma_5, vol_ma_20, fill_value=1.0) - 1

        obv = FeatureBuilderV2Optimized._compute_obv(df["close"], df["volume"])
        out["obv_change"] = obv.pct_change(10).fillna(0)

        # =====================================================================
        # NEW 18 OPTIMIZED FEATURES
        # =====================================================================

        # --- Return Enhancements (4) ---

        # Return skewness (asymmetry of return distribution)
        out["ret_skew_20"] = out["ret_1d"].rolling(20).apply(lambda x: stats.skew(x, nan_policy='omit'), raw=False)

        # Return kurtosis (tail risk indicator)
        out["ret_kurt_20"] = out["ret_1d"].rolling(20).apply(lambda x: stats.kurtosis(x, nan_policy='omit'), raw=False)

        # Return autocorrelation (mean reversion/momentum persistence)
        out["ret_autocorr_5"] = out["ret_1d"].rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False)

        # Return range ratio (high-low range normalized by close)
        ret_range = (df["high"] - df["low"]) / df["close"]
        out["ret_range_ratio_20"] = ret_range.rolling(20).mean()

        # --- Advanced Volatility Structure (4) ---

        # Parkinson volatility (range-based, more efficient)
        out["parkinson_vol_20"] = FeatureBuilderV2Optimized._compute_parkinson_volatility(
            df["high"], df["low"], window=20
        )

        # Garman-Klass volatility (OHLC-based, more robust)
        out["garman_klass_vol_20"] = FeatureBuilderV2Optimized._compute_garman_klass_volatility(
            df["open"], df["high"], df["low"], df["close"], window=20
        )

        # Rogers-Satchell volatility (handles trends better)
        out["rogers_satchell_vol_20"] = FeatureBuilderV2Optimized._compute_rogers_satchell_volatility(
            df["open"], df["high"], df["low"], df["close"], window=20
        )

        # Volatility asymmetry (downside vs upside volatility)
        out["vol_asymmetry_20"] = FeatureBuilderV2Optimized._compute_volatility_asymmetry(
            out["ret_1d"], window=20
        )

        # --- Microstructure Features (3) ---

        # High-Low spread normalized by ATR (liquidity/efficiency proxy)
        out["hl_spread_norm_20"] = FeatureBuilderV2Optimized._safe_division(
            (df["high"] - df["low"]).rolling(20).mean(),
            out["atr_14"],
            fill_value=1.0
        )

        # Intraday range volatility (volatility of daily ranges)
        daily_range = (df["high"] - df["low"]) / df["close"]
        out["intraday_range_vol_20"] = daily_range.rolling(20).std()

        # Close position in range (0=low, 1=high) - buying/selling pressure
        out["close_position_20"] = FeatureBuilderV2Optimized._safe_division(
            df["close"] - df["low"].rolling(20).min(),
            df["high"].rolling(20).max() - df["low"].rolling(20).min(),
            fill_value=0.5
        )

        # --- Cross-Feature Interactions (3) ---

        # Volatility-Momentum cross (vol expansion during momentum)
        out["vol_momentum_cross"] = out["vol_20"] * out["roc_10"]

        # Trend-Volatility interaction (trending markets with low vol)
        out["trend_vol_interaction"] = out["trend_slope_20"] * (1.0 / (out["vol_20"] + 0.0001))

        # Volume-Price correlation (smart money flow)
        out["volume_price_corr_20"] = out["ret_1d"].rolling(20).corr(
            df["volume"].pct_change().fillna(0)
        )

        # --- Regime-Sensitive Features (2) ---

        # Volatility regime ratio (current vol vs long-term regime)
        vol_120 = out["ret_1d"].rolling(120).std()
        out["vol_regime_ratio"] = FeatureBuilderV2Optimized._safe_division(
            out["vol_20"], vol_120, fill_value=1.0
        )

        # Momentum regime adjustment (momentum adjusted for volatility regime)
        out["momentum_regime_adj"] = out["roc_10"] * out["vol_regime_ratio"]

        # --- Nonlinear/Polynomial Terms (2) ---

        # Squared return (captures volatility clustering + outliers)
        out["ret_3d_squared"] = out["ret_3d"] ** 2

        # Squared volatility ratio (amplifies regime transitions)
        out["vol_ratio_squared"] = out["vol_ratio"] ** 2

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
            use_v2: If True, use optimized v2 features (42 features).
                   If False, use original v1 (9 features).

        Returns:
            Feature DataFrame
        """
        if use_v2:
            return FeatureBuilderV2Optimized.build_features_v2(df)
        else:
            # Fall back to original feature_builder.py logic
            from .feature_builder import FeatureBuilder
            return FeatureBuilder.build_features(df)


# Convenience function for backward compatibility
def build_features_v2_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Build optimized v2 features (42 features)"""
    return FeatureBuilderV2Optimized.build_features_v2(df)


__all__ = ['FeatureBuilderV2Optimized', 'build_features_v2_optimized']
