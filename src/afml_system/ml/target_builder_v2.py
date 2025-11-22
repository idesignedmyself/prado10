"""
Enhanced Target Builder V2 for ML Models

ADDITIVE DESIGN - Creates horizon-specific and regime-conditioned labels
This ensures ML models diverge across different horizons and regimes.

Target Types:
1. Horizon-Specific Labels: label_up_1d, label_up_3d, label_up_5d, label_up_10d
2. Regime-Conditioned Labels: label_regime_{regime_name}

Key Innovation:
- Each horizon gets its own forward-looking label
- Each regime gets volatility-adjusted labels
- This creates TRUE model diversity
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class TargetBuilderV2:
    """
    Builds horizon-specific and regime-conditioned ML training labels.

    Design Principles:
    1. Horizon labels use different forward windows (1d, 3d, 5d, 10d)
    2. Volatility normalization for fair comparison
    3. Regime conditioning for context-aware predictions
    4. Binary classification (1 = up, 0 = down)
    """

    @staticmethod
    def _detect_regime(df: pd.DataFrame, idx: int) -> str:
        """
        Detect market regime at given index.

        Regimes:
        - trend_up: Price above MA50, low volatility
        - trend_down: Price below MA50, low volatility
        - high_vol: Volatility > 2.5% annualized
        - low_vol: Volatility < 1.0% annualized
        - choppy: Everything else (no clear trend, moderate vol)

        Args:
            df: Price DataFrame with 'close' column
            idx: Index position

        Returns:
            Regime string
        """
        if idx < 50:
            return 'choppy'

        window = df.iloc[max(0, idx-60):idx+1]
        if len(window) < 20:
            return 'choppy'

        close = window['close'].iloc[-1]
        ma_50 = window['close'].rolling(50).mean().iloc[-1]
        returns = np.log(window['close'] / window['close'].shift(1)).dropna()
        vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0.015

        # High/Low volatility
        if vol_20 > 0.025:
            return 'high_vol'
        elif vol_20 < 0.010:
            return 'low_vol'

        # Trend detection
        if close > ma_50 * 1.02:
            return 'trend_up'
        elif close < ma_50 * 0.98:
            return 'trend_down'
        else:
            return 'choppy'

    @staticmethod
    def build_horizon_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build horizon-specific labels (1d, 3d, 5d, 10d).

        Each horizon predicts forward returns over its specific window.
        This creates TRUE model diversity - 1d model learns different patterns than 10d model.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with horizon-specific binary labels
        """
        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]

        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        out = pd.DataFrame(index=df.index)

        # Compute forward returns for each horizon
        out['forward_ret_1d'] = np.log(df['close'].shift(-1) / df['close'])
        out['forward_ret_3d'] = np.log(df['close'].shift(-3) / df['close'])
        out['forward_ret_5d'] = np.log(df['close'].shift(-5) / df['close'])
        out['forward_ret_10d'] = np.log(df['close'].shift(-10) / df['close'])

        # Compute rolling volatility for normalization
        ret_1d = np.log(df['close'] / df['close'].shift(1))
        vol_20 = ret_1d.rolling(20).std()

        # Volatility-adjusted thresholds (adaptive based on market conditions)
        # Use 0.25 * vol as threshold for "significant" move
        threshold_1d = vol_20 * 0.25
        threshold_3d = vol_20 * 0.4
        threshold_5d = vol_20 * 0.5
        threshold_10d = vol_20 * 0.7

        # Binary labels: 1 if forward return > threshold, else 0
        out['label_up_1d'] = (out['forward_ret_1d'] > threshold_1d).astype(int)
        out['label_up_3d'] = (out['forward_ret_3d'] > threshold_3d).astype(int)
        out['label_up_5d'] = (out['forward_ret_5d'] > threshold_5d).astype(int)
        out['label_up_10d'] = (out['forward_ret_10d'] > threshold_10d).astype(int)

        # Drop forward return columns (not needed for training)
        out = out[['label_up_1d', 'label_up_3d', 'label_up_5d', 'label_up_10d']]

        # Drop rows with NaN labels
        out.dropna(inplace=True)

        return out

    @staticmethod
    def build_regime_labels(df: pd.DataFrame, regime_column: Optional[str] = None) -> pd.DataFrame:
        """
        Build regime-conditioned labels.

        If regime_column provided, use it. Otherwise, detect regime automatically.

        Args:
            df: DataFrame with 'close' column
            regime_column: Optional column name containing regime labels

        Returns:
            DataFrame with regime-specific labels
        """
        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]

        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        out = pd.DataFrame(index=df.index)

        # Detect regime if not provided
        if regime_column is None or regime_column not in df.columns:
            regimes = []
            for i in range(len(df)):
                regimes.append(TargetBuilderV2._detect_regime(df, i))
            out['regime'] = regimes
        else:
            out['regime'] = df[regime_column]

        # Compute forward returns (using 5d as default for regime models)
        out['forward_ret'] = np.log(df['close'].shift(-5) / df['close'])

        # Compute volatility for normalization
        ret_1d = np.log(df['close'] / df['close'].shift(1))
        vol_20 = ret_1d.rolling(20).std()

        # Regime-specific threshold adjustment
        # High-vol regimes need higher threshold, low-vol need lower threshold
        base_threshold = vol_20 * 0.5

        # Build adjusted threshold series properly
        adjusted_threshold = pd.Series(index=out.index, dtype=float)
        for idx in out.index:
            regime_val = out.loc[idx, 'regime']
            if regime_val == 'high_vol':
                adjusted_threshold.loc[idx] = base_threshold.loc[idx] * 1.5
            elif regime_val == 'low_vol':
                adjusted_threshold.loc[idx] = base_threshold.loc[idx] * 0.7
            else:
                adjusted_threshold.loc[idx] = base_threshold.loc[idx]

        # Create binary labels for each regime
        out['label_regime_trend_up'] = ((out['regime'] == 'trend_up') & (out['forward_ret'] > adjusted_threshold)).astype(int)
        out['label_regime_trend_down'] = ((out['regime'] == 'trend_down') & (out['forward_ret'] > adjusted_threshold)).astype(int)
        out['label_regime_choppy'] = ((out['regime'] == 'choppy') & (out['forward_ret'] > adjusted_threshold)).astype(int)
        out['label_regime_high_vol'] = ((out['regime'] == 'high_vol') & (out['forward_ret'] > adjusted_threshold)).astype(int)
        out['label_regime_low_vol'] = ((out['regime'] == 'low_vol') & (out['forward_ret'] > adjusted_threshold)).astype(int)

        # Also create a general regime-agnostic label
        out['label_regime_up'] = (out['forward_ret'] > base_threshold).astype(int)

        # Keep only label columns
        label_cols = [col for col in out.columns if col.startswith('label_')]
        out = out[label_cols + ['regime']]

        # Drop NaN
        out.dropna(inplace=True)

        return out

    @staticmethod
    def build_all_labels(df: pd.DataFrame) -> tuple:
        """
        Build both horizon and regime labels.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (horizon_labels_df, regime_labels_df)
        """
        horizon_labels = TargetBuilderV2.build_horizon_labels(df)
        regime_labels = TargetBuilderV2.build_regime_labels(df)

        return horizon_labels, regime_labels

    @staticmethod
    def get_label_for_horizon(labels_df: pd.DataFrame, horizon: str) -> pd.Series:
        """
        Extract label for specific horizon.

        Args:
            labels_df: DataFrame from build_horizon_labels()
            horizon: '1d', '3d', '5d', or '10d'

        Returns:
            Series of binary labels
        """
        col_name = f'label_up_{horizon}'
        if col_name not in labels_df.columns:
            raise ValueError(f"Horizon {horizon} not found. Available: {list(labels_df.columns)}")

        return labels_df[col_name]

    @staticmethod
    def get_label_for_regime(labels_df: pd.DataFrame, regime: str) -> pd.Series:
        """
        Extract label for specific regime.

        Args:
            labels_df: DataFrame from build_regime_labels()
            regime: 'trend_up', 'trend_down', 'choppy', 'high_vol', 'low_vol'

        Returns:
            Series of binary labels
        """
        col_name = f'label_regime_{regime}'
        if col_name not in labels_df.columns:
            raise ValueError(f"Regime {regime} not found. Available: {list(labels_df.columns)}")

        return labels_df[col_name]


# Convenience functions
def build_horizon_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Build horizon-specific labels"""
    return TargetBuilderV2.build_horizon_labels(df)


def build_regime_labels(df: pd.DataFrame, regime_column: Optional[str] = None) -> pd.DataFrame:
    """Build regime-conditioned labels"""
    return TargetBuilderV2.build_regime_labels(df, regime_column)


__all__ = ['TargetBuilderV2', 'build_horizon_labels', 'build_regime_labels']
