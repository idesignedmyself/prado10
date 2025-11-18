"""
PRADO9_EVO Module X — ATR Volatility Targeting

Converts raw position signals into volatility-adjusted exposure for institutional-grade
risk management. This enables stable, high-Sharpe results by normalizing position sizes
to a target volatility level.

Key Features:
- ATR-based volatility estimation (14-period default)
- Position scaling to target volatility
- Leverage capping for safety
- Handles missing/invalid data gracefully

Use Case:
Instead of fixed position sizes, scale positions based on realized volatility:
- High volatility → smaller positions
- Low volatility → larger positions (capped)

This creates more consistent risk exposure across different market regimes.

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


class ATRVolTarget:
    """
    ATR-based volatility targeting for position sizing.

    This class implements institutional-grade risk management by scaling
    position sizes to achieve a target volatility level. This results in:
    - More stable returns across regimes
    - Higher risk-adjusted returns (Sharpe ratio)
    - Automatic leverage reduction during volatile periods

    Example:
        >>> atr_target = ATRVolTarget(target_vol=0.12)
        >>> df["atr"] = atr_target.compute_atr(df)
        >>> scaled_position = atr_target.scale_position(raw_position=1.0, atr=0.02)
        >>> # If ATR is 2%, target is 12%, scale = 12% / 2% = 6x
        >>> # But capped at 3x for safety, so scaled_position = 3.0
    """

    def __init__(
        self,
        target_vol: float = 0.12,
        atr_period: int = 14,
        max_leverage: float = 3.0,
        min_vol_threshold: float = 0.001
    ):
        """
        Initialize ATR Volatility Targeting.

        Args:
            target_vol: Target annualized volatility (default 12%)
            atr_period: ATR calculation period in bars (default 14)
            max_leverage: Maximum leverage multiplier cap (default 3.0x)
            min_vol_threshold: Minimum volatility floor to avoid division issues
        """
        self.target_vol = target_vol
        self.atr_period = atr_period
        self.max_leverage = max_leverage
        self.min_vol_threshold = min_vol_threshold

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Average True Range (ATR) for volatility estimation.

        ATR measures market volatility by decomposing the range of price movement.
        True Range is the maximum of:
        - High - Low (current bar range)
        - |High - Previous Close| (gap up range)
        - |Low - Previous Close| (gap down range)

        ATR is the moving average of True Range over N periods.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series containing ATR values

        Example:
            >>> df["atr"] = atr_target.compute_atr(df)
            >>> print(df[["close", "atr"]].tail())
        """
        # Calculate True Range components
        high_low = df["high"] - df["low"]
        high_prev_close = (df["high"] - df["close"].shift(1)).abs()
        low_prev_close = (df["low"] - df["close"].shift(1)).abs()

        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

        # ATR is the rolling mean of True Range
        atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()

        return atr

    def compute_atr_percent(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ATR as a percentage of close price.

        This normalizes ATR to price level, making it directly comparable
        to the target_vol parameter.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns

        Returns:
            Series containing ATR as percentage of close price

        Example:
            >>> df["atr_pct"] = atr_target.compute_atr_percent(df)
            >>> print(f"Current volatility: {df['atr_pct'].iloc[-1]:.2%}")
        """
        atr = self.compute_atr(df)
        close = df["close"]

        # Avoid division by zero
        atr_percent = atr / close.replace(0, np.nan)

        return atr_percent

    def scale_position(
        self,
        raw_position: float,
        atr: Union[float, None],
        close_price: Optional[float] = None
    ) -> float:
        """
        Scale position size to achieve target volatility.

        Formula:
            scaled_position = raw_position * (target_vol / realized_vol)
            with cap at max_leverage

        Args:
            raw_position: Unscaled position size (e.g., 1.0 for 100% long)
            atr: Average True Range value
            close_price: Current close price (optional, for percentage calculation)

        Returns:
            Volatility-adjusted position size, capped at max_leverage

        Example:
            >>> # Low volatility (1%) → scale up
            >>> scaled = atr_target.scale_position(1.0, atr=0.01, close_price=100)
            >>> # scaled = 1.0 * (0.12 / 0.01) = 12.0, but capped at 3.0
            >>> print(f"Scaled position: {scaled}x")  # 3.0x

            >>> # High volatility (20%) → scale down
            >>> scaled = atr_target.scale_position(1.0, atr=0.20, close_price=100)
            >>> # scaled = 1.0 * (0.12 / 0.20) = 0.6
            >>> print(f"Scaled position: {scaled}x")  # 0.6x
        """
        # Handle invalid ATR values
        if atr is None or np.isnan(atr) or atr <= 0:
            return raw_position

        # Convert ATR to percentage if close_price provided
        if close_price is not None and close_price > 0:
            realized_vol = atr / close_price
        else:
            realized_vol = atr

        # Apply minimum threshold to avoid extreme leverage
        realized_vol = max(realized_vol, self.min_vol_threshold)

        # Calculate volatility scaling factor
        vol_scalar = self.target_vol / realized_vol

        # Cap at maximum leverage for safety
        vol_scalar = min(vol_scalar, self.max_leverage)

        # Apply scaling to raw position
        scaled_position = raw_position * vol_scalar

        return scaled_position

    def scale_position_series(
        self,
        raw_positions: pd.Series,
        atr_series: pd.Series,
        close_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Scale a series of positions based on corresponding ATR values.

        Vectorized version of scale_position() for backtesting efficiency.

        Args:
            raw_positions: Series of unscaled position sizes
            atr_series: Series of ATR values
            close_series: Series of close prices (optional)

        Returns:
            Series of volatility-adjusted positions

        Example:
            >>> df["raw_signal"] = allocator_positions
            >>> df["atr"] = atr_target.compute_atr(df)
            >>> df["scaled_position"] = atr_target.scale_position_series(
            ...     df["raw_signal"], df["atr"], df["close"]
            ... )
        """
        # Calculate realized volatility
        if close_series is not None:
            realized_vol = atr_series / close_series.replace(0, np.nan)
        else:
            realized_vol = atr_series

        # Apply minimum threshold
        realized_vol = realized_vol.clip(lower=self.min_vol_threshold)

        # Calculate volatility scalars
        vol_scalars = (self.target_vol / realized_vol).clip(upper=self.max_leverage)

        # Apply scaling
        scaled_positions = raw_positions * vol_scalars

        # Fill NaN values with raw positions (no scaling when ATR unavailable)
        scaled_positions = scaled_positions.fillna(raw_positions)

        return scaled_positions

    def get_current_leverage(
        self,
        atr: float,
        close_price: Optional[float] = None
    ) -> float:
        """
        Calculate current leverage multiplier based on volatility.

        Useful for monitoring and debugging volatility targeting behavior.

        Args:
            atr: Current ATR value
            close_price: Current close price (optional)

        Returns:
            Current leverage multiplier (e.g., 2.5 means 2.5x leverage)

        Example:
            >>> current_leverage = atr_target.get_current_leverage(
            ...     atr=0.015, close_price=100
            ... )
            >>> print(f"Current leverage: {current_leverage:.2f}x")
        """
        if atr is None or np.isnan(atr) or atr <= 0:
            return 1.0

        if close_price is not None and close_price > 0:
            realized_vol = atr / close_price
        else:
            realized_vol = atr

        realized_vol = max(realized_vol, self.min_vol_threshold)
        leverage = min(self.target_vol / realized_vol, self.max_leverage)

        return leverage

    def __repr__(self) -> str:
        """String representation of ATRVolTarget configuration."""
        return (
            f"ATRVolTarget(target_vol={self.target_vol:.1%}, "
            f"atr_period={self.atr_period}, "
            f"max_leverage={self.max_leverage:.1f}x)"
        )
