"""
PRADO9_EVO Module X2 — Forward-Looking Volatility Engine

Implements forward-looking volatility estimation using:
- Realized volatility (historical)
- Regime-adjusted volatility
- GARCH(1,1) volatility forecasting with fallback

This module enhances Module X (ATR targeting) with forward-looking
volatility forecasts for more adaptive position sizing.

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import warnings

# Try to import arch for GARCH(1,1)
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn(
        "arch package not available. GARCH forecasting will use simplified implementation. "
        "Install with: pip install arch",
        UserWarning
    )


def realized_volatility(
    returns: Union[pd.Series, np.ndarray],
    window: int = 21,
    annualization_factor: int = 252
) -> float:
    """
    Calculate realized volatility from historical returns.

    Uses exponentially weighted moving average for more recent emphasis.

    Args:
        returns: Historical returns (can be pd.Series or np.ndarray)
        window: Lookback window for volatility calculation (default: 21 days)
        annualization_factor: Factor to annualize volatility (default: 252 trading days)

    Returns:
        Annualized realized volatility as a float

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01])
        >>> vol = realized_volatility(returns, window=3)
        >>> print(f"Realized vol: {vol:.2%}")
    """
    # Convert to pandas Series if numpy array
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Remove NaN values
    returns = returns.dropna()

    if len(returns) < 2:
        # Insufficient data, return default 15% volatility
        return 0.15

    # Use only the last 'window' periods
    returns_window = returns.tail(window)

    if len(returns_window) < 2:
        return 0.15

    # Calculate exponentially weighted standard deviation
    # Higher span = more weight on recent observations
    ewm_std = returns_window.ewm(span=window, adjust=False).std().iloc[-1]

    # Annualize
    realized_vol = ewm_std * np.sqrt(annualization_factor)

    # Sanity check: cap between 5% and 200%
    realized_vol = max(0.05, min(2.00, realized_vol))

    return float(realized_vol)


def ewma_vol_forecast(
    returns: Union[pd.Series, np.ndarray],
    window: int = 21,
    annualization_factor: int = 252
) -> float:
    """
    PATCH 3: EWMA-21 volatility forecast (fallback method 1).

    Exponentially Weighted Moving Average volatility estimate.
    This is more responsive than simple rolling std but more stable than GARCH.

    Args:
        returns: Historical returns
        window: EWMA window (default: 21 days)
        annualization_factor: Annualization factor (default: 252)

    Returns:
        EWMA volatility forecast

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015])
        >>> vol = ewma_vol_forecast(returns, window=21)
        >>> print(f"EWMA vol: {vol:.2%}")
    """
    # Convert to pandas Series if numpy array
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Remove NaN values
    returns = returns.dropna()

    if len(returns) < 2:
        return 0.015  # Default 1.5% if insufficient data

    # Calculate EWMA standard deviation
    ewm_std = returns.ewm(span=window, adjust=False).std().iloc[-1]

    # Annualize
    ewma_vol = ewm_std * np.sqrt(annualization_factor)

    # Apply floor and cap
    ewma_vol = max(0.005, min(2.00, ewma_vol))

    return float(ewma_vol)


def atr_vol_fallback(
    df: pd.DataFrame,
    window: int = 14,
    annualization_factor: int = 252
) -> float:
    """
    PATCH 3: ATR-14 volatility fallback (final fallback method).

    Average True Range-based volatility estimate.
    Most robust fallback that only requires OHLC data.

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns
        window: ATR window (default: 14 days)
        annualization_factor: Annualization factor (default: 252)

    Returns:
        ATR-based volatility estimate

    Example:
        >>> df = pd.DataFrame({'high': [...], 'low': [...], 'close': [...]})
        >>> vol = atr_vol_fallback(df, window=14)
        >>> print(f"ATR vol: {vol:.2%}")
    """
    if len(df) < 2:
        return 0.015  # Default 1.5% if insufficient data

    # Ensure we have required columns
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return 0.015  # Fallback to default

    # Calculate True Range
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    high_low = df['high'] - df['low']
    high_prev_close = (df['high'] - df['close'].shift(1)).abs()
    low_prev_close = (df['low'] - df['close'].shift(1)).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    # Calculate ATR (simple moving average of TR)
    atr = true_range.rolling(window=window, min_periods=1).mean().iloc[-1]

    # Convert ATR to volatility estimate
    # ATR is in price units, convert to percentage of price
    current_price = df['close'].iloc[-1]
    if current_price > 0:
        atr_pct = atr / current_price
    else:
        return 0.015

    # Annualize
    atr_vol = atr_pct * np.sqrt(annualization_factor)

    # Apply floor and cap
    atr_vol = max(0.005, min(2.00, atr_vol))

    return float(atr_vol)


def regime_adjusted_vol(
    base_vol: float,
    regime: str,
    adjustment_factor: Optional[dict] = None
) -> float:
    """
    Adjust volatility estimate based on detected market regime.

    Different regimes have different volatility characteristics:
    - HIGH_VOL: Increase volatility estimate (expect higher future vol)
    - LOW_VOL: Decrease volatility estimate (expect lower future vol)
    - TRENDING: Stable volatility
    - MEAN_REVERTING: Moderate volatility
    - NORMAL: No adjustment

    Args:
        base_vol: Base volatility estimate (e.g., from realized_volatility)
        regime: Current market regime (HIGH_VOL, LOW_VOL, TRENDING, etc.)
        adjustment_factor: Optional custom adjustment factors per regime
                          (default: uses standard factors)

    Returns:
        Regime-adjusted volatility estimate

    Example:
        >>> base_vol = 0.15
        >>> adjusted = regime_adjusted_vol(base_vol, "HIGH_VOL")
        >>> print(f"Adjusted vol: {adjusted:.2%}")
    """
    # Default regime adjustment factors
    # HIGH_VOL: expect vol to persist (increase forecast)
    # LOW_VOL: expect vol to persist (decrease forecast)
    # TRENDING: stable, slight increase
    # MEAN_REVERTING: moderate, slight decrease
    # NORMAL: no adjustment
    default_factors = {
        "HIGH_VOL": 1.3,        # Increase forecast by 30%
        "LOW_VOL": 0.8,         # Decrease forecast by 20%
        "TRENDING": 1.05,       # Slight increase (momentum)
        "MEAN_REVERTING": 0.95, # Slight decrease (stability)
        "NORMAL": 1.0,          # No adjustment
    }

    # Use custom factors if provided, otherwise use defaults
    factors = adjustment_factor or default_factors

    # Get adjustment factor for the regime
    factor = factors.get(regime, 1.0)  # Default to 1.0 if regime unknown

    # Apply adjustment
    adjusted_vol = base_vol * factor

    # Sanity check: cap between 5% and 200%
    adjusted_vol = max(0.05, min(2.00, adjusted_vol))

    return float(adjusted_vol)


def garch_vol_forecast(
    returns: Union[pd.Series, np.ndarray],
    horizon: int = 1,
    omega: float = 0.000001,
    alpha: float = 0.1,
    beta: float = 0.85,
    fallback_window: int = 21,
    annualization_factor: int = 252
) -> float:
    """
    Forecast volatility using simplified GARCH(1,1) model with fallback.

    GARCH(1,1) model:
        σ²(t+1) = ω + α·ε²(t) + β·σ²(t)

    Where:
        - ω (omega): Long-run variance constant
        - α (alpha): Weight on recent shock (ε²)
        - β (beta): Weight on lagged variance (σ²)
        - Constraint: α + β < 1 (for stationarity)

    If GARCH fails (insufficient data, numerical issues), falls back
    to realized volatility calculation.

    Args:
        returns: Historical returns (pd.Series or np.ndarray)
        horizon: Forecast horizon in periods (default: 1)
        omega: GARCH constant term (default: 0.000001)
        alpha: GARCH alpha parameter (default: 0.1)
        beta: GARCH beta parameter (default: 0.85)
        fallback_window: Window for fallback realized vol (default: 21)
        annualization_factor: Factor to annualize volatility (default: 252)

    Returns:
        Forecasted annualized volatility

    Example:
        >>> returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005])
        >>> forecast = garch_vol_forecast(returns, horizon=1)
        >>> print(f"GARCH forecast: {forecast:.2%}")
    """
    # Convert to pandas Series if numpy array
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Remove NaN values
    returns = returns.dropna()

    # Check if we have enough data
    if len(returns) < 30:
        # Insufficient data for GARCH, fall back to realized volatility
        warnings.warn(
            f"Insufficient data for GARCH ({len(returns)} < 30). "
            f"Falling back to realized volatility.",
            UserWarning
        )
        return realized_volatility(returns, window=fallback_window)

    try:
        # Validate GARCH parameters
        if alpha + beta >= 1.0:
            warnings.warn(
                f"GARCH parameters violate stationarity (α+β={alpha+beta:.3f} >= 1). "
                f"Falling back to realized volatility.",
                UserWarning
            )
            return realized_volatility(returns, window=fallback_window)

        # Step 1: Initialize variance (use sample variance)
        sigma2 = returns.var()

        # Step 2: Iterate through returns to update variance
        # (simplified GARCH without full ML estimation)
        for ret in returns.values[-100:]:  # Use last 100 periods for speed
            # Update variance using GARCH(1,1) formula
            # σ²(t+1) = ω + α·ε²(t) + β·σ²(t)
            epsilon_squared = ret ** 2  # Shock squared
            sigma2 = omega + alpha * epsilon_squared + beta * sigma2

        # Step 3: Forecast h-step ahead variance
        # For GARCH(1,1), h-step forecast converges to long-run variance
        # σ²(t+h) = V_L + (α+β)^(h-1) * (σ²(t+1) - V_L)
        # where V_L = ω / (1 - α - β) is the long-run variance

        long_run_var = omega / (1 - alpha - beta)

        if horizon == 1:
            forecast_var = sigma2
        else:
            # Multi-step forecast
            forecast_var = long_run_var + (alpha + beta) ** (horizon - 1) * (sigma2 - long_run_var)

        # Convert variance to volatility and annualize
        forecast_vol = np.sqrt(forecast_var) * np.sqrt(annualization_factor)

        # Sanity check: cap between 5% and 200%
        forecast_vol = max(0.05, min(2.00, forecast_vol))

        return float(forecast_vol)

    except Exception as e:
        # If GARCH fails for any reason, fall back to realized volatility
        warnings.warn(
            f"GARCH forecasting failed: {e}. Falling back to realized volatility.",
            UserWarning
        )
        return realized_volatility(returns, window=fallback_window)


def forward_vol_forecast(
    df: pd.DataFrame,
    use_arch_garch: bool = True,
    min_vol_floor: float = 0.005,
    annualization_factor: int = 252
) -> float:
    """
    PATCH 3: Unified forward volatility forecast with cascading fallbacks.

    Implements 3-tier fallback system:
    1. GARCH(1,1) using arch package (if available and sufficient data)
    2. EWMA-21 (if GARCH fails or unavailable)
    3. ATR-14 (final fallback requiring only OHLC data)

    All forecasts are clamped to minimum floor of 0.005 (0.5%).

    Args:
        df: OHLCV DataFrame with at minimum 'close' column (preferably also 'high', 'low')
        use_arch_garch: Whether to attempt GARCH using arch package (default: True)
        min_vol_floor: Minimum volatility floor (default: 0.005 = 0.5%)
        annualization_factor: Annualization factor (default: 252)

    Returns:
        1-step ahead volatility forecast (annualized)

    Example:
        >>> df = pd.DataFrame({'close': [...], 'high': [...], 'low': [...]})
        >>> vol_forecast = forward_vol_forecast(df)
        >>> print(f"Vol forecast: {vol_forecast:.2%}")
    """
    # Calculate returns for GARCH and EWMA
    returns = df['close'].pct_change().dropna()

    # Tier 1: Try GARCH(1,1) using arch package
    if use_arch_garch and ARCH_AVAILABLE and len(returns) >= 100:
        try:
            # Prepare returns (arch expects percentage returns scaled to 100)
            returns_scaled = returns * 100

            # Fit GARCH(1,1) model
            # p=1 (ARCH order), q=1 (GARCH order)
            model = arch_model(
                returns_scaled,
                vol='Garch',
                p=1,
                q=1,
                mean='Zero',  # Zero mean (we only care about volatility)
                rescale=False
            )

            # Fit model (suppress output)
            fitted = model.fit(disp='off', show_warning=False)

            # Forecast 1-step ahead variance
            forecast = fitted.forecast(horizon=1, reindex=False)
            forecast_var = forecast.variance.values[-1, 0]

            # Convert back to decimal returns and annualize
            # forecast_var is in (percentage)^2, so divide by 100^2
            forecast_vol = np.sqrt(forecast_var / 10000) * np.sqrt(annualization_factor)

            # Apply floor and cap
            forecast_vol = max(min_vol_floor, min(2.00, forecast_vol))

            return float(forecast_vol)

        except Exception as e:
            # GARCH failed, fall through to EWMA
            warnings.warn(
                f"GARCH (arch package) failed: {e}. Falling back to EWMA.",
                UserWarning
            )

    # Tier 2: EWMA-21 fallback
    try:
        ewma_vol = ewma_vol_forecast(returns, window=21, annualization_factor=annualization_factor)

        # Apply floor
        ewma_vol = max(min_vol_floor, ewma_vol)

        return float(ewma_vol)

    except Exception as e:
        # EWMA failed, fall through to ATR
        warnings.warn(
            f"EWMA fallback failed: {e}. Falling back to ATR.",
            UserWarning
        )

    # Tier 3: ATR-14 final fallback
    try:
        atr_vol = atr_vol_fallback(df, window=14, annualization_factor=annualization_factor)

        # Apply floor (already applied in atr_vol_fallback but double-check)
        atr_vol = max(min_vol_floor, atr_vol)

        return float(atr_vol)

    except Exception as e:
        # All methods failed, return conservative default
        warnings.warn(
            f"All volatility forecast methods failed: {e}. Using default 1.5%.",
            UserWarning
        )
        return max(min_vol_floor, 0.015)


def forward_volatility_estimate(
    returns: Union[pd.Series, np.ndarray],
    regime: Optional[str] = None,
    use_garch: bool = True,
    garch_weight: float = 0.7,
    window: int = 21,
    **kwargs
) -> float:
    """
    Comprehensive forward-looking volatility estimate.

    Combines multiple volatility estimation methods:
    1. GARCH(1,1) forecast (if use_garch=True)
    2. Realized volatility (as fallback or baseline)
    3. Regime adjustment (if regime provided)

    Final estimate is a weighted average of GARCH and realized vol,
    then adjusted for regime.

    Args:
        returns: Historical returns
        regime: Current market regime (optional)
        use_garch: Whether to use GARCH forecasting (default: True)
        garch_weight: Weight on GARCH vs realized vol (default: 0.7)
                     Final = garch_weight * GARCH + (1-garch_weight) * realized
        window: Window for realized volatility (default: 21)
        **kwargs: Additional parameters for garch_vol_forecast

    Returns:
        Forward-looking volatility estimate

    Example:
        >>> returns = pd.Series([...])  # Historical returns
        >>> vol = forward_volatility_estimate(
        ...     returns,
        ...     regime="HIGH_VOL",
        ...     use_garch=True,
        ...     garch_weight=0.7
        ... )
        >>> print(f"Forward vol estimate: {vol:.2%}")
    """
    # Calculate realized volatility (always needed as baseline/fallback)
    realized_vol = realized_volatility(returns, window=window)

    if use_garch and len(returns) >= 30:
        # Calculate GARCH forecast
        garch_vol = garch_vol_forecast(returns, **kwargs)

        # Weighted average of GARCH and realized
        base_vol = garch_weight * garch_vol + (1 - garch_weight) * realized_vol
    else:
        # Use realized volatility only
        base_vol = realized_vol

    # Apply regime adjustment if regime provided
    if regime is not None:
        final_vol = regime_adjusted_vol(base_vol, regime)
    else:
        final_vol = base_vol

    return float(final_vol)


class ForwardVolatilityEngine:
    """
    Forward-looking volatility estimation engine.

    Manages volatility forecasting with GARCH, realized vol, and regime adjustments.
    Can be used as a drop-in replacement for ATR-based volatility targeting.

    Example:
        >>> engine = ForwardVolatilityEngine(use_garch=True)
        >>> returns = df['close'].pct_change().dropna()
        >>> vol_forecast = engine.estimate(returns, regime="HIGH_VOL")
        >>> print(f"Vol forecast: {vol_forecast:.2%}")
    """

    def __init__(
        self,
        use_garch: bool = True,
        garch_weight: float = 0.7,
        window: int = 21,
        garch_params: Optional[dict] = None
    ):
        """
        Initialize Forward Volatility Engine.

        Args:
            use_garch: Whether to use GARCH forecasting (default: True)
            garch_weight: Weight on GARCH vs realized vol (default: 0.7)
            window: Window for realized volatility (default: 21)
            garch_params: Optional custom GARCH parameters
                         (omega, alpha, beta, horizon)
        """
        self.use_garch = use_garch
        self.garch_weight = garch_weight
        self.window = window
        self.garch_params = garch_params or {}

    def estimate(
        self,
        returns: Union[pd.Series, np.ndarray],
        regime: Optional[str] = None
    ) -> float:
        """
        Estimate forward-looking volatility.

        Args:
            returns: Historical returns
            regime: Current market regime (optional)

        Returns:
            Forward volatility estimate
        """
        return forward_volatility_estimate(
            returns=returns,
            regime=regime,
            use_garch=self.use_garch,
            garch_weight=self.garch_weight,
            window=self.window,
            **self.garch_params
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ForwardVolatilityEngine("
            f"use_garch={self.use_garch}, "
            f"garch_weight={self.garch_weight}, "
            f"window={self.window})"
        )
