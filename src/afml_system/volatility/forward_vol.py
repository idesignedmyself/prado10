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
