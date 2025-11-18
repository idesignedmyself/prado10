"""
PRADO9_EVO Risk Management Modules

Module X: ATR Volatility Targeting - Institutional-grade volatility normalization
Module Y: Position Scaling Engine - Confidence-based position adjustments
"""

from .atr_target import ATRVolTarget
from .position_scaler import PositionScaler, ScalingFactors

__all__ = ['ATRVolTarget', 'PositionScaler', 'ScalingFactors']
