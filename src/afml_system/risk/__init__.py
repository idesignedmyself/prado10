"""
PRADO9_EVO Risk Management Modules

Module X: ATR Volatility Targeting - Institutional-grade volatility normalization
Module Y: Position Scaling Engine - Confidence-based position adjustments
Module Y2: Adaptive Confidence Scaling - Regime-aware confidence threshold determination
"""

from .atr_target import ATRVolTarget
from .position_scaler import PositionScaler, ScalingFactors
from .adaptive_confidence import AdaptiveConfidence, ConfidenceThresholds

__all__ = [
    'ATRVolTarget',
    'PositionScaler',
    'ScalingFactors',
    'AdaptiveConfidence',
    'ConfidenceThresholds'
]
