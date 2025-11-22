"""
PRADO9_EVO Module R — Regime-Based Strategy Selector

Dynamically selects which strategies to activate based on detected market regime.
"""

from .selector import RegimeStrategySelector, RegimeStrategyConfig, DEFAULT_REGIME_MAP
from .regime_detector import RegimeDetector

__all__ = ['RegimeStrategySelector', 'RegimeStrategyConfig', 'DEFAULT_REGIME_MAP', 'RegimeDetector']
