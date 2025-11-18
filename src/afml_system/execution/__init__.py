"""
PRADO9_EVO Execution Module

Module H - Execution Engine (Market Simulator + Risk Layer + Trade Lifecycle)

Components:
- Trade Types: TradeIntent, TradeFill, PortfolioState
- Risk Engine: Multi-layered risk firewall
- Execution Engine: Market simulator with slippage and commissions

Author: PRADO9_EVO Builder
Date: 2025-01-16
"""

from .trade_types import (
    TradeIntent,
    TradeFill,
    PortfolioState,
)

from .risk_engine import (
    RiskEngine,
)

from .execution_engine import (
    ExecutionEngine,
    ExecutionManager,
    evo_execute,
)

__all__ = [
    # Trade Types
    'TradeIntent',
    'TradeFill',
    'PortfolioState',
    # Risk Engine
    'RiskEngine',
    # Execution Engine
    'ExecutionEngine',
    'ExecutionManager',
    'evo_execute',
]
