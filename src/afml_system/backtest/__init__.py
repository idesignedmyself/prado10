"""
PRADO9_EVO Backtest Module

Module I - Backtest Engine (Full AFML + EVO Integration)

Components:
- BacktestEngine: Main historical simulation engine
- WalkForwardEngine: Rolling window walk-forward optimizer
- CrisisStressEngine: Crisis period stress testing
- MonteCarloEngine: Statistical skill assessment
- BacktestReportBuilder: Comprehensive report builder

Author: PRADO9_EVO Builder
Date: 2025-01-17
Version: 1.0.0
"""

from .backtest_engine import (
    BacktestEngine,
    evo_backtest_standard,
    evo_backtest_walk_forward,
    evo_backtest_crisis,
    evo_backtest_monte_carlo,
    evo_backtest_comprehensive,
)

from .walk_forward import (
    WalkForwardEngine,
)

from .crisis_stress import (
    CrisisStressEngine,
)

from .monte_carlo import (
    MonteCarloEngine,
)

from .reporting import (
    BacktestReportBuilder,
)

__all__ = [
    # Main Engine
    'BacktestEngine',
    # Integration Hooks
    'evo_backtest_standard',
    'evo_backtest_walk_forward',
    'evo_backtest_crisis',
    'evo_backtest_monte_carlo',
    'evo_backtest_comprehensive',
    # Specialized Engines
    'WalkForwardEngine',
    'CrisisStressEngine',
    'MonteCarloEngine',
    # Reporting
    'BacktestReportBuilder',
]
