"""
PRADO9_EVO Backtest Module

Module I - Backtest Engine (Full AFML + EVO Integration)

Components:
- BacktestEngine: Main historical simulation engine
- WalkForwardEngine: Rolling window walk-forward optimizer
- CrisisStressEngine: Crisis period stress testing
- MonteCarloEngine: Statistical skill assessment
- MC2Engine: Monte Carlo Robustness Engine (Module MC2)
- BacktestReportBuilder: Comprehensive report builder

Author: PRADO9_EVO Builder
Date: 2025-01-17
Version: 1.0.0
"""

from .backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    evo_backtest_standard,
    evo_backtest_walk_forward,
    evo_backtest_crisis,
    evo_backtest_monte_carlo,
    evo_backtest_mc2,
    evo_backtest_unified_adaptive,
    evo_backtest_comprehensive,
)

from .walk_forward import (
    WalkForwardEngine,
)

from .crisis_stress import (
    CrisisStressEngine,
)

from .crisis_stress_cr2 import (
    EnhancedCrisisStressEngine,
    MultiCrisisDetector,
    SyntheticCrisisGenerator,
    CrisisType,
    DetectedCrisis,
    CrisisSignature,
)

from .monte_carlo import (
    MonteCarloEngine,
)

from .monte_carlo_mc2 import (
    MC2Engine,
    BlockBootstrappedMCSimulator,
    TurbulenceStressTester,
    SignalCorruptionTester,
    TurbulenceLevel,
    CorruptionType,
    MC2Result,
)

from .reporting import (
    BacktestReportBuilder,
)

from .combined_backtest import (
    evo_backtest_combined,
)

__all__ = [
    # Main Engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    # Integration Hooks
    'evo_backtest_standard',
    'evo_backtest_walk_forward',
    'evo_backtest_crisis',
    'evo_backtest_monte_carlo',
    'evo_backtest_mc2',
    'evo_backtest_unified_adaptive',  # BUILDER PROMPT FINAL
    'evo_backtest_comprehensive',
    'evo_backtest_combined',  # Combined standard + walk-forward
    # Specialized Engines
    'WalkForwardEngine',
    'CrisisStressEngine',
    'MonteCarloEngine',
    # Module MC2
    'MC2Engine',
    'BlockBootstrappedMCSimulator',
    'TurbulenceStressTester',
    'SignalCorruptionTester',
    'TurbulenceLevel',
    'CorruptionType',
    'MC2Result',
    # Module CR2
    'EnhancedCrisisStressEngine',
    'MultiCrisisDetector',
    'SyntheticCrisisGenerator',
    'CrisisType',
    'DetectedCrisis',
    'CrisisSignature',
    # Reporting
    'BacktestReportBuilder',
]
