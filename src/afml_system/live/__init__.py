"""
PRADO9_EVO Module J — Live Trading Engine

Complete live trading system integrating all PRADO9_EVO intelligence layers.

Components:
- LiveDataFeed: Real-time market data ingestion
- LiveSignalEngine: Feature + regime + strategy pipeline
- BrokerRouter: Multi-mode execution routing
- LivePortfolio: Persistent portfolio tracking
- LiveLogger: Structured logging
- LiveTradingEngine: Main orchestrator

Integration Hooks:
- evo_live_start(): Start live trading
- evo_live_stop(): Stop live trading
- evo_live_data_feed(): Get live market data
- evo_live_signal(): Generate live signal
- evo_broker_submit(): Submit order
- evo_live_portfolio(): Get/create portfolio
- evo_log_trade(): Log trade

Author: PRADO9_EVO Builder
Date: 2025-01-16
Version: 1.0.0
"""

# Data feed
from .data_feed import (
    LiveDataFeed,
    DataFeedResult,
    evo_live_data_feed
)

# Signal engine
from .signal_engine import (
    LiveSignalEngine,
    LiveSignalResult,
    StrategyResult,
    FeatureBuilder,
    RegimeDetector,
    momentum_strategy,
    mean_reversion_strategy,
    evo_live_signal
)

# Broker router
from .broker_router import (
    BrokerRouter,
    Order,
    Fill,
    evo_broker_submit
)

# Live portfolio
from .live_portfolio import (
    LivePortfolio,
    TradeFill,
    evo_live_portfolio
)

# Logger
from .logger import (
    LiveLogger,
    LogEntry,
    evo_log_trade
)

# Live engine
from .live_engine import (
    LiveTradingEngine,
    EngineConfig,
    evo_live_start,
    evo_live_stop
)


__all__ = [
    # Data feed
    'LiveDataFeed',
    'DataFeedResult',
    'evo_live_data_feed',

    # Signal engine
    'LiveSignalEngine',
    'LiveSignalResult',
    'StrategyResult',
    'FeatureBuilder',
    'RegimeDetector',
    'momentum_strategy',
    'mean_reversion_strategy',
    'evo_live_signal',

    # Broker router
    'BrokerRouter',
    'Order',
    'Fill',
    'evo_broker_submit',

    # Live portfolio
    'LivePortfolio',
    'TradeFill',
    'evo_live_portfolio',

    # Logger
    'LiveLogger',
    'LogEntry',
    'evo_log_trade',

    # Live engine
    'LiveTradingEngine',
    'EngineConfig',
    'evo_live_start',
    'evo_live_stop',
]


# Module metadata
__version__ = '1.0.0'
__author__ = 'PRADO9_EVO Builder'
__date__ = '2025-01-16'
