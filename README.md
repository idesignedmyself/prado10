# PRADO10 - Evolutionary Quantitative Trading System

An advanced quantitative trading framework combining AFML (Advances in Financial Machine Learning) methodology with evolutionary algorithms and adaptive learning - a mini-Medallion architecture for self-improving, regime-adaptive trading.

## Overview

PRADO10 is a two-part system:
- **AFML Core**: Complete research/trading pipeline with proper labeling, feature engineering, and backtesting
- **EVO Engine**: Self-optimizing layer with genetic algorithms, Thompson sampling, and continuous learning

## Key Features

- Triple-barrier labeling with meta-labels
- Multi-regime detection and strategy adaptation
- Hierarchical Thompson Sampling for strategy selection
- Evolutionary strategy genome with genetic algorithms
- Microstructure-aware execution (OFI, VPIN, Kyle Lambda)
- Advanced backtesting suite (walk-forward, crisis testing, Monte Carlo)

## CLI Usage

```bash
# Train models
prado train <symbol> start %mm %dd %yyyy end %mm %dd %yyyy

# Generate predictions
prado predict <symbol>

# Run backtests
prado backtest <symbol> --standard
prado backtest <symbol> --walk-forward
prado backtest <symbol> --crisis
prado backtest <symbol> --monte-carlo <n>
```

## Architecture

See [prado9_evo.md](./prado9_evo.md) for complete architecture documentation.

## Requirements

- Python 3.8+
- MacOS compatible
