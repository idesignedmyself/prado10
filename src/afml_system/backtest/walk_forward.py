"""
PRADO9_EVO — Walk-Forward Engine (Fixed & Deterministic)

This version properly:
- Builds rolling train/test windows
- Runs BacktestEngine for each window
- Aggregates metrics into a final combined result dict
- Preserves deterministic behavior
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .backtest_engine import BacktestEngine, BacktestConfig


@dataclass
class WFWindowResult:
    total_return: float
    sharpe: float
    sortino: float
    maxdd: float
    trades: int


class WalkForwardEngine:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig(symbol="WF")

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        train_window: int = 252,
        test_window: int = 63,
    ) -> Dict[str, Any]:

        if len(df) < train_window + test_window:
            return {
                "status": "error",
                "error": "Insufficient data for walk-forward windows",
                "num_folds": 0,
                "aggregated": {},
            }

        results: List[WFWindowResult] = []

        idx = train_window

        while idx + test_window < len(df):

            train_df = df.iloc[idx - train_window : idx]
            test_df = df.iloc[idx : idx + test_window]

            engine = BacktestEngine(config=self.config)
            result = engine.run_standard(symbol, pd.concat([train_df, test_df]))

            # extract metrics
            wf_res = WFWindowResult(
                total_return=result.total_return,
                sharpe=result.sharpe_ratio,
                sortino=result.sortino_ratio,
                maxdd=result.max_drawdown,
                trades=result.total_trades,
            )

            results.append(wf_res)

            idx += test_window

        if len(results) == 0:
            return {
                "status": "error",
                "error": "Walk-forward produced no valid windows",
                "num_folds": 0,
                "aggregated": {},
            }

        # ========================
        # Aggregate window results
        # ========================

        total_return = np.mean([r.total_return for r in results])
        sharpe_mean = np.mean([r.sharpe for r in results])
        sortino_mean = np.mean([r.sortino for r in results])
        max_dd = np.min([r.maxdd for r in results])  # worst drawdown
        total_trades = sum([r.trades for r in results])
        consistency = (np.sum([1 for r in results if r.total_return > 0]) / len(results)) * 100

        return {
            "status": "success",
            "symbol": symbol,
            "num_folds": len(results),
            "aggregated": {
                "total_return": float(total_return),
                "sharpe_mean": float(sharpe_mean),
                "sortino_mean": float(sortino_mean),
                "max_drawdown": float(max_dd),
                "total_trades": int(total_trades),
                "consistency_pct": float(consistency),
            },
        }
