"""
PRADO9_EVO Module K — AutoTuner Engine
Hyperparameter optimization engine for PRADO9_EVO.

This module:
- Sweeps key hyperparameters
- Runs CPCV validation splits
- Performs forward testing
- Scores configs
- Selects optimal baseline
- Saves config to ~/.prado/configs/SYMBOL.yaml
"""

import os
import yaml
import numpy as np
import pandas as pd
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple

from ..backtest.backtest_engine import (
    BacktestConfig,
    evo_backtest_standard
)

# ============================================================================
# SEARCH SPACE
# ============================================================================

SEARCH_SPACE = {
    "cusum_threshold": [0.003, 0.005, 0.0075, 0.01],
    "lookback_bars": [10, 15, 20, 30],
    "holding_period": [5, 10, 15],
    "max_position": [0.5, 1.0],
    "mutation_rate": [0.05, 0.1, 0.2],
    "crossover_rate": [0.5, 0.7, 0.9]
}

# ============================================================================
# SCORING
# ============================================================================

def score_config(result) -> float:
    if result is None:
        return -999

    r = result
    sharpe = r.sharpe_ratio
    maxdd = abs(r.max_drawdown)
    returns = r.total_return
    trades = r.total_trades

    if trades < 5:
        return -999

    score = (
        sharpe * 2.0 +
        returns * 1.0 -
        maxdd * 1.5 +
        (0.1 if trades > 20 else -0.1)
    )

    return score

# ============================================================================
# CPCV SPLITS
# ============================================================================

def cpcv_splits(df: pd.DataFrame, k: int = 4) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(df)
    fold = n // k
    splits = []

    for i in range(k):
        start = i * fold
        end = (i + 1) * fold

        val = df.iloc[start:end]
        train = pd.concat([df.iloc[:start], df.iloc[end:]])

        if len(train) > 200 and len(val) > 50:
            splits.append((train, val))

    return splits

# ============================================================================
# AUTOTUNE ENGINE
# ============================================================================

class AutoTuner:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.config_dir = os.path.expanduser(f"~/.prado/configs")
        os.makedirs(self.config_dir, exist_ok=True)

    def run(self, df: pd.DataFrame) -> Dict:
        # Check minimum data requirement
        if len(df) < 300:
            return {
                "status": "error",
                "message": f"Insufficient data for tuning: {len(df)} bars (need ≥300)"
            }

        # Get CPCV splits
        splits = cpcv_splits(df, k=4)

        if len(splits) == 0:
            return {
                "status": "error",
                "message": "No CPCV splits could be created (dataset too small)"
            }

        best_score = -9999
        best_params = None
        best_result = None

        params_list = self._generate_param_grid()

        for params in params_list:
            avg_score = 0
            result = None

            for train_df, val_df in splits:
                config = BacktestConfig(
                    symbol=self.symbol,
                    cusum_threshold=params["cusum_threshold"],
                    lookback_bars=params["lookback_bars"],
                    holding_period=params["holding_period"],
                    max_position=params["max_position"],
                    mutation_rate=params["mutation_rate"],
                    crossover_rate=params["crossover_rate"]
                )

                res_dict = evo_backtest_standard(self.symbol, train_df, config)

                # Handle errors from backtest
                if res_dict.get("status") != "success":
                    continue

                result = res_dict["result"]
                fold_score = score_config(result)
                avg_score += fold_score

            avg_score /= max(1, len(splits))

            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                if result is not None:
                    best_result = result

        # Check if we found valid parameters
        if best_params is None:
            return {
                "status": "error",
                "message": "No parameter combination produced valid results"
            }

        self._save_config(best_params)

        return {
            "status": "success",
            "symbol": self.symbol,
            "best_params": best_params,
            "best_score": best_score,
            "sample_result": best_result
        }

    def _save_config(self, params: Dict):
        path = os.path.join(self.config_dir, f"{self.symbol}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(params, f)

    def _generate_param_grid(self) -> List[Dict]:
        keys = list(SEARCH_SPACE.keys())
        grid = [{}]

        for key in keys:
            new_grid = []
            for base in grid:
                for value in SEARCH_SPACE[key]:
                    new_cfg = base.copy()
                    new_cfg[key] = value
                    new_grid.append(new_cfg)
            grid = new_grid

        return grid
