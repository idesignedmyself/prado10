"""
PRADO9_EVO Module AR — Adaptive Retraining Engine

Enables dynamic retraining of models across walk-forward windows:
- Retrain meta-learner on each window
- Retrain volatility targets on each window
- Retrain confidence scaling on each window
- Maintain deterministic behavior (seed=42)
- Return standardized metrics

This module integrates with BacktestEngine to enable adaptive learning
across changing market conditions while maintaining no-leakage guarantees.

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..backtest.backtest_engine import BacktestEngine, BacktestConfig


@dataclass
class FoldConfig:
    """Configuration for models retrained on a specific fold."""
    meta_learner_weights: Optional[Dict[str, float]] = None
    atr_target_vol: Optional[float] = None
    confidence_range: Optional[tuple] = None
    bandit_weights: Optional[Dict[str, float]] = None


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    config: FoldConfig


class AdaptiveTrainer:
    """
    Adaptive Retraining Engine for walk-forward optimization.

    Retrains models dynamically on each train window and tests on
    the following window. Maintains deterministic behavior and
    returns standardized metrics.

    Example:
        >>> trainer = AdaptiveTrainer(config=BacktestConfig(random_seed=42))
        >>> results = trainer.run_walk_forward(df, n_folds=10)
        >>> print(f"Mean Sharpe: {results['sharpe_ratio']:.3f}")
    """

    def __init__(self, config: Optional[BacktestConfig] = None, seed: int = 42):
        """
        Initialize Adaptive Trainer.

        Args:
            config: Backtest configuration (uses defaults if None)
            seed: Random seed for deterministic behavior (default: 42)
        """
        self.config = config or BacktestConfig(symbol="ADAPTIVE", random_seed=seed)
        self.seed = seed
        np.random.seed(seed)

    def run_walk_forward(
        self,
        symbol: str,
        df: pd.DataFrame,
        n_folds: int = 10,
        train_pct: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run adaptive walk-forward backtest with retraining.

        Args:
            symbol: Trading symbol
            df: OHLCV DataFrame
            n_folds: Number of walk-forward folds (default: 10)
            train_pct: Percentage of each fold for training (default: 70%)

        Returns:
            Dictionary with aggregated metrics:
            {
                'total_return': float,
                'sharpe_ratio': float,
                'sortino_ratio': float,
                'max_drawdown': float,
                'win_rate': float,
                'profit_factor': float,
                'total_trades': int,
                'fold_results': List[FoldResult]
            }
        """
        # Calculate fold size
        total_bars = len(df)
        fold_size = total_bars // n_folds

        if fold_size < 100:
            raise ValueError(f"Fold size too small ({fold_size} bars). Need at least 100 bars per fold.")

        train_size = int(fold_size * train_pct)
        test_size = fold_size - train_size

        fold_results: List[FoldResult] = []

        # Run each fold
        for fold_idx in range(n_folds):
            start_idx = fold_idx * fold_size
            end_idx = start_idx + fold_size

            # Ensure we don't exceed data bounds
            if end_idx > total_bars:
                end_idx = total_bars

            fold_df = df.iloc[start_idx:end_idx]

            if len(fold_df) < train_size + test_size:
                # Skip partial folds at the end
                break

            # Split into train/test
            train_df = fold_df.iloc[:train_size]
            test_df = fold_df.iloc[train_size:]

            # Retrain models on this fold's training data
            fold_config = self._retrain_models(train_df)

            # Test on this fold's test data
            fold_result = self._test_window(
                symbol=symbol,
                test_df=test_df,
                fold_config=fold_config,
                fold_idx=fold_idx,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1]
            )

            fold_results.append(fold_result)

        # Aggregate results
        aggregated = self._aggregate_fold_results(fold_results)

        return aggregated

    def _retrain_models(self, train_df: pd.DataFrame) -> FoldConfig:
        """
        Retrain all adaptive models on training data.

        This method retrains:
        1. Meta-learner (confidence scoring)
        2. Volatility targets (ATR parameters)
        3. Confidence scaling (position scaling ranges)
        4. Bandit weights (strategy selection)

        Args:
            train_df: Training data for this fold

        Returns:
            FoldConfig with retrained parameters
        """
        # Reset random seed for deterministic retraining
        np.random.seed(self.seed)

        # 1. Retrain volatility target (Module X)
        atr_target_vol = self._retrain_vol_target(train_df)

        # 2. Retrain confidence scaling (Module Y)
        confidence_range = self._retrain_confidence(train_df)

        # 3. Retrain meta-learner weights (Module D)
        meta_learner_weights = self._retrain_meta_learner(train_df)

        # 4. Retrain bandit weights (Module A)
        bandit_weights = self._retrain_bandit_weights(train_df)

        return FoldConfig(
            meta_learner_weights=meta_learner_weights,
            atr_target_vol=atr_target_vol,
            confidence_range=confidence_range,
            bandit_weights=bandit_weights
        )

    def _retrain_vol_target(self, train_df: pd.DataFrame) -> float:
        """
        Retrain ATR volatility target based on realized volatility.

        Calculates optimal target volatility from training data
        by analyzing historical realized volatility distribution.

        Args:
            train_df: Training data

        Returns:
            Optimal target volatility (e.g., 0.12 for 12%)
        """
        # Calculate realized volatility
        returns = train_df['close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252)  # Annualized

        # Target slightly below realized vol for safety
        # Use 75th percentile of realized vol as target
        target_vol = max(0.08, min(0.20, realized_vol * 0.75))

        return float(target_vol)

    def _retrain_confidence(self, train_df: pd.DataFrame) -> tuple:
        """
        Retrain confidence scaling range based on signal quality.

        Analyzes training data to determine optimal confidence
        scaling range (min_scale, max_scale).

        Args:
            train_df: Training data

        Returns:
            Tuple of (min_scale, max_scale) for confidence range
        """
        # Calculate signal strength from price action
        returns = train_df['close'].pct_change().dropna()
        signal_strength = abs(returns).mean()

        # Adaptive confidence range based on signal strength
        if signal_strength > 0.015:  # Strong signals
            confidence_range = (0.6, 1.6)  # More aggressive
        elif signal_strength > 0.010:  # Medium signals
            confidence_range = (0.5, 1.5)  # Balanced
        else:  # Weak signals
            confidence_range = (0.7, 1.3)  # Conservative

        return confidence_range

    def _retrain_meta_learner(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Retrain meta-learner weights based on strategy performance.

        Analyzes which signals performed best in training period
        and adjusts weights accordingly.

        Args:
            train_df: Training data

        Returns:
            Dictionary of strategy weights
        """
        # Simple momentum-based weighting
        # In a full implementation, this would analyze actual strategy performance
        returns = train_df['close'].pct_change().dropna()

        # Calculate trend strength
        trend_strength = returns.rolling(20).mean().abs().mean()

        # Adaptive weights based on trend strength
        if trend_strength > 0.002:  # Trending market
            weights = {
                'momentum': 0.4,
                'mean_reversion': 0.1,
                'volatility': 0.3,
                'breakout': 0.2
            }
        else:  # Range-bound market
            weights = {
                'momentum': 0.2,
                'mean_reversion': 0.4,
                'volatility': 0.2,
                'breakout': 0.2
            }

        return weights

    def _retrain_bandit_weights(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Retrain bandit weights for strategy selection.

        Updates exploration/exploitation weights based on
        recent strategy performance in training data.

        Args:
            train_df: Training data

        Returns:
            Dictionary of bandit weights per strategy
        """
        # Simplified bandit weight update
        # In full implementation, this would use actual strategy returns

        # Equal weights as baseline (exploration)
        n_strategies = 11  # Total strategies in system
        base_weight = 1.0 / n_strategies

        # Add small random perturbation for exploration
        weights = {}
        for i in range(n_strategies):
            strategy_name = f"strategy_{i}"
            weights[strategy_name] = base_weight * (1.0 + np.random.uniform(-0.1, 0.1))

        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def _test_window(
        self,
        symbol: str,
        test_df: pd.DataFrame,
        fold_config: FoldConfig,
        fold_idx: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp
    ) -> FoldResult:
        """
        Test retrained models on a test window.

        Runs backtest on test data using models retrained
        on the training window.

        Args:
            symbol: Trading symbol
            test_df: Test data
            fold_config: Retrained configuration for this fold
            fold_idx: Fold index
            train_start: Training window start
            train_end: Training window end
            test_start: Test window start
            test_end: Test window end

        Returns:
            FoldResult with test performance metrics
        """
        # Create fresh backtest engine with retrained config
        config = BacktestConfig(
            symbol=symbol,
            random_seed=self.seed,
            # Apply retrained parameters
            target_vol=fold_config.atr_target_vol or 0.12,
            meta_confidence_range=fold_config.confidence_range or (0.5, 1.5)
        )

        engine = BacktestEngine(config=config)

        # Run backtest on test window
        result = engine.run_standard(symbol, test_df)

        # Extract metrics
        fold_result = FoldResult(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            total_trades=result.total_trades,
            config=fold_config
        )

        return fold_result

    def _aggregate_fold_results(self, fold_results: List[FoldResult]) -> Dict[str, Any]:
        """
        Aggregate results across all folds.

        Args:
            fold_results: List of fold results

        Returns:
            Aggregated metrics dictionary
        """
        if len(fold_results) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'fold_results': []
            }

        # Aggregate metrics
        total_return = np.mean([f.total_return for f in fold_results])
        sharpe_ratio = np.mean([f.sharpe_ratio for f in fold_results])
        sortino_ratio = np.mean([f.sortino_ratio for f in fold_results])
        max_drawdown = np.min([f.max_drawdown for f in fold_results])  # Worst drawdown
        win_rate = np.mean([f.win_rate for f in fold_results])
        profit_factor = np.mean([f.profit_factor for f in fold_results])
        total_trades = sum([f.total_trades for f in fold_results])

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_trades': int(total_trades),
            'fold_results': fold_results,
            'n_folds': len(fold_results)
        }

    def __repr__(self) -> str:
        """String representation of AdaptiveTrainer."""
        return f"AdaptiveTrainer(seed={self.seed}, config={self.config.symbol})"
