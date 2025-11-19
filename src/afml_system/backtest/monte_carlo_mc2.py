"""
PRADO9_EVO Module MC2 — Monte Carlo Robustness Engine

Enhanced Monte Carlo testing with:
1. Block Bootstrapped Monte Carlo Simulator
2. Turbulence Stress Tests (extreme volatility scenarios)
3. Signal Corruption Tests (degraded signal quality)

This module goes beyond simple trade shuffling to test strategy robustness
under adverse market conditions and signal degradation.

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CONSTANTS
# ============================================================================

MC2_VERSION = '1.0.0'
EPSILON = 1e-12


# ============================================================================
# ENUMS
# ============================================================================

class TurbulenceLevel(Enum):
    """Market turbulence levels for stress testing."""
    MILD = "MILD"           # 1.5x volatility
    MODERATE = "MODERATE"   # 2.0x volatility
    SEVERE = "SEVERE"       # 3.0x volatility
    EXTREME = "EXTREME"     # 5.0x volatility (2008-style crash)


class CorruptionType(Enum):
    """Signal corruption types."""
    NOISE = "NOISE"                     # Add Gaussian noise to signals
    BIAS = "BIAS"                       # Systematic directional bias
    LAG = "LAG"                         # Introduce signal lag
    MISSING = "MISSING"                 # Random missing signals
    REVERSE = "REVERSE"                 # Randomly reverse signal direction


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BlockBootstrapConfig:
    """Configuration for block bootstrapping."""
    block_size: int = 20           # Size of blocks (trading days)
    overlap: bool = True           # Allow overlapping blocks
    preserve_volatility: bool = True  # Match original vol distribution
    seed: int = 42


@dataclass
class TurbulenceTestConfig:
    """Configuration for turbulence stress testing."""
    level: TurbulenceLevel = TurbulenceLevel.MODERATE
    apply_to_returns: bool = True  # Scale returns
    apply_to_signals: bool = True  # Add noise to signals
    preserve_mean: bool = True     # Keep mean return unchanged


@dataclass
class CorruptionTestConfig:
    """Configuration for signal corruption testing."""
    corruption_type: CorruptionType = CorruptionType.NOISE
    corruption_rate: float = 0.2   # Fraction of signals to corrupt (20%)
    noise_std: float = 0.5         # Std dev of noise (for NOISE type)
    lag_periods: int = 1           # Lag periods (for LAG type)


@dataclass
class MC2Result:
    """Results from MC2 simulation."""
    simulation_type: str           # "block_bootstrap", "turbulence", "corruption"
    n_simulations: int
    actual_sharpe: float
    mc_sharpe_mean: float
    mc_sharpe_std: float
    mc_sharpe_min: float
    mc_sharpe_max: float
    mc_sharpe_median: float
    skill_percentile: float        # Percentile of actual vs MC distribution
    p_value: float                 # Statistical significance
    significant: bool              # p < 0.05
    config: Any                    # Config used for this test
    distribution: Dict[str, float] # Full distribution stats


# ============================================================================
# BLOCK BOOTSTRAPPED MONTE CARLO SIMULATOR
# ============================================================================

class BlockBootstrappedMCSimulator:
    """
    Block bootstrapping Monte Carlo simulator.

    Instead of shuffling individual trades (which destroys autocorrelation),
    this simulator resamples blocks of returns to preserve serial correlation
    while still testing statistical significance.

    This is critical for momentum/mean-reversion strategies where
    autocorrelation is a key feature.

    Example:
        >>> simulator = BlockBootstrappedMCSimulator(seed=42)
        >>> result = simulator.run(returns_series, block_size=20, n_sim=10000)
        >>> print(f"Skill percentile: {result.skill_percentile:.1f}%")
    """

    def __init__(self, seed: int = 42):
        """
        Initialize Block Bootstrapped MC Simulator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def run(
        self,
        returns: pd.Series,
        block_size: int = 20,
        n_sim: int = 10000,
        preserve_volatility: bool = True,
        overlap: bool = True
    ) -> MC2Result:
        """
        Run block bootstrapped Monte Carlo simulation.

        Args:
            returns: Time series of returns
            block_size: Size of blocks to resample (default: 20 days)
            n_sim: Number of simulations (default: 10000)
            preserve_volatility: Match original volatility (default: True)
            overlap: Allow overlapping blocks (default: True)

        Returns:
            MC2Result with simulation statistics
        """
        # Reset seed for reproducibility
        np.random.seed(self.seed)

        if len(returns) < block_size:
            raise ValueError(f"Insufficient data: {len(returns)} < block_size {block_size}")

        # Compute actual Sharpe
        actual_sharpe = self._compute_sharpe(returns)

        # Generate blocks
        blocks = self._create_blocks(returns, block_size, overlap)

        # Run simulations
        mc_sharpes = []

        for _ in range(n_sim):
            # Resample blocks
            bootstrap_returns = self._resample_blocks(blocks, len(returns))

            # Preserve volatility if requested
            if preserve_volatility:
                bootstrap_returns = self._match_volatility(
                    bootstrap_returns, returns
                )

            # Compute Sharpe for bootstrapped series
            mc_sharpe = self._compute_sharpe(bootstrap_returns)
            mc_sharpes.append(mc_sharpe)

        mc_sharpes = np.array(mc_sharpes)

        # Clamp to [-10, 10] to avoid outliers
        mc_sharpes = np.clip(mc_sharpes, -10.0, 10.0)

        # Compute statistics
        mc_mean = float(np.mean(mc_sharpes))
        mc_std = float(np.std(mc_sharpes))
        mc_min = float(np.min(mc_sharpes))
        mc_max = float(np.max(mc_sharpes))
        mc_median = float(np.median(mc_sharpes))

        # Skill percentile
        better_count = np.sum(actual_sharpe >= mc_sharpes)
        skill_percentile = float(better_count / n_sim * 100.0)

        # P-value (two-tailed)
        worse_count = np.sum(actual_sharpe <= mc_sharpes)
        p_value = float(min(better_count, worse_count) / n_sim * 2.0)
        p_value = min(p_value, 1.0)

        # Significance
        significant = p_value < 0.05

        # Distribution stats
        distribution = {
            'q05': float(np.percentile(mc_sharpes, 5)),
            'q25': float(np.percentile(mc_sharpes, 25)),
            'q50': mc_median,
            'q75': float(np.percentile(mc_sharpes, 75)),
            'q95': float(np.percentile(mc_sharpes, 95))
        }

        config = BlockBootstrapConfig(
            block_size=block_size,
            overlap=overlap,
            preserve_volatility=preserve_volatility,
            seed=self.seed
        )

        return MC2Result(
            simulation_type="block_bootstrap",
            n_simulations=n_sim,
            actual_sharpe=float(actual_sharpe),
            mc_sharpe_mean=mc_mean,
            mc_sharpe_std=mc_std,
            mc_sharpe_min=mc_min,
            mc_sharpe_max=mc_max,
            mc_sharpe_median=mc_median,
            skill_percentile=skill_percentile,
            p_value=p_value,
            significant=significant,
            config=config,
            distribution=distribution
        )

    def _create_blocks(
        self,
        returns: pd.Series,
        block_size: int,
        overlap: bool
    ) -> List[np.ndarray]:
        """
        Create blocks from returns series.

        Args:
            returns: Returns series
            block_size: Size of each block
            overlap: Allow overlapping blocks

        Returns:
            List of blocks (numpy arrays)
        """
        blocks = []
        step = 1 if overlap else block_size

        for i in range(0, len(returns) - block_size + 1, step):
            block = returns.iloc[i:i+block_size].values
            blocks.append(block)

        return blocks

    def _resample_blocks(
        self,
        blocks: List[np.ndarray],
        target_length: int
    ) -> np.ndarray:
        """
        Resample blocks to create bootstrap series.

        Args:
            blocks: List of blocks
            target_length: Target length of resampled series

        Returns:
            Bootstrap returns series
        """
        bootstrap_returns = []
        current_length = 0

        while current_length < target_length:
            # Randomly select a block
            block_idx = np.random.randint(0, len(blocks))
            block = blocks[block_idx]

            # Add block to bootstrap series
            remaining = target_length - current_length
            if len(block) <= remaining:
                bootstrap_returns.extend(block)
                current_length += len(block)
            else:
                # Partial block to match target length
                bootstrap_returns.extend(block[:remaining])
                current_length = target_length

        return np.array(bootstrap_returns)

    def _match_volatility(
        self,
        bootstrap_returns: np.ndarray,
        original_returns: pd.Series
    ) -> np.ndarray:
        """
        Scale bootstrap returns to match original volatility.

        Args:
            bootstrap_returns: Bootstrap returns
            original_returns: Original returns series

        Returns:
            Scaled bootstrap returns
        """
        original_std = original_returns.std()
        bootstrap_std = bootstrap_returns.std()

        if bootstrap_std < EPSILON:
            return bootstrap_returns

        # Scale to match volatility
        scaled_returns = bootstrap_returns * (original_std / bootstrap_std)

        return scaled_returns

    def _compute_sharpe(self, returns: np.ndarray) -> float:
        """
        Compute annualized Sharpe ratio.

        Args:
            returns: Returns series

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return < EPSILON:
            return 0.0

        sharpe = np.sqrt(252) * mean_return / std_return
        return float(sharpe)


# ============================================================================
# TURBULENCE STRESS TESTS
# ============================================================================

class TurbulenceStressTester:
    """
    Turbulence stress testing engine.

    Tests strategy performance under extreme market volatility by scaling
    returns and adding noise to simulate crash conditions.

    Example:
        >>> tester = TurbulenceStressTester(seed=42)
        >>> result = tester.run(
        ...     df, backtest_engine,
        ...     level=TurbulenceLevel.SEVERE
        ... )
    """

    def __init__(self, seed: int = 42):
        """
        Initialize Turbulence Stress Tester.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def run(
        self,
        df: pd.DataFrame,
        backtest_func: callable,
        level: TurbulenceLevel = TurbulenceLevel.MODERATE,
        n_sim: int = 1000,
        apply_to_returns: bool = True,
        preserve_mean: bool = True
    ) -> MC2Result:
        """
        Run turbulence stress test.

        Args:
            df: Original OHLCV DataFrame
            backtest_func: Function to run backtest (takes df, returns Sharpe)
            level: Turbulence level
            n_sim: Number of simulations
            apply_to_returns: Scale returns volatility
            preserve_mean: Preserve mean return

        Returns:
            MC2Result with stress test statistics
        """
        # Reset seed
        np.random.seed(self.seed)

        # Get volatility multiplier
        vol_multipliers = {
            TurbulenceLevel.MILD: 1.5,
            TurbulenceLevel.MODERATE: 2.0,
            TurbulenceLevel.SEVERE: 3.0,
            TurbulenceLevel.EXTREME: 5.0
        }
        vol_mult = vol_multipliers[level]

        # Run baseline backtest
        actual_sharpe = backtest_func(df)

        # Run turbulence simulations
        mc_sharpes = []

        for _ in range(n_sim):
            # Create turbulent version of data
            turbulent_df = self._apply_turbulence(
                df, vol_mult, apply_to_returns, preserve_mean
            )

            # Run backtest on turbulent data
            mc_sharpe = backtest_func(turbulent_df)
            mc_sharpes.append(mc_sharpe)

        mc_sharpes = np.array(mc_sharpes)
        mc_sharpes = np.clip(mc_sharpes, -10.0, 10.0)

        # Compute statistics
        mc_mean = float(np.mean(mc_sharpes))
        mc_std = float(np.std(mc_sharpes))
        mc_min = float(np.min(mc_sharpes))
        mc_max = float(np.max(mc_sharpes))
        mc_median = float(np.median(mc_sharpes))

        # Skill percentile (how often actual beats turbulent)
        better_count = np.sum(actual_sharpe >= mc_sharpes)
        skill_percentile = float(better_count / n_sim * 100.0)

        # P-value
        worse_count = np.sum(actual_sharpe <= mc_sharpes)
        p_value = float(min(better_count, worse_count) / n_sim * 2.0)
        p_value = min(p_value, 1.0)

        # Significance
        significant = p_value < 0.05

        # Distribution
        distribution = {
            'q05': float(np.percentile(mc_sharpes, 5)),
            'q25': float(np.percentile(mc_sharpes, 25)),
            'q50': mc_median,
            'q75': float(np.percentile(mc_sharpes, 75)),
            'q95': float(np.percentile(mc_sharpes, 95))
        }

        config = TurbulenceTestConfig(
            level=level,
            apply_to_returns=apply_to_returns,
            preserve_mean=preserve_mean
        )

        return MC2Result(
            simulation_type=f"turbulence_{level.value.lower()}",
            n_simulations=n_sim,
            actual_sharpe=float(actual_sharpe),
            mc_sharpe_mean=mc_mean,
            mc_sharpe_std=mc_std,
            mc_sharpe_min=mc_min,
            mc_sharpe_max=mc_max,
            mc_sharpe_median=mc_median,
            skill_percentile=skill_percentile,
            p_value=p_value,
            significant=significant,
            config=config,
            distribution=distribution
        )

    def _apply_turbulence(
        self,
        df: pd.DataFrame,
        vol_mult: float,
        apply_to_returns: bool,
        preserve_mean: bool
    ) -> pd.DataFrame:
        """
        Apply turbulence to OHLCV data.

        Args:
            df: Original DataFrame
            vol_mult: Volatility multiplier
            apply_to_returns: Scale returns
            preserve_mean: Preserve mean return

        Returns:
            Turbulent DataFrame
        """
        df_turb = df.copy()

        if apply_to_returns:
            # Compute returns
            returns = df_turb['close'].pct_change().fillna(0)

            if preserve_mean:
                # Zero-mean returns
                returns_centered = returns - returns.mean()

                # Scale volatility
                returns_scaled = returns_centered * vol_mult

                # Add back mean
                returns_turbulent = returns_scaled + returns.mean()
            else:
                # Just scale returns
                returns_turbulent = returns * vol_mult

            # Reconstruct prices
            price_path = (1 + returns_turbulent).cumprod()
            df_turb['close'] = df['close'].iloc[0] * price_path

            # Scale OHLV proportionally
            scale_factor = df_turb['close'] / df['close']
            df_turb['open'] = df['open'] * scale_factor
            df_turb['high'] = df['high'] * scale_factor
            df_turb['low'] = df['low'] * scale_factor

        return df_turb


# ============================================================================
# SIGNAL CORRUPTION TESTS
# ============================================================================

class SignalCorruptionTester:
    """
    Signal corruption testing engine.

    Tests strategy robustness under degraded signal quality by adding noise,
    lag, missing values, or reversing signals.

    Example:
        >>> tester = SignalCorruptionTester(seed=42)
        >>> result = tester.run(
        ...     signals, backtest_func,
        ...     corruption_type=CorruptionType.NOISE,
        ...     corruption_rate=0.3
        ... )
    """

    def __init__(self, seed: int = 42):
        """
        Initialize Signal Corruption Tester.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def run(
        self,
        df: pd.DataFrame,
        backtest_func: callable,
        corruption_type: CorruptionType = CorruptionType.NOISE,
        corruption_rate: float = 0.2,
        n_sim: int = 1000,
        **kwargs
    ) -> MC2Result:
        """
        Run signal corruption test.

        Args:
            df: Original OHLCV DataFrame (signals computed internally)
            backtest_func: Function to run backtest with corrupted signals
            corruption_type: Type of corruption to apply
            corruption_rate: Fraction of signals to corrupt (0.0 - 1.0)
            n_sim: Number of simulations
            **kwargs: Additional corruption parameters (noise_std, lag_periods)

        Returns:
            MC2Result with corruption test statistics
        """
        # Reset seed
        np.random.seed(self.seed)

        # Run baseline
        actual_sharpe = backtest_func(df, corruption_rate=0.0)

        # Run corruption simulations
        mc_sharpes = []

        for _ in range(n_sim):
            # Apply corruption via backtest_func
            # (backtest_func should handle corruption internally)
            mc_sharpe = backtest_func(
                df,
                corruption_rate=corruption_rate,
                corruption_type=corruption_type,
                **kwargs
            )
            mc_sharpes.append(mc_sharpe)

        mc_sharpes = np.array(mc_sharpes)
        mc_sharpes = np.clip(mc_sharpes, -10.0, 10.0)

        # Compute statistics
        mc_mean = float(np.mean(mc_sharpes))
        mc_std = float(np.std(mc_sharpes))
        mc_min = float(np.min(mc_sharpes))
        mc_max = float(np.max(mc_sharpes))
        mc_median = float(np.median(mc_sharpes))

        # Skill percentile
        better_count = np.sum(actual_sharpe >= mc_sharpes)
        skill_percentile = float(better_count / n_sim * 100.0)

        # P-value
        worse_count = np.sum(actual_sharpe <= mc_sharpes)
        p_value = float(min(better_count, worse_count) / n_sim * 2.0)
        p_value = min(p_value, 1.0)

        # Significance
        significant = p_value < 0.05

        # Distribution
        distribution = {
            'q05': float(np.percentile(mc_sharpes, 5)),
            'q25': float(np.percentile(mc_sharpes, 25)),
            'q50': mc_median,
            'q75': float(np.percentile(mc_sharpes, 75)),
            'q95': float(np.percentile(mc_sharpes, 95))
        }

        config = CorruptionTestConfig(
            corruption_type=corruption_type,
            corruption_rate=corruption_rate,
            noise_std=kwargs.get('noise_std', 0.5),
            lag_periods=kwargs.get('lag_periods', 1)
        )

        return MC2Result(
            simulation_type=f"corruption_{corruption_type.value.lower()}",
            n_simulations=n_sim,
            actual_sharpe=float(actual_sharpe),
            mc_sharpe_mean=mc_mean,
            mc_sharpe_std=mc_std,
            mc_sharpe_min=mc_min,
            mc_sharpe_max=mc_max,
            mc_sharpe_median=mc_median,
            skill_percentile=skill_percentile,
            p_value=p_value,
            significant=significant,
            config=config,
            distribution=distribution
        )


# ============================================================================
# UNIFIED MC2 ENGINE
# ============================================================================

class MC2Engine:
    """
    Unified Monte Carlo Robustness Engine.

    Combines block bootstrapping, turbulence stress tests, and signal
    corruption tests into a single comprehensive robustness assessment.

    Example:
        >>> engine = MC2Engine(seed=42)
        >>> results = engine.run_comprehensive(
        ...     df, backtest_func,
        ...     n_sim=10000
        ... )
        >>> for test_name, result in results.items():
        ...     print(f"{test_name}: {result.skill_percentile:.1f}%")
    """

    def __init__(self, seed: int = 42):
        """
        Initialize MC2 Engine.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.block_bootstrap = BlockBootstrappedMCSimulator(seed=seed)
        self.turbulence = TurbulenceStressTester(seed=seed)
        self.corruption = SignalCorruptionTester(seed=seed)

    def run_comprehensive(
        self,
        df: pd.DataFrame,
        backtest_func: callable,
        n_sim: int = 1000
    ) -> Dict[str, MC2Result]:
        """
        Run comprehensive robustness assessment.

        Args:
            df: OHLCV DataFrame
            backtest_func: Backtest function
            n_sim: Number of simulations per test

        Returns:
            Dictionary of test results
        """
        results = {}

        # 1. Block Bootstrap Test
        returns = df['close'].pct_change().dropna()
        results['block_bootstrap'] = self.block_bootstrap.run(
            returns=returns,
            block_size=20,
            n_sim=n_sim
        )

        # 2. Turbulence Tests (multiple levels)
        for level in [TurbulenceLevel.MODERATE, TurbulenceLevel.SEVERE]:
            test_name = f"turbulence_{level.value.lower()}"
            results[test_name] = self.turbulence.run(
                df=df,
                backtest_func=backtest_func,
                level=level,
                n_sim=n_sim
            )

        # Note: Signal corruption requires backtest_func to support corruption
        # This is optional and should be implemented in BacktestEngine

        return results

    def __repr__(self) -> str:
        """String representation."""
        return f"MC2Engine(seed={self.seed}, version={MC2_VERSION})"
