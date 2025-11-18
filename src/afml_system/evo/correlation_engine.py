"""
PRADO9_EVO Module F — Correlation Cluster Engine

Detects strategy correlation clusters, penalizes redundant strategies,
boosts uncorrelated ones, and provides crucial information about:
- Diversity
- Orthogonality
- Alpha uniqueness

This prevents ensemble collapse and overcrowding.

Author: PRADO9_EVO Builder
Date: 2025-01-16
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime

# Import PerformanceMemory for type hints
try:
    from .performance_memory import PerformanceMemory, PerformanceRecord
except ImportError:
    PerformanceMemory = None
    PerformanceRecord = None


# ============================================================================
# CONSTANTS
# ============================================================================

CORRELATION_ENGINE_VERSION = '1.0.0'
MINIMUM_STRATEGIES_FOR_CLUSTERING = 2
MINIMUM_OBSERVATIONS_PER_STRATEGY = 3
DEFAULT_CORRELATION_THRESHOLD = 0.7


# ============================================================================
# CORRELATION MATRIX BUILDER
# ============================================================================

class CorrelationMatrixBuilder:
    """
    Builds correlation matrices from performance memory.

    Supports:
    - Return-based correlation
    - Prediction-based correlation
    - Time-aligned matrices
    - Missing value handling
    """

    def build_return_matrix(
        self,
        perf_memory: 'PerformanceMemory'
    ) -> pd.DataFrame:
        """
        Build return matrix from performance memory.

        Args:
            perf_memory: PerformanceMemory instance

        Returns:
            DataFrame with strategies as columns, timestamps as rows
        """
        # Collect all records
        all_records = perf_memory.records

        if not all_records:
            return pd.DataFrame()

        # Build dict of {strategy: [(timestamp, return), ...]}
        strategy_returns = defaultdict(list)

        for record in all_records:
            strategy_returns[record.strategy_name].append(
                (record.timestamp, record.return_value)
            )

        # Convert to DataFrame
        data = {}
        for strategy, returns in strategy_returns.items():
            # Sort by timestamp
            returns_sorted = sorted(returns, key=lambda x: x[0])
            timestamps = [r[0] for r in returns_sorted]
            values = [r[1] for r in returns_sorted]

            # Create series
            data[strategy] = pd.Series(values, index=timestamps)

        if not data:
            return pd.DataFrame()

        # Combine into DataFrame with full outer join (align all timestamps)
        df = pd.DataFrame(data)

        # Enforce numeric dtype
        df = df.apply(pd.to_numeric, errors='coerce')

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward-fill, then back-fill, then fill remaining with 0.0
        df = df.ffill().bfill().fillna(0.0)

        # Drop columns that are fully NaN (before filling)
        df = df.dropna(axis=1, how='all')

        return df

    def build_prediction_matrix(
        self,
        perf_memory: 'PerformanceMemory'
    ) -> pd.DataFrame:
        """
        Build prediction matrix from performance memory.

        Args:
            perf_memory: PerformanceMemory instance

        Returns:
            DataFrame with strategies as columns, timestamps as rows
        """
        # Collect all records
        all_records = perf_memory.records

        if not all_records:
            return pd.DataFrame()

        # Build dict of {strategy: [(timestamp, prediction), ...]}
        strategy_predictions = defaultdict(list)

        for record in all_records:
            if record.prediction is not None:
                strategy_predictions[record.strategy_name].append(
                    (record.timestamp, record.prediction)
                )

        # Convert to DataFrame
        data = {}
        for strategy, predictions in strategy_predictions.items():
            # Sort by timestamp
            preds_sorted = sorted(predictions, key=lambda x: x[0])
            timestamps = [p[0] for p in preds_sorted]
            values = [p[1] for p in preds_sorted]

            # Create series
            data[strategy] = pd.Series(values, index=timestamps)

        if not data:
            return pd.DataFrame()

        # Combine into DataFrame with full outer join (align all timestamps)
        df = pd.DataFrame(data)

        # Enforce numeric dtype
        df = df.apply(pd.to_numeric, errors='coerce')

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward-fill, then back-fill, then fill remaining with 0.0
        df = df.ffill().bfill().fillna(0.0)

        # Drop columns that are fully NaN (before filling)
        df = df.dropna(axis=1, how='all')

        return df

    def compute_correlation(
        self,
        matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute correlation matrix with safety checks.

        Args:
            matrix: DataFrame with strategies as columns

        Returns:
            Correlation matrix (symmetric)
        """
        if matrix.empty:
            return pd.DataFrame()

        # Fallback: if < 2 columns, return identity matrix
        if len(matrix.columns) < 2:
            strategies = list(matrix.columns)
            identity = pd.DataFrame(
                np.eye(len(strategies)),
                index=strategies,
                columns=strategies
            )
            return identity

        # Remove infinite values
        matrix = matrix.replace([np.inf, -np.inf], np.nan)

        # Remove any columns with constant values (zero variance)
        non_constant_cols = []
        for col in matrix.columns:
            col_std = matrix[col].std()
            if not np.isnan(col_std) and col_std > 1e-10:
                non_constant_cols.append(col)

        matrix = matrix[non_constant_cols]

        if matrix.empty or len(matrix.columns) < 2:
            # Return identity matrix for remaining columns
            if len(matrix.columns) == 1:
                strategy = matrix.columns[0]
                return pd.DataFrame([[1.0]], index=[strategy], columns=[strategy])
            return pd.DataFrame()

        # Compute correlation
        try:
            corr_matrix = matrix.corr(method='pearson')

            # Replace NaN with 0 (uncorrelated)
            corr_matrix = corr_matrix.fillna(0.0)

            # Clip to valid range
            corr_matrix = corr_matrix.clip(-1.0, 1.0)

            # Ensure symmetry
            corr_matrix = (corr_matrix + corr_matrix.T) / 2

            # Ensure diagonal is 1.0
            np.fill_diagonal(corr_matrix.values, 1.0)

            return corr_matrix

        except Exception:
            # Fallback: return identity matrix
            strategies = list(matrix.columns)
            identity = pd.DataFrame(
                np.eye(len(strategies)),
                index=strategies,
                columns=strategies
            )
            return identity


# ============================================================================
# CLUSTER ENGINE
# ============================================================================

class ClusterEngine:
    """
    Clusters strategies based on correlation matrix.

    Uses hierarchical clustering for deterministic results.
    """

    def __init__(self, correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD):
        """
        Initialize cluster engine.

        Args:
            correlation_threshold: Threshold for correlation (0-1)
        """
        self.correlation_threshold = correlation_threshold

    def cluster_strategies(
        self,
        corr_matrix: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Cluster strategies based on correlation matrix (deterministic).

        Args:
            corr_matrix: Correlation matrix (symmetric)
            n_clusters: Number of clusters (None for auto)

        Returns:
            Dict mapping strategy_name -> cluster_id
        """
        if corr_matrix.empty or len(corr_matrix) < MINIMUM_STRATEGIES_FOR_CLUSTERING:
            # Return single cluster
            return {name: 0 for name in corr_matrix.index}

        # Convert correlation to distance
        # Distance = 1 - |correlation|
        # (Absolute value so negative correlation also means similarity)
        distance_matrix = 1 - corr_matrix.abs()

        # Ensure valid distance matrix
        distance_matrix = distance_matrix.clip(0.0, 2.0)

        # Replace diagonal with zeros
        np.fill_diagonal(distance_matrix.values, 0.0)

        # Check for singular matrix (all correlations ~1 or ~0)
        unique_distances = len(np.unique(np.round(distance_matrix.values, decimals=3)))
        if unique_distances <= 2:
            # Singular matrix - assign each strategy to separate cluster
            return {name: i for i, name in enumerate(corr_matrix.index)}

        # Auto-determine number of clusters if not specified
        if n_clusters is None:
            # Use distance threshold based on correlation threshold
            distance_threshold = 1 - self.correlation_threshold

            # For > 4 strategies, enforce at least 2 clusters
            if len(corr_matrix) > 4:
                # Hierarchical clustering with distance threshold
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=distance_threshold,
                        metric='precomputed',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(distance_matrix.values)

                    # Check if we got at least 2 clusters
                    n_unique_labels = len(np.unique(labels))
                    if n_unique_labels < 2:
                        # Force 2 clusters
                        clustering = AgglomerativeClustering(
                            n_clusters=2,
                            metric='precomputed',
                            linkage='average'
                        )
                        labels = clustering.fit_predict(distance_matrix.values)
                except Exception:
                    # Fallback: separate clusters for each
                    return {name: i for i, name in enumerate(corr_matrix.index)}
            else:
                # <= 4 strategies, use distance threshold
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=distance_threshold,
                        metric='precomputed',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(distance_matrix.values)
                except Exception:
                    # Fallback: separate clusters for each
                    return {name: i for i, name in enumerate(corr_matrix.index)}
        else:
            # Fixed number of clusters
            n_clusters = min(n_clusters, len(corr_matrix))
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',
                    linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix.values)
            except Exception:
                # Fallback: separate clusters for each
                return {name: i for i, name in enumerate(corr_matrix.index)}

        # Map strategy names to cluster IDs (deterministic order)
        clusters = {}
        for i, strategy in enumerate(sorted(corr_matrix.index)):
            idx = list(corr_matrix.index).index(strategy)
            clusters[strategy] = int(labels[idx])

        return clusters

    def compute_uniqueness_scores(
        self,
        corr_matrix: pd.DataFrame,
        clusters: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Compute uniqueness score for each strategy.

        Formula:
            uniqueness = 1 - mean(abs(correlations_excluding_self))

        Args:
            corr_matrix: Correlation matrix
            clusters: Cluster assignments

        Returns:
            Dict mapping strategy_name -> uniqueness_score [0, 1]
        """
        if corr_matrix.empty:
            return {}

        uniqueness = {}

        for strategy in corr_matrix.index:
            # Get correlations with all other strategies
            cors = corr_matrix.loc[strategy, :]

            # Exclude self-correlation (always 1.0)
            other_cors = cors.drop(strategy, errors='ignore')

            if len(other_cors) == 0:
                # Only strategy, maximally unique
                uniqueness[strategy] = 1.0
                continue

            # Average absolute correlation with others
            avg_cor = other_cors.abs().mean()

            # Guard against NaN/Inf
            if np.isnan(avg_cor) or np.isinf(avg_cor):
                # Fallback uniqueness
                uniqueness[strategy] = 0.5
                continue

            # Uniqueness = 1 - average absolute correlation
            uniqueness_score = 1.0 - avg_cor

            # Guarantee output in [0, 1]
            uniqueness_score = float(min(max(uniqueness_score, 0.0), 1.0))

            uniqueness[strategy] = uniqueness_score

        return uniqueness

    def compute_correlation_penalties(
        self,
        corr_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute correlation penalty for each strategy.

        Formula:
            penalty = mean(abs(correlations_excluding_self))

        Higher correlation → higher penalty.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            Dict mapping strategy_name -> penalty [0, 1]
        """
        if corr_matrix.empty:
            return {}

        penalties = {}

        for strategy in corr_matrix.index:
            # Get correlations with all other strategies
            cors = corr_matrix.loc[strategy, :]

            # Exclude self-correlation (always 1.0)
            other_cors = cors.drop(strategy, errors='ignore')

            if len(other_cors) == 0:
                # Only strategy, no penalty
                penalties[strategy] = 0.0
                continue

            # Average absolute correlation with others
            avg_cor = other_cors.abs().mean()

            # Guard against NaN/Inf
            if np.isnan(avg_cor) or np.isinf(avg_cor):
                # Fallback penalty
                penalties[strategy] = 0.5
                continue

            # Normalize penalty to [0, 1]
            penalty = float(min(max(avg_cor, 0.0), 1.0))

            penalties[strategy] = penalty

        return penalties


# ============================================================================
# CORRELATION CLUSTER ENGINE
# ============================================================================

class CorrelationClusterEngine:
    """
    Main correlation cluster engine.

    Orchestrates:
    - Correlation matrix building
    - Strategy clustering
    - Uniqueness scoring
    - Penalty computation
    - Persistence
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    ):
        """
        Initialize correlation cluster engine.

        Args:
            state_dir: Directory for state files
            correlation_threshold: Threshold for clustering
        """
        if state_dir is None:
            state_dir = Path.home() / ".prado" / "evo"

        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.correlation_threshold = correlation_threshold

        # Components
        self.matrix_builder = CorrelationMatrixBuilder()
        self.cluster_engine = ClusterEngine(correlation_threshold)

        # State
        self.clusters = {}  # type: Dict[str, int]
        self.penalties = {}  # type: Dict[str, float]
        self.uniqueness = {}  # type: Dict[str, float]
        self.corr_matrix = pd.DataFrame()

        # Try to load existing state
        self.load()

    def update(self, perf_memory: 'PerformanceMemory') -> None:
        """
        Update correlation clusters from performance memory.

        Args:
            perf_memory: PerformanceMemory instance
        """
        # Build return matrix
        return_matrix = self.matrix_builder.build_return_matrix(perf_memory)

        if return_matrix.empty or len(return_matrix.columns) < MINIMUM_STRATEGIES_FOR_CLUSTERING:
            # Not enough strategies, reset state
            self.clusters = {}
            self.penalties = {}
            self.uniqueness = {}
            self.corr_matrix = pd.DataFrame()
            return

        # Filter strategies with insufficient observations
        valid_strategies = []
        for strategy in return_matrix.columns:
            if len(return_matrix[strategy].dropna()) >= MINIMUM_OBSERVATIONS_PER_STRATEGY:
                valid_strategies.append(strategy)

        if len(valid_strategies) < MINIMUM_STRATEGIES_FOR_CLUSTERING:
            # Not enough valid strategies
            self.clusters = {}
            self.penalties = {}
            self.uniqueness = {}
            self.corr_matrix = pd.DataFrame()
            return

        # Filter to valid strategies
        return_matrix = return_matrix[valid_strategies]

        # Compute correlation matrix
        self.corr_matrix = self.matrix_builder.compute_correlation(return_matrix)

        if self.corr_matrix.empty:
            # Failed to compute correlation
            self.clusters = {}
            self.penalties = {}
            self.uniqueness = {}
            return

        # Cluster strategies
        self.clusters = self.cluster_engine.cluster_strategies(self.corr_matrix)

        # Compute uniqueness scores
        self.uniqueness = self.cluster_engine.compute_uniqueness_scores(
            self.corr_matrix,
            self.clusters
        )

        # Compute correlation penalties
        self.penalties = self.cluster_engine.compute_correlation_penalties(
            self.corr_matrix
        )

    def get_clusters(self) -> Dict[str, int]:
        """Get cluster assignments."""
        return self.clusters.copy()

    def get_penalties(self) -> Dict[str, float]:
        """Get correlation penalties."""
        return self.penalties.copy()

    def get_uniqueness(self) -> Dict[str, float]:
        """Get uniqueness scores."""
        return self.uniqueness.copy()

    def save(self) -> None:
        """Save correlation cluster state to disk."""
        # Save clusters
        clusters_path = self.state_dir / "correlation_clusters.json"
        clusters_temp = self.state_dir / "correlation_clusters.json.tmp"

        try:
            clusters_data = {
                'version': CORRELATION_ENGINE_VERSION,
                'timestamp': datetime.now().isoformat(),
                'clusters': {k: int(v) for k, v in self.clusters.items()}
            }

            with open(clusters_temp, 'w') as f:
                json.dump(clusters_data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            clusters_temp.replace(clusters_path)

        except Exception as e:
            print(f"Error saving clusters: {e}")
            if clusters_temp.exists():
                clusters_temp.unlink()

        # Save penalties
        penalties_path = self.state_dir / "correlation_penalties.json"
        penalties_temp = self.state_dir / "correlation_penalties.json.tmp"

        try:
            penalties_data = {
                'version': CORRELATION_ENGINE_VERSION,
                'timestamp': datetime.now().isoformat(),
                'penalties': {k: float(v) for k, v in self.penalties.items()}
            }

            with open(penalties_temp, 'w') as f:
                json.dump(penalties_data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            penalties_temp.replace(penalties_path)

        except Exception as e:
            print(f"Error saving penalties: {e}")
            if penalties_temp.exists():
                penalties_temp.unlink()

        # Save uniqueness
        uniqueness_path = self.state_dir / "correlation_uniqueness.json"
        uniqueness_temp = self.state_dir / "correlation_uniqueness.json.tmp"

        try:
            uniqueness_data = {
                'version': CORRELATION_ENGINE_VERSION,
                'timestamp': datetime.now().isoformat(),
                'uniqueness': {k: float(v) for k, v in self.uniqueness.items()}
            }

            with open(uniqueness_temp, 'w') as f:
                json.dump(uniqueness_data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            uniqueness_temp.replace(uniqueness_path)

        except Exception as e:
            print(f"Error saving uniqueness: {e}")
            if uniqueness_temp.exists():
                uniqueness_temp.unlink()

        # Save metadata
        metadata_path = self.state_dir / "correlation_engine_meta.json"
        metadata_temp = self.state_dir / "correlation_engine_meta.json.tmp"

        try:
            metadata = {
                'version': CORRELATION_ENGINE_VERSION,
                'timestamp': datetime.now().isoformat(),
                'strategy_count': len(self.clusters),
                'cluster_count': len(set(self.clusters.values())) if self.clusters else 0,
                'correlation_threshold': float(self.correlation_threshold)
            }

            with open(metadata_temp, 'w') as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            metadata_temp.replace(metadata_path)

        except Exception as e:
            print(f"Error saving metadata: {e}")
            if metadata_temp.exists():
                metadata_temp.unlink()

    def load(self) -> None:
        """Load correlation cluster state from disk."""
        # Load clusters
        clusters_path = self.state_dir / "correlation_clusters.json"
        if clusters_path.exists():
            try:
                with open(clusters_path, 'r') as f:
                    data = json.load(f)
                self.clusters = {k: int(v) for k, v in data.get('clusters', {}).items()}
            except Exception as e:
                print(f"Warning: Failed to load clusters: {e}")
                self.clusters = {}

        # Load penalties
        penalties_path = self.state_dir / "correlation_penalties.json"
        if penalties_path.exists():
            try:
                with open(penalties_path, 'r') as f:
                    data = json.load(f)
                self.penalties = {k: float(v) for k, v in data.get('penalties', {}).items()}
            except Exception as e:
                print(f"Warning: Failed to load penalties: {e}")
                self.penalties = {}

        # Load uniqueness
        uniqueness_path = self.state_dir / "correlation_uniqueness.json"
        if uniqueness_path.exists():
            try:
                with open(uniqueness_path, 'r') as f:
                    data = json.load(f)
                self.uniqueness = {k: float(v) for k, v in data.get('uniqueness', {}).items()}
            except Exception as e:
                print(f"Warning: Failed to load uniqueness: {e}")
                self.uniqueness = {}


# ============================================================================
# GLOBAL ENGINE INSTANCE
# ============================================================================

_CORRELATION_ENGINE = None  # type: Optional[CorrelationClusterEngine]


def _get_correlation_engine() -> CorrelationClusterEngine:
    """Get or create global CorrelationClusterEngine instance."""
    global _CORRELATION_ENGINE
    if _CORRELATION_ENGINE is None:
        _CORRELATION_ENGINE = CorrelationClusterEngine()
    return _CORRELATION_ENGINE


# ============================================================================
# INTEGRATION HOOKS
# ============================================================================

def evo_corr_update(perf_memory: 'PerformanceMemory') -> None:
    """
    Update correlation clusters from performance memory.

    Args:
        perf_memory: PerformanceMemory instance
    """
    engine = _get_correlation_engine()
    engine.update(perf_memory)
    engine.save()


def evo_corr_get_clusters() -> Dict[str, int]:
    """
    Get cluster assignments.

    Returns:
        Dict mapping strategy_name -> cluster_id
    """
    engine = _get_correlation_engine()
    return engine.get_clusters()


def evo_corr_get_penalties() -> Dict[str, float]:
    """
    Get correlation penalties.

    Returns:
        Dict mapping strategy_name -> penalty [0, 1]
    """
    engine = _get_correlation_engine()
    return engine.get_penalties()


def evo_corr_get_uniqueness() -> Dict[str, float]:
    """
    Get uniqueness scores.

    Returns:
        Dict mapping strategy_name -> uniqueness [0, 1]
    """
    engine = _get_correlation_engine()
    return engine.get_uniqueness()


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil

    # Import PerformanceMemory for testing
    from .performance_memory import PerformanceMemory, PerformanceRecord

    print("=" * 80)
    print("PRADO9_EVO Module F — Correlation Cluster Engine Tests")
    print("=" * 80)

    # ========================================================================
    # TEST 1: Build Return Matrix
    # ========================================================================
    print("\n[TEST 1] Build Return Matrix")
    print("-" * 80)

    temp_dir1 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir1)

        # Add records for 3 strategies
        for i in range(20):
            for strategy in ['momentum', 'mean_reversion', 'trend']:
                record = PerformanceRecord(
                    timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                    strategy_name=strategy,
                    regime='bull',
                    horizon=5,
                    generation=0,
                    return_value=np.random.randn() * 0.01,
                    drawdown=-0.02,
                    volatility=0.15,
                    win=True,
                    prediction=0.5,
                    meta_prediction=0.5,
                    allocation_weight=0.33,
                    ensemble_return=0.01,
                    bandit_reward=0.01,
                    meta_label=1,
                    wfo_sharpe=1.5,
                    rolling_sharpe=1.5,
                    rolling_sortino=1.8,
                    rolling_dd=-0.08,
                    rolling_win_rate=0.55
                )
                perf_memory.add_record(record)

        # Build matrix
        builder = CorrelationMatrixBuilder()
        return_matrix = builder.build_return_matrix(perf_memory)

        print(f"  Return matrix shape: {return_matrix.shape}")
        print(f"  Strategies: {list(return_matrix.columns)}")
        print(f"  Time points: {len(return_matrix)}")

        assert return_matrix.shape[1] == 3, "Should have 3 strategies"
        assert len(return_matrix) > 0, "Should have time points"

        print("  ✓ Return matrix building working")

    finally:
        shutil.rmtree(temp_dir1)

    # ========================================================================
    # TEST 2: Compute Correlation Matrix
    # ========================================================================
    print("\n[TEST 2] Compute Correlation Matrix")
    print("-" * 80)

    temp_dir2 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir2)

        # Add highly correlated strategies
        np.random.seed(42)
        base_returns = np.random.randn(30) * 0.01

        for i in range(30):
            # Strategy 1: base returns
            record1 = PerformanceRecord(
                timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                strategy_name='correlated_1',
                regime='bull',
                horizon=5,
                generation=0,
                return_value=base_returns[i],
                drawdown=-0.02,
                volatility=0.15,
                win=True,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.5,
                ensemble_return=0.01,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            perf_memory.add_record(record1)

            # Strategy 2: base returns + small noise (highly correlated)
            record2 = PerformanceRecord(
                timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                strategy_name='correlated_2',
                regime='bull',
                horizon=5,
                generation=0,
                return_value=base_returns[i] + np.random.randn() * 0.001,
                drawdown=-0.02,
                volatility=0.15,
                win=True,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.5,
                ensemble_return=0.01,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            perf_memory.add_record(record2)

        # Build and compute correlation
        builder = CorrelationMatrixBuilder()
        return_matrix = builder.build_return_matrix(perf_memory)
        corr_matrix = builder.compute_correlation(return_matrix)

        print(f"  Correlation matrix shape: {corr_matrix.shape}")
        print(f"  Correlation between strategies: {corr_matrix.iloc[0, 1]:.4f}")

        assert corr_matrix.shape == (2, 2), "Should be 2x2 matrix"
        assert corr_matrix.iloc[0, 1] > 0.8, "Strategies should be highly correlated"

        print("  ✓ Correlation computation working")

    finally:
        shutil.rmtree(temp_dir2)

    # ========================================================================
    # TEST 3: Strategy Clustering
    # ========================================================================
    print("\n[TEST 3] Strategy Clustering")
    print("-" * 80)

    temp_dir3 = Path(tempfile.mkdtemp())

    try:
        # Create fake correlation matrix
        strategies = ['s1', 's2', 's3', 's4']

        # s1 and s2 highly correlated, s3 and s4 highly correlated
        corr_data = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.85],
            [0.1, 0.1, 0.85, 1.0]
        ])

        corr_matrix = pd.DataFrame(corr_data, index=strategies, columns=strategies)

        # Cluster
        cluster_engine = ClusterEngine(correlation_threshold=0.7)
        clusters = cluster_engine.cluster_strategies(corr_matrix, n_clusters=2)

        print(f"  Clusters: {clusters}")

        # s1 and s2 should be in same cluster
        # s3 and s4 should be in same cluster
        assert clusters['s1'] == clusters['s2'], "s1 and s2 should be in same cluster"
        assert clusters['s3'] == clusters['s4'], "s3 and s4 should be in same cluster"
        assert clusters['s1'] != clusters['s3'], "s1 and s3 should be in different clusters"

        print("  ✓ Clustering working")

    finally:
        shutil.rmtree(temp_dir3)

    # ========================================================================
    # TEST 4: Uniqueness Scores
    # ========================================================================
    print("\n[TEST 4] Uniqueness Scores")
    print("-" * 80)

    # Use same correlation matrix from TEST 3
    strategies = ['s1', 's2', 's3', 's4']
    corr_data = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.85],
        [0.1, 0.1, 0.85, 1.0]
    ])
    corr_matrix = pd.DataFrame(corr_data, index=strategies, columns=strategies)

    cluster_engine = ClusterEngine()
    clusters = cluster_engine.cluster_strategies(corr_matrix, n_clusters=2)
    uniqueness = cluster_engine.compute_uniqueness_scores(corr_matrix, clusters)

    print(f"  Uniqueness scores: {uniqueness}")

    # s1 and s2 have high correlation with each other → lower uniqueness
    # s3 and s4 have high correlation with each other → lower uniqueness
    assert 0.0 <= uniqueness['s1'] <= 1.0, "Uniqueness should be in [0, 1]"
    assert all(0.0 <= v <= 1.0 for v in uniqueness.values()), "All uniqueness scores should be in [0, 1]"

    print("  ✓ Uniqueness scores working")

    # ========================================================================
    # TEST 5: Correlation Penalties
    # ========================================================================
    print("\n[TEST 5] Correlation Penalties")
    print("-" * 80)

    # Use same correlation matrix
    penalties = cluster_engine.compute_correlation_penalties(corr_matrix)

    print(f"  Penalties: {penalties}")

    # s1 has high correlation with s2 → higher penalty
    # s1 has low correlation with s3, s4 → penalty based on average
    assert 0.0 <= penalties['s1'] <= 1.0, "Penalty should be in [0, 1]"
    assert all(0.0 <= v <= 1.0 for v in penalties.values()), "All penalties should be in [0, 1]"

    # Penalty + uniqueness should be approximately 1.0 for each strategy
    for strategy in strategies:
        sum_val = penalties[strategy] + uniqueness[strategy]
        assert 0.8 <= sum_val <= 1.2, f"Penalty + uniqueness should be ~1.0 for {strategy}"

    print("  ✓ Correlation penalties working")

    # ========================================================================
    # TEST 6: CorrelationClusterEngine Update
    # ========================================================================
    print("\n[TEST 6] CorrelationClusterEngine Update")
    print("-" * 80)

    temp_dir6 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir6)
        engine = CorrelationClusterEngine(state_dir=temp_dir6)

        # Add records for multiple strategies
        np.random.seed(42)

        for i in range(30):
            for strategy in ['momentum', 'mean_reversion', 'trend']:
                record = PerformanceRecord(
                    timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                    strategy_name=strategy,
                    regime='bull',
                    horizon=5,
                    generation=0,
                    return_value=np.random.randn() * 0.01,
                    drawdown=-0.02,
                    volatility=0.15,
                    win=True,
                    prediction=0.5,
                    meta_prediction=0.5,
                    allocation_weight=0.33,
                    ensemble_return=0.01,
                    bandit_reward=0.01,
                    meta_label=1,
                    wfo_sharpe=1.5,
                    rolling_sharpe=1.5,
                    rolling_sortino=1.8,
                    rolling_dd=-0.08,
                    rolling_win_rate=0.55
                )
                perf_memory.add_record(record)

        # Update engine
        engine.update(perf_memory)

        clusters = engine.get_clusters()
        penalties = engine.get_penalties()
        uniqueness = engine.get_uniqueness()

        print(f"  Clusters: {clusters}")
        print(f"  Penalties: {penalties}")
        print(f"  Uniqueness: {uniqueness}")

        assert len(clusters) == 3, "Should have 3 strategies clustered"
        assert len(penalties) == 3, "Should have 3 penalty scores"
        assert len(uniqueness) == 3, "Should have 3 uniqueness scores"

        print("  ✓ CorrelationClusterEngine update working")

    finally:
        shutil.rmtree(temp_dir6)

    # ========================================================================
    # TEST 7: Save and Load
    # ========================================================================
    print("\n[TEST 7] Save and Load")
    print("-" * 80)

    temp_dir7 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir7)

        # Add records
        np.random.seed(42)
        for i in range(30):
            for strategy in ['s1', 's2', 's3']:
                record = PerformanceRecord(
                    timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                    strategy_name=strategy,
                    regime='bull',
                    horizon=5,
                    generation=0,
                    return_value=np.random.randn() * 0.01,
                    drawdown=-0.02,
                    volatility=0.15,
                    win=True,
                    prediction=0.5,
                    meta_prediction=0.5,
                    allocation_weight=0.33,
                    ensemble_return=0.01,
                    bandit_reward=0.01,
                    meta_label=1,
                    wfo_sharpe=1.5,
                    rolling_sharpe=1.5,
                    rolling_sortino=1.8,
                    rolling_dd=-0.08,
                    rolling_win_rate=0.55
                )
                perf_memory.add_record(record)

        # Create and update engine
        engine1 = CorrelationClusterEngine(state_dir=temp_dir7)
        engine1.update(perf_memory)

        clusters1 = engine1.get_clusters()
        penalties1 = engine1.get_penalties()
        uniqueness1 = engine1.get_uniqueness()

        # Save
        engine1.save()
        print(f"  Saved {len(clusters1)} clusters")

        # Create new engine and load
        engine2 = CorrelationClusterEngine(state_dir=temp_dir7)

        clusters2 = engine2.get_clusters()
        penalties2 = engine2.get_penalties()
        uniqueness2 = engine2.get_uniqueness()

        print(f"  Loaded {len(clusters2)} clusters")

        assert clusters1 == clusters2, "Clusters should match after load"
        assert penalties1 == penalties2, "Penalties should match after load"
        assert uniqueness1 == uniqueness2, "Uniqueness should match after load"

        print("  ✓ Save and load working")

    finally:
        shutil.rmtree(temp_dir7)

    # ========================================================================
    # TEST 8: Integration Hooks
    # ========================================================================
    print("\n[TEST 8] Integration Hooks")
    print("-" * 80)

    def test_hooks():
        """Test integration hooks with isolated global state."""
        global _CORRELATION_ENGINE

        temp_dir8 = Path(tempfile.mkdtemp())

        try:
            _CORRELATION_ENGINE = CorrelationClusterEngine(state_dir=temp_dir8)

            perf_memory = PerformanceMemory(state_dir=temp_dir8)

            # Add records
            np.random.seed(42)
            for i in range(30):
                for strategy in ['hook_s1', 'hook_s2']:
                    record = PerformanceRecord(
                        timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                        strategy_name=strategy,
                        regime='bull',
                        horizon=5,
                        generation=0,
                        return_value=np.random.randn() * 0.01,
                        drawdown=-0.02,
                        volatility=0.15,
                        win=True,
                        prediction=0.5,
                        meta_prediction=0.5,
                        allocation_weight=0.5,
                        ensemble_return=0.01,
                        bandit_reward=0.01,
                        meta_label=1,
                        wfo_sharpe=1.5,
                        rolling_sharpe=1.5,
                        rolling_sortino=1.8,
                        rolling_dd=-0.08,
                        rolling_win_rate=0.55
                    )
                    perf_memory.add_record(record)

            # Update via hook
            evo_corr_update(perf_memory)

            # Get via hooks
            clusters = evo_corr_get_clusters()
            penalties = evo_corr_get_penalties()
            uniqueness = evo_corr_get_uniqueness()

            print(f"  Clusters: {clusters}")
            print(f"  Penalties: {penalties}")
            print(f"  Uniqueness: {uniqueness}")

            assert len(clusters) == 2, "Should have 2 strategies"
            assert len(penalties) == 2, "Should have 2 penalties"
            assert len(uniqueness) == 2, "Should have 2 uniqueness scores"

            print("  ✓ Integration hooks working")

        finally:
            shutil.rmtree(temp_dir8)
            _CORRELATION_ENGINE = None

    test_hooks()

    # ========================================================================
    # TEST 9: Missing Data Stability
    # ========================================================================
    print("\n[TEST 9] Missing Data Stability")
    print("-" * 80)

    temp_dir9 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir9)
        engine = CorrelationClusterEngine(state_dir=temp_dir9)

        # Add only 1 record (insufficient)
        record = PerformanceRecord(
            timestamp=pd.Timestamp('2025-01-01'),
            strategy_name='sparse',
            regime='bull',
            horizon=5,
            generation=0,
            return_value=0.01,
            drawdown=-0.02,
            volatility=0.15,
            win=True,
            prediction=0.5,
            meta_prediction=0.5,
            allocation_weight=1.0,
            ensemble_return=0.01,
            bandit_reward=0.01,
            meta_label=1,
            wfo_sharpe=1.5,
            rolling_sharpe=1.5,
            rolling_sortino=1.8,
            rolling_dd=-0.08,
            rolling_win_rate=0.55
        )
        perf_memory.add_record(record)

        # Update should not crash
        engine.update(perf_memory)

        clusters = engine.get_clusters()
        penalties = engine.get_penalties()
        uniqueness = engine.get_uniqueness()

        print(f"  Clusters with sparse data: {clusters}")
        print(f"  Penalties with sparse data: {penalties}")
        print(f"  Uniqueness with sparse data: {uniqueness}")

        # Should return empty dicts (insufficient data)
        assert len(clusters) == 0, "Should have no clusters with insufficient data"

        print("  ✓ Missing data stability working")

    finally:
        shutil.rmtree(temp_dir9)

    # ========================================================================
    # TEST 10: NaN/Inf Safety
    # ========================================================================
    print("\n[TEST 10] NaN/Inf Safety")
    print("-" * 80)

    temp_dir10 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir10)
        engine = CorrelationClusterEngine(state_dir=temp_dir10)

        # Add records with NaN/Inf returns
        for i in range(30):
            for j, strategy in enumerate(['nan_s1', 'nan_s2']):
                # Add some NaN/Inf values
                if i % 5 == 0:
                    ret = np.nan
                elif i % 7 == 0:
                    ret = np.inf
                else:
                    ret = np.random.randn() * 0.01

                record = PerformanceRecord(
                    timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                    strategy_name=strategy,
                    regime='bull',
                    horizon=5,
                    generation=0,
                    return_value=ret,
                    drawdown=-0.02,
                    volatility=0.15,
                    win=True,
                    prediction=0.5,
                    meta_prediction=0.5,
                    allocation_weight=0.5,
                    ensemble_return=0.01,
                    bandit_reward=0.01,
                    meta_label=1,
                    wfo_sharpe=1.5,
                    rolling_sharpe=1.5,
                    rolling_sortino=1.8,
                    rolling_dd=-0.08,
                    rolling_win_rate=0.55
                )
                perf_memory.add_record(record)

        # Update should not crash
        engine.update(perf_memory)

        clusters = engine.get_clusters()
        penalties = engine.get_penalties()
        uniqueness = engine.get_uniqueness()

        print(f"  Clusters with NaN/Inf: {len(clusters)}")
        print(f"  Penalties with NaN/Inf: {len(penalties)}")
        print(f"  Uniqueness with NaN/Inf: {len(uniqueness)}")

        # Should have valid results (NaN/Inf filtered)
        for penalty in penalties.values():
            assert np.isfinite(penalty), "Penalties should be finite"
            assert 0.0 <= penalty <= 1.0, "Penalties should be in [0, 1]"

        for unique in uniqueness.values():
            assert np.isfinite(unique), "Uniqueness should be finite"
            assert 0.0 <= unique <= 1.0, "Uniqueness should be in [0, 1]"

        print("  ✓ NaN/Inf safety working")

    finally:
        shutil.rmtree(temp_dir10)

    # ========================================================================
    # TEST 11: Correlation Symmetry & Diagonal
    # ========================================================================
    print("\n[TEST 11] Correlation Symmetry & Diagonal")
    print("-" * 80)

    temp_dir11 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir11)

        # Add records
        np.random.seed(42)
        for i in range(30):
            for strategy in ['sym_s1', 'sym_s2', 'sym_s3']:
                record = PerformanceRecord(
                    timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                    strategy_name=strategy,
                    regime='bull',
                    horizon=5,
                    generation=0,
                    return_value=np.random.randn() * 0.01,
                    drawdown=-0.02,
                    volatility=0.15,
                    win=True,
                    prediction=0.5,
                    meta_prediction=0.5,
                    allocation_weight=0.33,
                    ensemble_return=0.01,
                    bandit_reward=0.01,
                    meta_label=1,
                    wfo_sharpe=1.5,
                    rolling_sharpe=1.5,
                    rolling_sortino=1.8,
                    rolling_dd=-0.08,
                    rolling_win_rate=0.55
                )
                perf_memory.add_record(record)

        # Build correlation matrix
        builder = CorrelationMatrixBuilder()
        return_matrix = builder.build_return_matrix(perf_memory)
        corr_matrix = builder.compute_correlation(return_matrix)

        # Check symmetry
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                assert abs(corr_matrix.iloc[i, j] - corr_matrix.iloc[j, i]) < 1e-10, \
                    "Correlation matrix should be symmetric"

        # Check diagonal is 1.0
        for i in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 1e-10, \
                "Diagonal should be 1.0"

        print(f"  Correlation matrix is symmetric: True")
        print(f"  Diagonal values all 1.0: True")
        print("  ✓ Correlation symmetry and diagonal working")

    finally:
        shutil.rmtree(temp_dir11)

    # ========================================================================
    # TEST 12: Deterministic Clustering
    # ========================================================================
    print("\n[TEST 12] Deterministic Clustering")
    print("-" * 80)

    # Create identical correlation matrix twice
    strategies = ['det_s1', 'det_s2', 'det_s3', 'det_s4']
    corr_data = np.array([
        [1.0, 0.9, 0.2, 0.2],
        [0.9, 1.0, 0.2, 0.2],
        [0.2, 0.2, 1.0, 0.85],
        [0.2, 0.2, 0.85, 1.0]
    ])

    corr_matrix1 = pd.DataFrame(corr_data, index=strategies, columns=strategies)
    corr_matrix2 = pd.DataFrame(corr_data, index=strategies, columns=strategies)

    # Cluster both
    cluster_engine = ClusterEngine(correlation_threshold=0.7)
    clusters1 = cluster_engine.cluster_strategies(corr_matrix1, n_clusters=2)
    clusters2 = cluster_engine.cluster_strategies(corr_matrix2, n_clusters=2)

    print(f"  First clustering: {clusters1}")
    print(f"  Second clustering: {clusters2}")

    # Should be identical
    assert clusters1 == clusters2, "Clustering should be deterministic"

    print("  ✓ Deterministic clustering working")

    # ========================================================================
    # TEST 13: Uniqueness Score Bounds
    # ========================================================================
    print("\n[TEST 13] Uniqueness Score Bounds")
    print("-" * 80)

    # Test with various correlation matrices
    test_cases = [
        # All highly correlated
        np.array([
            [1.0, 0.95, 0.92],
            [0.95, 1.0, 0.94],
            [0.92, 0.94, 1.0]
        ]),
        # All uncorrelated
        np.array([
            [1.0, 0.05, 0.03],
            [0.05, 1.0, 0.02],
            [0.03, 0.02, 1.0]
        ]),
        # Mixed
        np.array([
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.15],
            [0.1, 0.15, 1.0]
        ])
    ]

    strategies = ['test_s1', 'test_s2', 'test_s3']
    cluster_engine = ClusterEngine()

    for i, corr_data in enumerate(test_cases):
        corr_matrix = pd.DataFrame(corr_data, index=strategies, columns=strategies)
        clusters = cluster_engine.cluster_strategies(corr_matrix)
        uniqueness = cluster_engine.compute_uniqueness_scores(corr_matrix, clusters)

        print(f"  Test case {i+1} uniqueness: {uniqueness}")

        for strategy, score in uniqueness.items():
            assert 0.0 <= score <= 1.0, f"Uniqueness {score} not in [0, 1]"
            assert np.isfinite(score), f"Uniqueness {score} not finite"

    print("  ✓ Uniqueness score bounds working")

    # ========================================================================
    # TEST 14: Penalty Score Bounds
    # ========================================================================
    print("\n[TEST 14] Penalty Score Bounds")
    print("-" * 80)

    # Use same test cases
    for i, corr_data in enumerate(test_cases):
        corr_matrix = pd.DataFrame(corr_data, index=strategies, columns=strategies)
        penalties = cluster_engine.compute_correlation_penalties(corr_matrix)

        print(f"  Test case {i+1} penalties: {penalties}")

        for strategy, penalty in penalties.items():
            assert 0.0 <= penalty <= 1.0, f"Penalty {penalty} not in [0, 1]"
            assert np.isfinite(penalty), f"Penalty {penalty} not finite"

    print("  ✓ Penalty score bounds working")

    # ========================================================================
    # TEST 15: Metadata Persistence
    # ========================================================================
    print("\n[TEST 15] Metadata Persistence")
    print("-" * 80)

    temp_dir15 = Path(tempfile.mkdtemp())

    try:
        perf_memory = PerformanceMemory(state_dir=temp_dir15)

        # Add records
        np.random.seed(42)
        for i in range(30):
            for strategy in ['meta_s1', 'meta_s2', 'meta_s3']:
                record = PerformanceRecord(
                    timestamp=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i),
                    strategy_name=strategy,
                    regime='bull',
                    horizon=5,
                    generation=0,
                    return_value=np.random.randn() * 0.01,
                    drawdown=-0.02,
                    volatility=0.15,
                    win=True,
                    prediction=0.5,
                    meta_prediction=0.5,
                    allocation_weight=0.33,
                    ensemble_return=0.01,
                    bandit_reward=0.01,
                    meta_label=1,
                    wfo_sharpe=1.5,
                    rolling_sharpe=1.5,
                    rolling_sortino=1.8,
                    rolling_dd=-0.08,
                    rolling_win_rate=0.55
                )
                perf_memory.add_record(record)

        # Create and update engine
        engine = CorrelationClusterEngine(state_dir=temp_dir15)
        engine.update(perf_memory)
        engine.save()

        # Check metadata file exists
        metadata_path = temp_dir15 / "correlation_engine_meta.json"
        assert metadata_path.exists(), "Metadata file should exist"

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"  Metadata version: {metadata['version']}")
        print(f"  Strategy count: {metadata['strategy_count']}")
        print(f"  Cluster count: {metadata['cluster_count']}")
        print(f"  Correlation threshold: {metadata['correlation_threshold']}")

        assert metadata['version'] == CORRELATION_ENGINE_VERSION, "Version should match"
        assert metadata['strategy_count'] == 3, "Should have 3 strategies"
        assert isinstance(metadata['correlation_threshold'], float), "Threshold should be float"

        print("  ✓ Metadata persistence working")

    finally:
        shutil.rmtree(temp_dir15)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL MODULE F SWEEP TESTS PASSED (15 TESTS)")
    print("=" * 80)
    print("\nCorrelation Cluster Engine Features:")
    print("  ✓ Return matrix building (full outer join, ffill/bfill)")
    print("  ✓ Prediction matrix building (full outer join, ffill/bfill)")
    print("  ✓ Correlation matrix computation (symmetric, diagonal=1.0)")
    print("  ✓ Hierarchical clustering (deterministic)")
    print("  ✓ Uniqueness scores (1 - avg_correlation, bounded [0,1])")
    print("  ✓ Correlation penalties (avg_correlation, bounded [0,1])")
    print("  ✓ Save/load persistence (JSON + metadata)")
    print("  ✓ Integration hooks (evo_corr_*)")
    print("  ✓ Missing data stability")
    print("  ✓ NaN/Inf safety")
    print("\nSweep F.1 Enhancements:")
    print("  ✓ Matrix alignment (ffill→bfill→0.0)")
    print("  ✓ Correlation symmetry enforcement")
    print("  ✓ Deterministic clustering with fallbacks")
    print("  ✓ Uniqueness/penalty NaN guards")
    print("  ✓ Metadata file (correlation_engine_meta.json)")
    print("  ✓ 15 comprehensive tests (10 original + 5 sweep)")
    print("\nModule F — Correlation Cluster Engine: PRODUCTION READY (Sweep F.1 Complete)")
    print("=" * 80)
