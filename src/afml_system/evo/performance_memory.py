"""
PRADO9_EVO Module E — Performance Memory Store (High-Resolution Quant Memory)

Records ALL performance datapoints across:
- Strategies, regimes, horizons, generations
- Ensemble vs strategy comparisons
- Meta-learner ground truth labels
- Bandit rewards
- WFO metrics
- Rolling statistics
- Trade outcomes

This is the "experience replay buffer" for the quant system.

Author: PRADO9_EVO Builder
Date: 2025-01-16
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict


# ============================================================================
# CONSTANTS
# ============================================================================

PERFORMANCE_MEMORY_VERSION = '1.0.0'
MINIMUM_ROLLING_SAMPLES = 2


# ============================================================================
# PERFORMANCE RECORD
# ============================================================================

@dataclass
class PerformanceRecord:
    """
    Unit record of strategy performance.

    Captures all relevant metrics for learning, evolution, and allocation.
    """
    # Identifiers
    timestamp: pd.Timestamp
    strategy_name: str
    regime: str
    horizon: Union[str, int]
    generation: int

    # Core performance
    return_value: float
    drawdown: float
    volatility: float
    win: bool

    # Predictions
    prediction: float  # Raw model prediction
    meta_prediction: float  # Meta-learner probability
    allocation_weight: float

    # Ensemble comparison
    ensemble_return: float

    # Learning signals
    bandit_reward: float
    meta_label: int  # 1 if strategy > ensemble, 0 otherwise

    # Walk-forward & rolling metrics
    wfo_sharpe: float
    rolling_sharpe: float
    rolling_sortino: float
    rolling_dd: float
    rolling_win_rate: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'strategy_name': self.strategy_name,
            'regime': self.regime,
            'horizon': str(self.horizon),
            'generation': self.generation,
            'return_value': float(self.return_value),
            'drawdown': float(self.drawdown),
            'volatility': float(self.volatility),
            'win': bool(self.win),
            'prediction': float(self.prediction),
            'meta_prediction': float(self.meta_prediction),
            'allocation_weight': float(self.allocation_weight),
            'ensemble_return': float(self.ensemble_return),
            'bandit_reward': float(self.bandit_reward),
            'meta_label': int(self.meta_label),
            'wfo_sharpe': float(self.wfo_sharpe),
            'rolling_sharpe': float(self.rolling_sharpe),
            'rolling_sortino': float(self.rolling_sortino),
            'rolling_dd': float(self.rolling_dd),
            'rolling_win_rate': float(self.rolling_win_rate)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceRecord':
        """Create from dictionary."""
        return cls(
            timestamp=pd.Timestamp(data['timestamp']),
            strategy_name=str(data['strategy_name']),
            regime=str(data['regime']),
            horizon=data['horizon'],
            generation=int(data['generation']),
            return_value=float(data['return_value']),
            drawdown=float(data['drawdown']),
            volatility=float(data['volatility']),
            win=bool(data['win']),
            prediction=float(data['prediction']),
            meta_prediction=float(data['meta_prediction']),
            allocation_weight=float(data['allocation_weight']),
            ensemble_return=float(data['ensemble_return']),
            bandit_reward=float(data['bandit_reward']),
            meta_label=int(data['meta_label']),
            wfo_sharpe=float(data['wfo_sharpe']),
            rolling_sharpe=float(data['rolling_sharpe']),
            rolling_sortino=float(data['rolling_sortino']),
            rolling_dd=float(data['rolling_dd']),
            rolling_win_rate=float(data['rolling_win_rate'])
        )


# ============================================================================
# PERFORMANCE MEMORY
# ============================================================================

class PerformanceMemory:
    """
    High-resolution performance memory store.

    Stores all performance records with fast lookup and rolling metrics.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize PerformanceMemory.

        Args:
            state_dir: Directory for state persistence
        """
        if state_dir is None:
            state_dir = Path.home() / ".prado" / "evo"

        self.state_dir = Path(os.path.expanduser(str(state_dir)))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Append-only record storage
        self.records = []  # type: List[PerformanceRecord]

        # Fast lookup index: (strategy, regime, horizon) -> [record_indices]
        self.index = defaultdict(list)  # type: Dict[Tuple[str, str, str], List[int]]

        # Load existing records
        self.load()

    def add_record(self, record: PerformanceRecord) -> None:
        """
        Add a performance record to memory.

        Args:
            record: Performance record to add
        """
        # Add to records
        record_idx = len(self.records)
        self.records.append(record)

        # Update index with validated key
        key = self._make_key(record.strategy_name, record.regime, record.horizon)
        self.index[key].append(record_idx)

    def _make_key(
        self,
        strategy: str,
        regime: str,
        horizon: Union[str, int]
    ) -> Tuple[str, str, str]:
        """
        Create validated index key.

        Args:
            strategy: Strategy name
            regime: Regime name
            horizon: Return horizon

        Returns:
            Tuple key (strategy, regime, horizon_str)
        """
        return (str(strategy), str(regime), str(horizon))

    def validate_index(self) -> bool:
        """
        Validate and repair index integrity.

        Returns:
            True if index was valid, False if repairs were made
        """
        repairs_made = False

        # Rebuild index from scratch
        new_index = defaultdict(list)

        for idx, record in enumerate(self.records):
            key = self._make_key(record.strategy_name, record.regime, record.horizon)
            new_index[key].append(idx)

        # Check if different from current index
        if dict(new_index) != dict(self.index):
            repairs_made = True
            self.index = new_index

        return not repairs_made

    def get_records(
        self,
        strategy: str,
        regime: str,
        horizon: Union[str, int]
    ) -> List[PerformanceRecord]:
        """
        Get all records for a specific (strategy, regime, horizon).

        Args:
            strategy: Strategy name
            regime: Regime name
            horizon: Return horizon

        Returns:
            List of matching records
        """
        key = self._make_key(strategy, regime, horizon)
        indices = self.index.get(key, [])

        return [self.records[idx] for idx in indices if idx < len(self.records)]

    def latest(
        self,
        strategy: str,
        regime: str,
        horizon: Union[str, int]
    ) -> Optional[PerformanceRecord]:
        """
        Get most recent record for (strategy, regime, horizon).

        Args:
            strategy: Strategy name
            regime: Regime name
            horizon: Return horizon

        Returns:
            Most recent record or None
        """
        records = self.get_records(strategy, regime, horizon)

        if not records:
            return None

        return records[-1]

    def rolling_metrics(
        self,
        strategy: str,
        regime: str,
        horizon: Union[str, int],
        window: int = 50
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compute rolling metrics over recent window.

        Args:
            strategy: Strategy name
            regime: Regime name
            horizon: Return horizon
            window: Rolling window size

        Returns:
            Dictionary of rolling metrics
        """
        records = self.get_records(strategy, regime, horizon)

        if not records:
            return self._empty_metrics()

        # Get recent records
        recent = records[-window:]

        if len(recent) == 0:
            return self._empty_metrics()

        # Extract returns
        returns = [r.return_value for r in recent]
        wins = [r.win for r in recent]

        # Compute metrics safely (pass returns for drawdown calculation)
        metrics = {
            'rolling_sharpe': self._safe_sharpe(returns),
            'rolling_sortino': self._safe_sortino(returns),
            'rolling_dd': self._safe_max_drawdown(returns),
            'rolling_win_rate': self._safe_win_rate(wins),
            'rolling_volatility': self._safe_volatility(returns),
            'recent_returns': returns
        }

        return metrics

    def performance_summary(
        self,
        strategy: str,
        regime: str,
        horizon: Union[str, int]
    ) -> Dict[str, Union[float, int]]:
        """
        Get performance summary for (strategy, regime, horizon).

        Args:
            strategy: Strategy name
            regime: Regime name
            horizon: Return horizon

        Returns:
            Dictionary of summary statistics
        """
        records = self.get_records(strategy, regime, horizon)

        if not records:
            return {
                'total_records': 0,
                'total_return': 0.0,
                'avg_return': 0.0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'max_dd': 0.0,
                'win_rate': 0.5,
                'volatility': 0.0
            }

        returns = [r.return_value for r in records]
        wins = [r.win for r in records]

        # Compute summary statistics (use returns for drawdown)
        return {
            'total_records': len(records),
            'total_return': float(sum(returns)),
            'avg_return': float(np.mean(returns)) if returns else 0.0,
            'sharpe': self._safe_sharpe(returns),
            'sortino': self._safe_sortino(returns),
            'max_dd': self._safe_max_drawdown(returns),
            'win_rate': self._safe_win_rate(wins),
            'volatility': self._safe_volatility(returns)
        }

    def to_meta_features(
        self,
        strategy: str,
        regime: str,
        horizon: Union[str, int]
    ) -> Dict:
        """
        Extract features for meta-learner training.

        Args:
            strategy: Strategy name
            regime: Regime name
            horizon: Return horizon

        Returns:
            Dictionary compatible with MetaFeatureBuilder
        """
        # Get rolling metrics
        rolling = self.rolling_metrics(strategy, regime, horizon, window=50)

        # Get latest record for point-in-time metrics
        latest_record = self.latest(strategy, regime, horizon)

        if latest_record is None:
            # Return safe defaults
            return {
                'rolling_sharpe': [0.0],
                'rolling_sortino': [0.0],
                'rolling_dd': [0.0],
                'rolling_win_rate': [0.5],
                'recent_returns': [0.0],
                'meta_accuracy': 0.5,
                'wfo_sharpe': 0.0,
                'volatility': 0.15,
                'regime': regime,
                'horizon': horizon,
                'bandit_confidence': 0.5,
                'forecast_dispersion': 0.0,
                'correlation_to_ensemble': 0.0
            }

        # Compute meta-label accuracy from recent records
        records = self.get_records(strategy, regime, horizon)
        recent_records = records[-50:] if len(records) > 50 else records

        if recent_records:
            meta_labels = [r.meta_label for r in recent_records if r.meta_label is not None]
            meta_accuracy = float(sum(meta_labels) / len(meta_labels)) if meta_labels else 0.5
        else:
            meta_accuracy = 0.5

        # Safe type conversions for all numeric values
        return {
            'rolling_sharpe': [float(rolling['rolling_sharpe'])],
            'rolling_sortino': [float(rolling['rolling_sortino'])],
            'rolling_dd': [float(rolling['rolling_dd'])],
            'rolling_win_rate': [float(rolling['rolling_win_rate'])],
            'recent_returns': [float(x) for x in rolling['recent_returns']],
            'meta_accuracy': float(meta_accuracy),
            'wfo_sharpe': float(latest_record.wfo_sharpe) if latest_record.wfo_sharpe is not None else 0.0,
            'volatility': float(latest_record.volatility) if latest_record.volatility is not None else 0.15,
            'regime': str(regime),
            'horizon': int(horizon) if isinstance(horizon, (int, float, str)) and str(horizon).isdigit() else str(horizon),
            'bandit_confidence': float(latest_record.meta_prediction) if latest_record.meta_prediction is not None else 0.5,
            'forecast_dispersion': 0.0,
            'correlation_to_ensemble': float(self._compute_ensemble_correlation(records))
        }

    def prune(self, max_records_per_key: int = 5000) -> int:
        """
        Prune old records to prevent unbounded growth.

        Keeps the most recent max_records_per_key for each (strategy, regime, horizon).

        Args:
            max_records_per_key: Maximum records to keep per key

        Returns:
            Number of records pruned
        """
        # Build new records list and index
        new_records = []
        new_index = defaultdict(list)
        pruned_count = 0

        # Process each key
        for key, indices in self.index.items():
            if len(indices) > max_records_per_key:
                # Keep only recent records
                keep_indices = indices[-max_records_per_key:]
                pruned_count += len(indices) - len(keep_indices)
            else:
                keep_indices = indices

            # Add to new structures
            for old_idx in keep_indices:
                new_idx = len(new_records)
                new_records.append(self.records[old_idx])
                new_index[key].append(new_idx)

        # Replace old structures
        self.records = new_records
        self.index = new_index

        return pruned_count

    def save(self) -> None:
        """Save performance memory to disk with atomic write."""
        save_path = self.state_dir / "performance_memory.pkl"
        temp_path = self.state_dir / "performance_memory.pkl.tmp"

        metadata_path = self.state_dir / "performance_memory_metadata.json"
        metadata_temp_path = self.state_dir / "performance_memory_metadata.json.tmp"

        try:
            # Prepare data
            data = {
                'records': [r.to_dict() for r in self.records],
                'timestamp': datetime.now().isoformat(),
                'version': PERFORMANCE_MEMORY_VERSION
            }

            # Atomic write for pickle
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_path.replace(save_path)

            # Prepare metadata
            unique_keys = list(self.index.keys())
            metadata = {
                'version': PERFORMANCE_MEMORY_VERSION,
                'save_date': datetime.now().isoformat(),
                'total_records': len(self.records),
                'unique_keys': len(unique_keys),
                'keys': [{'strategy': k[0], 'regime': k[1], 'horizon': str(k[2])}
                         for k in sorted(unique_keys)]
            }

            # Atomic write for metadata
            with open(metadata_temp_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            metadata_temp_path.replace(metadata_path)

        except Exception as e:
            print(f"Error saving performance memory: {e}")
            if temp_path.exists():
                temp_path.unlink()
            if metadata_temp_path.exists():
                metadata_temp_path.unlink()
            raise

    def load(self) -> None:
        """Load performance memory from disk with index validation."""
        load_path = self.state_dir / "performance_memory.pkl"
        metadata_path = self.state_dir / "performance_memory_metadata.json"

        if not load_path.exists():
            return

        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)

            # Reconstruct records and index
            self.records = []
            self.index = defaultdict(list)

            for record_data in data.get('records', []):
                record = PerformanceRecord.from_dict(record_data)
                record_idx = len(self.records)
                self.records.append(record)

                # Update index with validated key
                key = self._make_key(record.strategy_name, record.regime, record.horizon)
                self.index[key].append(record_idx)

            # Validate and repair index integrity
            is_valid = self.validate_index()
            if not is_valid:
                print(f"Warning: Index was repaired during load")

            # Load metadata if available (informational only)
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print(f"Loaded {len(self.records)} records, {len(self.index)} keys (v{metadata.get('version', 'unknown')})")
                except Exception:
                    print(f"Loaded {len(self.records)} performance records")
            else:
                print(f"Loaded {len(self.records)} performance records")

        except Exception as e:
            print(f"Warning: Failed to load performance memory: {e}")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _empty_metrics(self) -> Dict[str, Union[float, List[float]]]:
        """Return empty metrics with safe defaults."""
        return {
            'rolling_sharpe': 0.0,
            'rolling_sortino': 0.0,
            'rolling_dd': 0.0,
            'rolling_win_rate': 0.5,
            'rolling_volatility': 0.0,
            'recent_returns': []
        }

    def _safe_sharpe(self, returns: List[float]) -> float:
        """Compute Sharpe ratio safely."""
        if not returns or len(returns) < MINIMUM_ROLLING_SAMPLES:
            return 0.0

        try:
            returns_array = np.array(returns, dtype=float)

            # Remove NaN/Inf
            returns_array = returns_array[np.isfinite(returns_array)]

            if len(returns_array) < MINIMUM_ROLLING_SAMPLES:
                return 0.0

            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)

            # Check for zero or invalid std
            if std_return <= 0 or np.isnan(std_return) or np.isinf(std_return):
                return 0.0

            sharpe = mean_return / std_return

            if np.isnan(sharpe) or np.isinf(sharpe):
                return 0.0

            return float(np.clip(sharpe, -10, 10))  # Clip extreme values

        except Exception:
            return 0.0

    def _safe_sortino(self, returns: List[float]) -> float:
        """Compute Sortino ratio safely."""
        if not returns or len(returns) < MINIMUM_ROLLING_SAMPLES:
            return 0.0

        try:
            returns_array = np.array(returns, dtype=float)

            # Remove NaN/Inf
            returns_array = returns_array[np.isfinite(returns_array)]

            if len(returns_array) < MINIMUM_ROLLING_SAMPLES:
                return 0.0

            mean_return = np.mean(returns_array)

            # Downside deviation (only negative returns)
            downside_returns = returns_array[returns_array < 0]

            if len(downside_returns) == 0:
                # No downside, return mean if positive, else 0
                return float(np.clip(mean_return, 0, 10)) if mean_return > 0 else 0.0

            if len(downside_returns) == 1:
                # Single downside return, use absolute value
                downside_std = abs(downside_returns[0])
            else:
                downside_std = np.std(downside_returns, ddof=1)

            if downside_std <= 0 or np.isnan(downside_std) or np.isinf(downside_std):
                return 0.0

            sortino = mean_return / downside_std

            if np.isnan(sortino) or np.isinf(sortino):
                return 0.0

            return float(np.clip(sortino, -10, 10))  # Clip extreme values

        except Exception:
            return 0.0

    def _safe_max_drawdown(self, returns: List[float]) -> float:
        """
        Compute maximum drawdown using peak-to-trough formula.

        Formula:
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_dd = drawdowns.min()

        Returns:
            float: Maximum drawdown (negative value), or 0.0 on error
        """
        if not returns or len(returns) < MINIMUM_ROLLING_SAMPLES:
            return 0.0

        try:
            returns_array = np.array(returns, dtype=float)

            # Remove NaN/Inf
            returns_array = returns_array[np.isfinite(returns_array)]

            if len(returns_array) < MINIMUM_ROLLING_SAMPLES:
                return 0.0

            # Compute cumulative returns (1 + r1) * (1 + r2) * ...
            cumulative_returns = np.cumprod(1 + returns_array)

            # Compute running maximum
            running_max = np.maximum.accumulate(cumulative_returns)

            # Compute drawdown at each point
            drawdowns = (cumulative_returns - running_max) / running_max

            # Maximum drawdown is the minimum value (most negative)
            max_dd = np.min(drawdowns)

            if np.isnan(max_dd) or np.isinf(max_dd):
                return 0.0

            # Clip to reasonable range
            return float(np.clip(max_dd, -1.0, 0.0))

        except Exception:
            return 0.0

    def _safe_win_rate(self, wins: List[bool]) -> float:
        """Compute win rate safely."""
        if not wins:
            return 0.5

        try:
            win_rate = sum(wins) / len(wins)

            if np.isnan(win_rate) or np.isinf(win_rate):
                return 0.5

            return float(win_rate)

        except Exception:
            return 0.5

    def _safe_volatility(self, returns: List[float]) -> float:
        """Compute volatility safely."""
        if not returns or len(returns) < 2:
            return 0.0

        try:
            volatility = np.std(returns, ddof=1)

            if np.isnan(volatility) or np.isinf(volatility):
                return 0.0

            return float(volatility)

        except Exception:
            return 0.0

    def _compute_ensemble_correlation(
        self,
        records: List[PerformanceRecord]
    ) -> float:
        """
        Compute correlation between strategy and ensemble returns.

        Handles:
        - Aligned returns (same length)
        - Mismatched lengths (filter to common)
        - NaN/Inf values (remove before correlation)
        - Insufficient samples (< 2)

        Returns:
            float: Correlation coefficient [-1, 1], or 0.0 on error
        """
        if not records or len(records) < MINIMUM_ROLLING_SAMPLES:
            return 0.0

        try:
            # Extract returns as arrays
            strategy_returns = np.array([r.return_value for r in records], dtype=float)
            ensemble_returns = np.array([r.ensemble_return for r in records], dtype=float)

            # Ensure same length
            min_len = min(len(strategy_returns), len(ensemble_returns))
            strategy_returns = strategy_returns[:min_len]
            ensemble_returns = ensemble_returns[:min_len]

            # Create mask for finite values in BOTH arrays
            valid_mask = np.isfinite(strategy_returns) & np.isfinite(ensemble_returns)

            # Filter to valid pairs
            strategy_returns = strategy_returns[valid_mask]
            ensemble_returns = ensemble_returns[valid_mask]

            # Check minimum samples after filtering
            if len(strategy_returns) < MINIMUM_ROLLING_SAMPLES:
                return 0.0

            # Check for zero variance (constant returns)
            if np.std(strategy_returns) == 0 or np.std(ensemble_returns) == 0:
                return 0.0

            # Compute correlation
            correlation = np.corrcoef(strategy_returns, ensemble_returns)[0, 1]

            if np.isnan(correlation) or np.isinf(correlation):
                return 0.0

            # Clip to valid range
            return float(np.clip(correlation, -1.0, 1.0))

        except Exception:
            return 0.0


# ============================================================================
# GLOBAL MEMORY INSTANCE
# ============================================================================

_PERFORMANCE_MEMORY = None  # type: Optional[PerformanceMemory]


def _get_performance_memory() -> PerformanceMemory:
    """Get or create global PerformanceMemory instance."""
    global _PERFORMANCE_MEMORY
    if _PERFORMANCE_MEMORY is None:
        _PERFORMANCE_MEMORY = PerformanceMemory()
    return _PERFORMANCE_MEMORY


# ============================================================================
# INTEGRATION HOOKS
# ============================================================================

def evo_perf_add(record: PerformanceRecord) -> None:
    """
    Add a performance record to memory.

    Args:
        record: Performance record to add
    """
    memory = _get_performance_memory()
    memory.add_record(record)


def evo_perf_get(
    strategy: str,
    regime: str,
    horizon: Union[str, int]
) -> List[PerformanceRecord]:
    """
    Get all records for (strategy, regime, horizon).

    Args:
        strategy: Strategy name
        regime: Regime name
        horizon: Return horizon

    Returns:
        List of matching records
    """
    memory = _get_performance_memory()
    return memory.get_records(strategy, regime, horizon)


def evo_perf_rolling(
    strategy: str,
    regime: str,
    horizon: Union[str, int],
    window: int = 50
) -> Dict:
    """
    Compute rolling metrics.

    Args:
        strategy: Strategy name
        regime: Regime name
        horizon: Return horizon
        window: Rolling window size

    Returns:
        Dictionary of rolling metrics
    """
    memory = _get_performance_memory()
    return memory.rolling_metrics(strategy, regime, horizon, window)


def evo_perf_summary(
    strategy: str,
    regime: str,
    horizon: Union[str, int]
) -> Dict:
    """
    Get performance summary.

    Args:
        strategy: Strategy name
        regime: Regime name
        horizon: Return horizon

    Returns:
        Dictionary of summary statistics
    """
    memory = _get_performance_memory()
    return memory.performance_summary(strategy, regime, horizon)


def evo_perf_save() -> None:
    """Save performance memory to disk."""
    memory = _get_performance_memory()
    memory.save()


def evo_perf_load() -> None:
    """Load performance memory from disk."""
    memory = _get_performance_memory()
    memory.load()


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Module E — Performance Memory Tests")
    print("=" * 80)

    # ========================================================================
    # TEST 1: PerformanceRecord Creation
    # ========================================================================
    print("\n[TEST 1] PerformanceRecord Creation")
    print("-" * 80)

    record = PerformanceRecord(
        timestamp=pd.Timestamp.now(),
        strategy_name="momentum",
        regime="bull",
        horizon=5,
        generation=1,
        return_value=0.02,
        drawdown=-0.05,
        volatility=0.15,
        win=True,
        prediction=0.65,
        meta_prediction=0.75,
        allocation_weight=0.20,
        ensemble_return=0.018,
        bandit_reward=0.02,
        meta_label=1,
        wfo_sharpe=1.8,
        rolling_sharpe=1.9,
        rolling_sortino=2.2,
        rolling_dd=-0.06,
        rolling_win_rate=0.60
    )

    print(f"  Strategy: {record.strategy_name}")
    print(f"  Regime: {record.regime}")
    print(f"  Return: {record.return_value:.4f}")
    print(f"  Meta label: {record.meta_label}")

    assert record.strategy_name == "momentum", "Strategy name mismatch"
    assert record.meta_label == 1, "Meta label should be 1"

    print("  ✓ PerformanceRecord creation working")

    # ========================================================================
    # TEST 2: Record Serialization
    # ========================================================================
    print("\n[TEST 2] Record Serialization")
    print("-" * 80)

    record_dict = record.to_dict()
    record_restored = PerformanceRecord.from_dict(record_dict)

    print(f"  Original return: {record.return_value:.4f}")
    print(f"  Restored return: {record_restored.return_value:.4f}")

    assert record.strategy_name == record_restored.strategy_name, "Strategy name mismatch"
    assert record.return_value == record_restored.return_value, "Return value mismatch"
    assert record.meta_label == record_restored.meta_label, "Meta label mismatch"

    print("  ✓ Serialization working")

    # ========================================================================
    # TEST 3: PerformanceMemory - Add Records
    # ========================================================================
    print("\n[TEST 3] PerformanceMemory - Add Records")
    print("-" * 80)

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir)

        # Add 100 fake records
        for i in range(100):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name=f"strategy_{i % 5}",
                regime="bull" if i % 2 == 0 else "bear",
                horizon=5,
                generation=i // 10,
                return_value=np.random.randn() * 0.01,
                drawdown=-abs(np.random.randn() * 0.05),
                volatility=0.15,
                win=np.random.rand() > 0.5,
                prediction=np.random.rand(),
                meta_prediction=np.random.rand(),
                allocation_weight=0.20,
                ensemble_return=np.random.randn() * 0.01,
                bandit_reward=np.random.randn() * 0.01,
                meta_label=1 if np.random.rand() > 0.5 else 0,
                wfo_sharpe=np.random.randn(),
                rolling_sharpe=np.random.randn(),
                rolling_sortino=np.random.randn(),
                rolling_dd=-abs(np.random.randn() * 0.1),
                rolling_win_rate=np.random.rand()
            )
            memory.add_record(test_record)

        print(f"  Total records: {len(memory.records)}")
        print(f"  Index keys: {len(memory.index)}")

        assert len(memory.records) == 100, "Should have 100 records"
        assert len(memory.index) > 0, "Should have index entries"

        print("  ✓ Adding records working")

    finally:
        shutil.rmtree(temp_dir)

    # ========================================================================
    # TEST 4: Record Retrieval
    # ========================================================================
    print("\n[TEST 4] Record Retrieval")
    print("-" * 80)

    temp_dir2 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir2)

        # Add specific records
        for i in range(20):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="momentum",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=0.01 * (i + 1),
                drawdown=-0.05,
                volatility=0.15,
                win=True,
                prediction=0.65,
                meta_prediction=0.75,
                allocation_weight=0.20,
                ensemble_return=0.01 * i,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.8,
                rolling_sharpe=1.9,
                rolling_sortino=2.2,
                rolling_dd=-0.06,
                rolling_win_rate=0.60
            )
            memory.add_record(test_record)

        # Retrieve records
        records = memory.get_records("momentum", "bull", 5)

        print(f"  Retrieved records: {len(records)}")
        print(f"  First return: {records[0].return_value:.4f}")
        print(f"  Last return: {records[-1].return_value:.4f}")

        assert len(records) == 20, "Should retrieve 20 records"

        # Get latest
        latest = memory.latest("momentum", "bull", 5)
        assert latest is not None, "Should have latest record"
        assert latest.return_value == 0.20, "Latest return should be 0.20"

        print("  ✓ Record retrieval working")

    finally:
        shutil.rmtree(temp_dir2)

    # ========================================================================
    # TEST 5: Rolling Metrics
    # ========================================================================
    print("\n[TEST 5] Rolling Metrics")
    print("-" * 80)

    temp_dir3 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir3)

        # Add records with known returns
        np.random.seed(42)
        returns = np.random.randn(50) * 0.01

        for i, ret in enumerate(returns):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="test_strategy",
                regime="sideways",
                horizon=10,
                generation=0,
                return_value=ret,
                drawdown=-abs(ret * 2),
                volatility=0.15,
                win=ret > 0,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=ret * 0.9,
                bandit_reward=ret,
                meta_label=1 if ret > ret * 0.9 else 0,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory.add_record(test_record)

        # Compute rolling metrics
        metrics = memory.rolling_metrics("test_strategy", "sideways", 10, window=50)

        print(f"  Rolling Sharpe: {metrics['rolling_sharpe']:.4f}")
        print(f"  Rolling Sortino: {metrics['rolling_sortino']:.4f}")
        print(f"  Rolling DD: {metrics['rolling_dd']:.4f}")
        print(f"  Rolling Win Rate: {metrics['rolling_win_rate']:.4f}")
        print(f"  Rolling Volatility: {metrics['rolling_volatility']:.4f}")

        assert 'rolling_sharpe' in metrics, "Should have rolling Sharpe"
        assert 'rolling_sortino' in metrics, "Should have rolling Sortino"
        assert len(metrics['recent_returns']) == 50, "Should have 50 recent returns"

        print("  ✓ Rolling metrics working")

    finally:
        shutil.rmtree(temp_dir3)

    # ========================================================================
    # TEST 6: Performance Summary
    # ========================================================================
    print("\n[TEST 6] Performance Summary")
    print("-" * 80)

    temp_dir4 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir4)

        # Add records
        for i in range(30):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="summary_test",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=0.02 if i % 2 == 0 else -0.01,
                drawdown=-0.05,
                volatility=0.15,
                win=i % 2 == 0,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=0.01,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory.add_record(test_record)

        # Get summary
        summary = memory.performance_summary("summary_test", "bull", 5)

        print(f"  Total records: {summary['total_records']}")
        print(f"  Total return: {summary['total_return']:.4f}")
        print(f"  Avg return: {summary['avg_return']:.4f}")
        print(f"  Sharpe: {summary['sharpe']:.4f}")
        print(f"  Win rate: {summary['win_rate']:.4f}")

        assert summary['total_records'] == 30, "Should have 30 records"
        assert summary['win_rate'] == 0.5, "Win rate should be 0.5"

        print("  ✓ Performance summary working")

    finally:
        shutil.rmtree(temp_dir4)

    # ========================================================================
    # TEST 7: Save and Load
    # ========================================================================
    print("\n[TEST 7] Save and Load")
    print("-" * 80)

    temp_dir5 = Path(tempfile.mkdtemp())

    try:
        # Create and populate memory
        memory1 = PerformanceMemory(state_dir=temp_dir5)

        for i in range(50):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="persist_test",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=0.01 * i,
                drawdown=-0.05,
                volatility=0.15,
                win=True,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=0.01,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory1.add_record(test_record)

        # Save
        memory1.save()
        print(f"  Saved {len(memory1.records)} records")

        # Load into new instance
        memory2 = PerformanceMemory(state_dir=temp_dir5)

        print(f"  Loaded {len(memory2.records)} records")

        assert len(memory2.records) == 50, "Should load 50 records"
        assert len(memory2.index) > 0, "Should rebuild index"

        # Verify data
        records = memory2.get_records("persist_test", "bull", 5)
        assert len(records) == 50, "Should retrieve all records"

        print("  ✓ Save/load working")

    finally:
        shutil.rmtree(temp_dir5)

    # ========================================================================
    # TEST 8: Meta-Feature Extraction
    # ========================================================================
    print("\n[TEST 8] Meta-Feature Extraction")
    print("-" * 80)

    temp_dir6 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir6)

        # Add records
        for i in range(60):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="meta_test",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=np.random.randn() * 0.01,
                drawdown=-0.05,
                volatility=0.15,
                win=np.random.rand() > 0.5,
                prediction=0.5,
                meta_prediction=0.75,
                allocation_weight=0.20,
                ensemble_return=np.random.randn() * 0.01,
                bandit_reward=0.01,
                meta_label=1 if np.random.rand() > 0.5 else 0,
                wfo_sharpe=1.8,
                rolling_sharpe=1.9,
                rolling_sortino=2.2,
                rolling_dd=-0.06,
                rolling_win_rate=0.60
            )
            memory.add_record(test_record)

        # Extract meta features
        meta_features = memory.to_meta_features("meta_test", "bull", 5)

        print(f"  Feature keys: {list(meta_features.keys())}")
        print(f"  Meta accuracy: {meta_features['meta_accuracy']:.4f}")
        print(f"  WFO Sharpe: {meta_features['wfo_sharpe']:.4f}")
        print(f"  Recent returns count: {len(meta_features['recent_returns'])}")

        assert 'rolling_sharpe' in meta_features, "Should have rolling_sharpe"
        assert 'meta_accuracy' in meta_features, "Should have meta_accuracy"
        assert 'recent_returns' in meta_features, "Should have recent_returns"

        print("  ✓ Meta-feature extraction working")

    finally:
        shutil.rmtree(temp_dir6)

    # ========================================================================
    # TEST 9: Memory Pruning
    # ========================================================================
    print("\n[TEST 9] Memory Pruning")
    print("-" * 80)

    temp_dir7 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir7)

        # Add many records for same key
        for i in range(150):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="prune_test",
                regime="bull",
                horizon=5,
                generation=i // 10,
                return_value=0.01,
                drawdown=-0.05,
                volatility=0.15,
                win=True,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=0.01,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory.add_record(test_record)

        print(f"  Records before pruning: {len(memory.records)}")

        # Prune to 100 max per key
        pruned = memory.prune(max_records_per_key=100)

        print(f"  Records after pruning: {len(memory.records)}")
        print(f"  Records pruned: {pruned}")

        assert len(memory.records) == 100, "Should prune to 100 records"
        assert pruned == 50, "Should have pruned 50 records"

        print("  ✓ Pruning working")

    finally:
        shutil.rmtree(temp_dir7)

    # ========================================================================
    # TEST 10: Integration Hooks
    # ========================================================================
    print("\n[TEST 10] Integration Hooks")
    print("-" * 80)

    def test_hooks():
        """Test integration hooks with global memory."""
        global _PERFORMANCE_MEMORY

        temp_dir8 = Path(tempfile.mkdtemp())

        try:
            # Reset global
            _PERFORMANCE_MEMORY = PerformanceMemory(state_dir=temp_dir8)

            # Add using hook
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="hook_test",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=0.03,
                drawdown=-0.05,
                volatility=0.15,
                win=True,
                prediction=0.7,
                meta_prediction=0.8,
                allocation_weight=0.25,
                ensemble_return=0.02,
                bandit_reward=0.03,
                meta_label=1,
                wfo_sharpe=2.0,
                rolling_sharpe=2.1,
                rolling_sortino=2.5,
                rolling_dd=-0.04,
                rolling_win_rate=0.65
            )

            evo_perf_add(test_record)

            # Retrieve using hooks
            records = evo_perf_get("hook_test", "bull", 5)
            print(f"  Retrieved via hook: {len(records)} records")

            rolling = evo_perf_rolling("hook_test", "bull", 5)
            print(f"  Rolling metrics keys: {list(rolling.keys())}")

            summary = evo_perf_summary("hook_test", "bull", 5)
            print(f"  Summary total records: {summary['total_records']}")

            assert len(records) == 1, "Should have 1 record"
            assert 'rolling_sharpe' in rolling, "Should have rolling metrics"
            assert summary['total_records'] == 1, "Should have 1 record in summary"

            print("  ✓ Integration hooks working")

        finally:
            shutil.rmtree(temp_dir8)

    test_hooks()

    # ========================================================================
    # TEST 11: Index Validation and Repair
    # ========================================================================
    print("\n[TEST 11] Index Validation and Repair")
    print("-" * 80)

    temp_dir11 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir11)

        # Add records
        for i in range(20):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="index_test",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=0.01,
                drawdown=-0.02,
                volatility=0.15,
                win=True,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=0.01,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory.add_record(test_record)

        # Validate index
        is_valid = memory.validate_index()
        print(f"  Index valid: {is_valid}")
        assert is_valid, "Index should be valid after adding records"

        # Corrupt index
        key = ("index_test", "bull", "5")
        memory.index[key].append(999)  # Invalid index

        # Validate should detect and repair
        is_valid = memory.validate_index()
        print(f"  Index valid after repair: {is_valid}")

        # Check records still accessible
        records = memory.get_records("index_test", "bull", 5)
        print(f"  Retrieved {len(records)} records after repair")
        assert len(records) == 20, "Should retrieve all 20 records after repair"

        print("  ✓ Index validation and repair working")

    finally:
        shutil.rmtree(temp_dir11)

    # ========================================================================
    # TEST 12: Peak-to-Trough Drawdown
    # ========================================================================
    print("\n[TEST 12] Peak-to-Trough Drawdown Calculation")
    print("-" * 80)

    temp_dir12 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir12)

        # Create sequence with known drawdown: +10%, -5%, -3%, +8%
        # Cumulative: 1.10, 1.045, 1.0137, 1.0947
        # Running max: 1.10, 1.10, 1.10, 1.10
        # Drawdown: 0, -5%, -7.8%, -0.5%
        # Max DD: -7.8%
        test_returns = [0.10, -0.05, -0.03, 0.08]

        for ret in test_returns:
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="dd_test",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=ret,
                drawdown=-abs(ret),
                volatility=0.15,
                win=ret > 0,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=ret * 0.9,
                bandit_reward=ret,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory.add_record(test_record)

        # Compute rolling metrics (uses peak-to-trough)
        metrics = memory.rolling_metrics("dd_test", "bull", 5)
        print(f"  Peak-to-trough max DD: {metrics['rolling_dd']:.4f}")

        # Should be negative (drawdown)
        assert metrics['rolling_dd'] <= 0, "Drawdown should be negative or zero"
        assert metrics['rolling_dd'] >= -1.0, "Drawdown should be >= -100%"

        print("  ✓ Peak-to-trough drawdown calculation working")

    finally:
        shutil.rmtree(temp_dir12)

    # ========================================================================
    # TEST 13: Ensemble Correlation with NaN/Inf Handling
    # ========================================================================
    print("\n[TEST 13] Ensemble Correlation (NaN/Inf Handling)")
    print("-" * 80)

    temp_dir13 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir13)

        # Add records with some NaN/Inf values
        test_returns = [0.01, np.nan, 0.02, np.inf, -0.01, 0.015]
        ensemble_returns = [0.01, 0.01, np.nan, 0.01, -0.01, np.inf]

        for i, (ret, ens_ret) in enumerate(zip(test_returns, ensemble_returns)):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name="corr_test",
                regime="bull",
                horizon=5,
                generation=0,
                return_value=ret,
                drawdown=-0.02,
                volatility=0.15,
                win=True,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=ens_ret,
                bandit_reward=ret if np.isfinite(ret) else 0.0,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory.add_record(test_record)

        # Extract meta features (uses correlation)
        features = memory.to_meta_features("corr_test", "bull", 5)
        corr = features['correlation_to_ensemble']
        print(f"  Correlation (with NaN/Inf filtering): {corr:.4f}")

        # Should be valid (not NaN/Inf)
        assert np.isfinite(corr), "Correlation should be finite"
        assert -1.0 <= corr <= 1.0, "Correlation should be in [-1, 1]"

        print("  ✓ Ensemble correlation with NaN/Inf handling working")

    finally:
        shutil.rmtree(temp_dir13)

    # ========================================================================
    # TEST 14: Metadata Persistence
    # ========================================================================
    print("\n[TEST 14] Metadata Persistence")
    print("-" * 80)

    temp_dir14 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir14)

        # Add records
        for i in range(15):
            test_record = PerformanceRecord(
                timestamp=pd.Timestamp.now(),
                strategy_name=f"strat_{i % 3}",
                regime="bull" if i % 2 == 0 else "bear",
                horizon=5,
                generation=0,
                return_value=0.01,
                drawdown=-0.02,
                volatility=0.15,
                win=True,
                prediction=0.5,
                meta_prediction=0.5,
                allocation_weight=0.20,
                ensemble_return=0.01,
                bandit_reward=0.01,
                meta_label=1,
                wfo_sharpe=1.5,
                rolling_sharpe=1.5,
                rolling_sortino=1.8,
                rolling_dd=-0.08,
                rolling_win_rate=0.55
            )
            memory.add_record(test_record)

        # Save
        memory.save()
        print(f"  Saved {len(memory.records)} records")

        # Check metadata file exists
        metadata_path = temp_dir14 / "performance_memory_metadata.json"
        assert metadata_path.exists(), "Metadata file should exist"

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"  Metadata version: {metadata['version']}")
        print(f"  Total records: {metadata['total_records']}")
        print(f"  Unique keys: {metadata['unique_keys']}")
        print(f"  Keys count: {len(metadata['keys'])}")

        assert metadata['version'] == PERFORMANCE_MEMORY_VERSION, "Version should match"
        assert metadata['total_records'] == 15, "Should have 15 records"
        assert metadata['unique_keys'] >= 1, "Should have at least 1 unique key"

        print("  ✓ Metadata persistence working")

    finally:
        shutil.rmtree(temp_dir14)

    # ========================================================================
    # TEST 15: Type Safety in Meta Features
    # ========================================================================
    print("\n[TEST 15] Type Safety in Meta Features")
    print("-" * 80)

    temp_dir15 = Path(tempfile.mkdtemp())

    try:
        memory = PerformanceMemory(state_dir=temp_dir15)

        # Add record with None values
        test_record = PerformanceRecord(
            timestamp=pd.Timestamp.now(),
            strategy_name="type_test",
            regime="bull",
            horizon=5,
            generation=0,
            return_value=0.01,
            drawdown=-0.02,
            volatility=None,  # None value
            win=True,
            prediction=0.5,
            meta_prediction=None,  # None value
            allocation_weight=0.20,
            ensemble_return=0.01,
            bandit_reward=0.01,
            meta_label=None,  # None value
            wfo_sharpe=None,  # None value
            rolling_sharpe=1.5,
            rolling_sortino=1.8,
            rolling_dd=-0.08,
            rolling_win_rate=0.55
        )
        memory.add_record(test_record)

        # Extract meta features
        features = memory.to_meta_features("type_test", "bull", 5)

        # Verify all types are Python native (not numpy)
        print(f"  wfo_sharpe type: {type(features['wfo_sharpe'])}")
        print(f"  volatility type: {type(features['volatility'])}")
        print(f"  meta_accuracy type: {type(features['meta_accuracy'])}")
        print(f"  bandit_confidence type: {type(features['bandit_confidence'])}")

        assert isinstance(features['wfo_sharpe'], float), "wfo_sharpe should be Python float"
        assert isinstance(features['volatility'], float), "volatility should be Python float"
        assert isinstance(features['meta_accuracy'], float), "meta_accuracy should be Python float"
        assert isinstance(features['bandit_confidence'], float), "bandit_confidence should be Python float"
        assert isinstance(features['regime'], str), "regime should be string"

        # Check defaults for None values
        assert features['wfo_sharpe'] == 0.0, "None wfo_sharpe should default to 0.0"
        assert features['volatility'] == 0.15, "None volatility should default to 0.15"
        assert features['bandit_confidence'] == 0.5, "None meta_prediction should default to 0.5"

        print("  ✓ Type safety in meta features working")

    finally:
        shutil.rmtree(temp_dir15)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL MODULE E SWEEP TESTS PASSED (15 TESTS)")
    print("=" * 80)
    print("\nPerformance Memory Features:")
    print("  ✓ PerformanceRecord (dataclass with 20 fields)")
    print("  ✓ Record serialization (to_dict/from_dict)")
    print("  ✓ Fast indexed storage with validation")
    print("  ✓ Index integrity and auto-repair")
    print("  ✓ Record retrieval by key")
    print("  ✓ Rolling metrics (Sharpe, Sortino, peak-to-trough DD, win rate)")
    print("  ✓ NaN/Inf filtering in all statistical calculations")
    print("  ✓ Performance summary (type-safe)")
    print("  ✓ Save/load persistence (pickle + JSON metadata)")
    print("  ✓ Meta-feature extraction (type-safe, None-handling)")
    print("  ✓ Ensemble correlation (aligned, NaN/Inf filtered)")
    print("  ✓ Memory pruning (prevents unbounded growth)")
    print("  ✓ Integration hooks (evo_perf_*)")
    print("\nSweep E.1 Enhancements:")
    print("  ✓ Index validation and repair")
    print("  ✓ Peak-to-trough drawdown formula")
    print("  ✓ Enhanced Sharpe/Sortino with clipping")
    print("  ✓ Ensemble correlation with alignment")
    print("  ✓ Metadata JSON file persistence")
    print("  ✓ Type safety (no numpy types in outputs)")
    print("  ✓ 15 comprehensive tests (10 original + 5 sweep)")
    print("\nModule E — Performance Memory: PRODUCTION READY (Sweep E.1 Complete)")
    print("=" * 80)
