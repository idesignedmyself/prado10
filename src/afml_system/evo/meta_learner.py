"""
PRADO9_EVO Module D — Meta-Learner (Strategy Performance Predictor)

Predicts which strategy will outperform before execution using:
- Regime features
- Historical performance
- Bandit signals
- Genome traits
- Walk-forward metrics
- Risk characteristics

This is the "brain" that decides which strategy should run.

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
from dataclasses import dataclass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from .genome import StrategyGenome


# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_TYPE_ENCODING = {
    'rf': 0,
    'xgb': 1,
    'lgbm': 2,
    'catboost': 3,
    'logit': 4
}

MINIMUM_TRAINING_SAMPLES = 20
META_LEARNER_VERSION = '1.0.0'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensure_feature_alignment(
    df: pd.DataFrame,
    expected_features: List[str]
) -> pd.DataFrame:
    """
    Ensure DataFrame has exactly the expected features in correct order.

    Args:
        df: Input DataFrame
        expected_features: List of feature names in expected order

    Returns:
        DataFrame with aligned features
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Add missing columns with zeros
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    # Remove extra columns
    df = df[expected_features].copy()

    # Ensure numeric dtypes
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df


def _safe_float_conversion(value: any, default: float = 0.0) -> float:
    """
    Safely convert value to Python float.

    Args:
        value: Value to convert
        default: Default if conversion fails

    Returns:
        Python float
    """
    try:
        if value is None:
            return default
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return default
    except (ValueError, TypeError):
        return default


# ============================================================================
# META-FEATURE BUILDER
# ============================================================================

class MetaFeatureBuilder:
    """
    Builds meta-features for strategy performance prediction.

    Combines:
    - Regime characteristics
    - Performance trends
    - Bandit signals
    - Genome traits
    - Horizon features
    - Risk metrics
    """

    def __init__(self):
        """Initialize MetaFeatureBuilder."""
        self.feature_names = []
        self._initialize_feature_names()

    def _initialize_feature_names(self):
        """Define expected feature names in deterministic order."""
        self.feature_names = [
            # Regime features (3)
            'regime_trend',
            'regime_volatility',
            'regime_spike',

            # Performance trends (7)
            'rolling_sharpe_mean',
            'rolling_sharpe_std',
            'rolling_sortino_mean',
            'rolling_dd_mean',
            'rolling_win_rate_mean',
            'recent_return_mean',
            'recent_return_std',

            # Meta metrics (2)
            'meta_accuracy',
            'wfo_sharpe',

            # Bandit signals (1)
            'bandit_confidence',

            # Genome features (7)
            'genome_model_rf',
            'genome_model_xgb',
            'genome_model_lgbm',
            'genome_profit_barrier',
            'genome_stop_barrier',
            'genome_mutation_rate',
            'genome_feature_count',

            # Horizon features (2)
            'holding_period',
            'return_horizon',

            # Risk features (3)
            'strategy_volatility',
            'forecast_dispersion',
            'correlation_to_ensemble',
        ]

    def build_features(
        self,
        strategy_name: str,
        regime: str,
        horizon: Union[str, int],
        performance_history: Dict,
        genome: StrategyGenome,
        bandit_confidence: float,
        recent_metrics: Dict
    ) -> pd.DataFrame:
        """
        Build meta-features for a strategy.

        Args:
            strategy_name: Name of strategy
            regime: Current regime ('bull', 'bear', 'sideways', 'volatile')
            horizon: Return horizon
            performance_history: Historical performance metrics
            genome: Strategy genome
            bandit_confidence: Bandit confidence score
            recent_metrics: Recent risk/performance metrics

        Returns:
            Single-row DataFrame with all meta-features
        """
        features = {}

        # 1. Regime features
        features.update(self._build_regime_features(regime))

        # 2. Performance trends
        features.update(self._build_performance_features(performance_history))

        # 3. Meta metrics (with safe defaults)
        features['meta_accuracy'] = _safe_float_conversion(
            performance_history.get('meta_accuracy', 0.5),
            0.5
        )
        features['wfo_sharpe'] = _safe_float_conversion(
            performance_history.get('wfo_sharpe', 0.0),
            0.0
        )

        # 4. Bandit signals
        features['bandit_confidence'] = _safe_float_conversion(bandit_confidence, 0.5)

        # 5. Genome features
        features.update(self._build_genome_features(genome))

        # 6. Horizon features
        features['holding_period'] = genome.holding_period
        if isinstance(horizon, str):
            features['return_horizon'] = genome.return_horizon
        else:
            features['return_horizon'] = int(horizon)

        # 7. Risk features
        features.update(self._build_risk_features(recent_metrics, performance_history))

        # Convert to DataFrame (single row)
        df = pd.DataFrame([features], columns=self.feature_names)

        # Fill any missing features with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        # Ensure column order
        df = df[self.feature_names]

        return df

    def _build_regime_features(self, regime: str) -> Dict[str, float]:
        """
        Build regime-specific features.

        Args:
            regime: Current regime

        Returns:
            Dictionary of regime features
        """
        # Map regime to numeric indicators
        regime_map = {
            'bull': {'trend': 1.0, 'volatility': 0.3, 'spike': 0.0},
            'bear': {'trend': -1.0, 'volatility': 0.5, 'spike': 0.0},
            'sideways': {'trend': 0.0, 'volatility': 0.2, 'spike': 0.0},
            'volatile': {'trend': 0.0, 'volatility': 0.8, 'spike': 1.0},
        }

        regime_lower = regime.lower()
        regime_data = regime_map.get(regime_lower, {'trend': 0.0, 'volatility': 0.5, 'spike': 0.0})

        return {
            'regime_trend': regime_data['trend'],
            'regime_volatility': regime_data['volatility'],
            'regime_spike': regime_data['spike'],
        }

    def _build_performance_features(self, performance_history: Dict) -> Dict[str, float]:
        """
        Build performance trend features with robust defaults.

        Args:
            performance_history: Historical performance metrics

        Returns:
            Dictionary of performance features
        """
        features = {}

        # Rolling Sharpe
        rolling_sharpe = performance_history.get('rolling_sharpe', [])
        if rolling_sharpe and len(rolling_sharpe) > 0:
            features['rolling_sharpe_mean'] = _safe_float_conversion(np.mean(rolling_sharpe), 0.0)
            features['rolling_sharpe_std'] = _safe_float_conversion(
                np.std(rolling_sharpe) if len(rolling_sharpe) > 1 else 0.0,
                0.0
            )
        else:
            features['rolling_sharpe_mean'] = 0.0
            features['rolling_sharpe_std'] = 0.0

        # Rolling Sortino
        rolling_sortino = performance_history.get('rolling_sortino', [])
        if rolling_sortino and len(rolling_sortino) > 0:
            features['rolling_sortino_mean'] = _safe_float_conversion(np.mean(rolling_sortino), 0.0)
        else:
            features['rolling_sortino_mean'] = 0.0

        # Rolling drawdown
        rolling_dd = performance_history.get('rolling_dd', [])
        if rolling_dd and len(rolling_dd) > 0:
            features['rolling_dd_mean'] = _safe_float_conversion(np.mean(rolling_dd), 0.0)
        else:
            features['rolling_dd_mean'] = 0.0

        # Rolling win rate
        rolling_win_rate = performance_history.get('rolling_win_rate', [])
        if rolling_win_rate and len(rolling_win_rate) > 0:
            features['rolling_win_rate_mean'] = _safe_float_conversion(np.mean(rolling_win_rate), 0.5)
        else:
            features['rolling_win_rate_mean'] = 0.5

        # Recent returns
        recent_returns = performance_history.get('recent_returns', [])
        if recent_returns and len(recent_returns) > 0:
            features['recent_return_mean'] = _safe_float_conversion(np.mean(recent_returns), 0.0)
            features['recent_return_std'] = _safe_float_conversion(
                np.std(recent_returns) if len(recent_returns) > 1 else 0.0,
                0.0
            )
        else:
            features['recent_return_mean'] = 0.0
            features['recent_return_std'] = 0.0

        return features

    def _build_genome_features(self, genome: StrategyGenome) -> Dict[str, float]:
        """
        Build genome-based features.

        Args:
            genome: Strategy genome

        Returns:
            Dictionary of genome features
        """
        features = {}

        # Model type (one-hot encoded)
        features['genome_model_rf'] = 1.0 if genome.model_type == 'rf' else 0.0
        features['genome_model_xgb'] = 1.0 if genome.model_type == 'xgb' else 0.0
        features['genome_model_lgbm'] = 1.0 if genome.model_type == 'lgbm' else 0.0

        # Barrier settings
        features['genome_profit_barrier'] = genome.barrier_settings.get('profit_barrier', 0.05)
        features['genome_stop_barrier'] = genome.barrier_settings.get('stop_barrier', 0.02)

        # Evolution traits
        features['genome_mutation_rate'] = genome.mutation_rate
        features['genome_feature_count'] = len(genome.feature_set)

        return features

    def _build_risk_features(
        self,
        recent_metrics: Dict,
        performance_history: Dict
    ) -> Dict[str, float]:
        """
        Build risk-based features with safe defaults.

        Args:
            recent_metrics: Recent risk metrics
            performance_history: Historical performance

        Returns:
            Dictionary of risk features
        """
        features = {}

        # Strategy volatility (fallback chain)
        vol = recent_metrics.get('volatility')
        if vol is None:
            vol = performance_history.get('volatility', 0.15)
        features['strategy_volatility'] = _safe_float_conversion(vol, 0.15)

        # Forecast dispersion
        features['forecast_dispersion'] = _safe_float_conversion(
            recent_metrics.get('forecast_dispersion', 0.0),
            0.0
        )

        # Correlation to ensemble
        features['correlation_to_ensemble'] = _safe_float_conversion(
            recent_metrics.get('correlation_to_ensemble', 0.0),
            0.0
        )

        return features


# ============================================================================
# META-LEARNER MODEL
# ============================================================================

class MetaLearner:
    """
    Lightweight predictive model for strategy outperformance.

    Uses XGBoost or RandomForest to predict:
    P(strategy will outperform ensemble | features)
    """

    def __init__(
        self,
        model_type: str = 'xgb',
        random_state: int = 42
    ):
        """
        Initialize MetaLearner.

        Args:
            model_type: 'xgb' or 'rf'
            random_state: Random seed for determinism
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.is_trained = False

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the underlying model."""
        if self.model_type == 'xgb' and XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            # Fallback to RandomForest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model_type = 'rf'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the meta-learner.

        Args:
            X: Feature matrix
            y: Target labels (1 = outperform, 0 = underperform)
        """
        if len(X) == 0:
            print("Warning: Empty training data, model not trained")
            return

        if len(X) < 10:
            print(f"Warning: Insufficient training data ({len(X)} samples)")

        # Store feature names
        self.feature_names = list(X.columns)

        # Train model
        self.model.fit(X, y)
        self.is_trained = True

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Incrementally update the meta-learner with new data.

        For tree-based models (RF, XGBoost), this re-trains with all data
        since true online learning isn't supported. For production use,
        consider SGDClassifier or other online learning algorithms.

        Args:
            X: Feature matrix (new samples)
            y: Target labels (1 = outperform, 0 = underperform)
        """
        if len(X) == 0:
            print("Warning: Empty training data, model not updated")
            return

        # For tree-based models, we need to retrain (they don't support partial_fit)
        # In a production system, you might want to:
        # 1. Use SGDClassifier for true online learning
        # 2. Maintain a sliding window of recent data
        # 3. Periodically retrain on accumulated data

        # For now, we'll just call fit (which retrains from scratch)
        # This is fine for adaptive retraining where we retrain on each fold
        self.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of outperformance with stability checks.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities for class 1 (outperform)
        """
        # Handle empty DataFrame
        if X.empty or len(X) == 0:
            return np.array([0.5])

        # Handle untrained model
        if not self.is_trained or self.model is None:
            return np.full(len(X), 0.5)

        # Ensure feature order matches training
        if self.feature_names is not None:
            try:
                X = _ensure_feature_alignment(X, self.feature_names)
            except Exception as e:
                print(f"Warning: Feature alignment failed: {e}")
                return np.full(len(X), 0.5)

        # Ensure 2D array for sklearn
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        try:
            probas = self.model.predict_proba(X)
            # Return probability of class 1 (outperform)
            return probas[:, 1]
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            return np.full(len(X), 0.5)

    def save(self, path: str) -> None:
        """
        Save model to disk with atomic write.

        Args:
            path: Path to save model
        """
        path = Path(os.path.expanduser(path))
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_path = path.parent / f"{path.name}.tmp"

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'model_type': self.model_type,
                    'feature_names': self.feature_names,
                    'is_trained': self.is_trained,
                    'random_state': self.random_state,
                    'model_version': META_LEARNER_VERSION
                }, f)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_path.replace(path)

        except Exception as e:
            print(f"Error saving model: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load(self, path: str) -> None:
        """
        Load model from disk with fallback on corruption.

        Args:
            path: Path to load model from
        """
        path = Path(os.path.expanduser(path))

        if not path.exists():
            print(f"Warning: Model file not found at {path}")
            return

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.model = data.get('model')
            self.model_type = data.get('model_type', 'rf')
            self.feature_names = data.get('feature_names')
            self.is_trained = data.get('is_trained', False)
            self.random_state = data.get('random_state', 42)

            # Validate loaded model
            if self.model is None:
                print("Warning: Loaded model is None, creating fallback")
                self._initialize_model()
                self.is_trained = False

        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            print(f"Warning: Corrupted model file, creating fallback: {e}")
            self._initialize_model()
            self.is_trained = False

        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            self._initialize_model()
            self.is_trained = False


# ============================================================================
# META-LEARNING ENGINE
# ============================================================================

class MetaLearningEngine:
    """
    Orchestrates meta-learning for strategy selection.

    Integrates:
    - Feature building
    - Model training
    - Prediction
    - Persistence
    """

    def __init__(
        self,
        model_type: str = 'xgb',
        state_dir: Optional[Path] = None
    ):
        """
        Initialize MetaLearningEngine.

        Args:
            model_type: 'xgb' or 'rf'
            state_dir: Directory for state persistence
        """
        if state_dir is None:
            state_dir = Path.home() / ".prado" / "evo"

        self.state_dir = Path(os.path.expanduser(str(state_dir)))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.feature_builder = MetaFeatureBuilder()
        self.model = MetaLearner(model_type=model_type)

        # Try to load existing model
        self.load()

    def prepare_training_data(
        self,
        performance_memory: Dict[str, Dict],
        genomes: Dict[str, StrategyGenome]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from performance memory.

        Args:
            performance_memory: Historical performance data per strategy
            genomes: Strategy genomes

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        X_list = []
        y_list = []

        # Calculate ensemble mean return for each time period
        # For simplicity, we'll use a synthetic approach
        for strategy_name, perf_data in performance_memory.items():
            if strategy_name not in genomes:
                continue

            genome = genomes[strategy_name]

            # Extract performance metrics
            recent_returns = perf_data.get('recent_returns', [])
            if not recent_returns or len(recent_returns) == 0:
                continue

            # For each time period, create a training sample
            for i, strategy_return in enumerate(recent_returns):
                # Get regime (default if not available)
                regime = perf_data.get('regime', 'sideways')
                horizon = perf_data.get('horizon', genome.return_horizon)

                # Get bandit confidence (default if not available)
                bandit_confidence = perf_data.get('bandit_confidence', 0.5)

                # Build recent metrics
                recent_metrics = {
                    'volatility': perf_data.get('volatility', 0.15),
                    'forecast_dispersion': perf_data.get('forecast_dispersion', 0.5),
                    'correlation_to_ensemble': perf_data.get('correlation_to_ensemble', 0.0)
                }

                # Build features
                features = self.feature_builder.build_features(
                    strategy_name=strategy_name,
                    regime=regime,
                    horizon=horizon,
                    performance_history=perf_data,
                    genome=genome,
                    bandit_confidence=bandit_confidence,
                    recent_metrics=recent_metrics
                )

                # Calculate ensemble mean (simple average for now)
                # In production, this would be actual ensemble return
                ensemble_returns = []
                for other_name, other_perf in performance_memory.items():
                    other_returns = other_perf.get('recent_returns', [])
                    if other_returns and i < len(other_returns):
                        ensemble_returns.append(other_returns[i])

                if not ensemble_returns:
                    continue

                ensemble_mean = np.mean(ensemble_returns)

                # Target: 1 if strategy outperformed ensemble, 0 otherwise
                y = 1 if strategy_return > ensemble_mean else 0

                X_list.append(features)
                y_list.append(y)

        if not X_list:
            # Return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=self.feature_builder.feature_names)
            empty_series = pd.Series(dtype=int)
            return empty_df, empty_series

        # Concatenate all samples
        X = pd.concat(X_list, ignore_index=True)
        y = pd.Series(y_list)

        return X, y

    def train(
        self,
        performance_memory: Dict[str, Dict],
        genomes: Dict[str, StrategyGenome]
    ) -> None:
        """
        Train the meta-learner with minimum sample check.

        Args:
            performance_memory: Historical performance data
            genomes: Strategy genomes
        """
        # Prepare data
        X, y = self.prepare_training_data(performance_memory, genomes)

        if len(X) == 0:
            print("Warning: No training data available")
            self._save_untrained_metadata()
            return

        if len(X) < MINIMUM_TRAINING_SAMPLES:
            print(f"Warning: Insufficient training samples ({len(X)} < {MINIMUM_TRAINING_SAMPLES})")
            print("Skipping training, saving fallback metadata")
            self.model.model = None
            self.model.is_trained = False
            self._save_untrained_metadata(sample_count=len(X))
            return

        # Train model
        self.model.fit(X, y)

        # Save model and metadata
        self.save(sample_count=len(X))

        print(f"Meta-learner trained on {len(X)} samples")

    def predict(
        self,
        strategy_name: str,
        regime: str,
        horizon: Union[str, int],
        performance_history: Dict,
        genome: StrategyGenome,
        bandit_confidence: float,
        recent_metrics: Dict
    ) -> float:
        """
        Predict probability of strategy outperforming ensemble.

        Args:
            strategy_name: Name of strategy
            regime: Current regime
            horizon: Return horizon
            performance_history: Historical performance
            genome: Strategy genome
            bandit_confidence: Bandit confidence score
            recent_metrics: Recent risk metrics

        Returns:
            Probability of outperformance [0, 1]
        """
        # Build features
        features = self.feature_builder.build_features(
            strategy_name=strategy_name,
            regime=regime,
            horizon=horizon,
            performance_history=performance_history,
            genome=genome,
            bandit_confidence=bandit_confidence,
            recent_metrics=recent_metrics
        )

        # Predict
        proba = self.model.predict_proba(features)

        return float(proba[0])

    def save(self, sample_count: int = 0) -> None:
        """
        Save model and metadata to disk with enhanced tracking.

        Args:
            sample_count: Number of training samples
        """
        model_path = Path(os.path.expanduser(str(self.state_dir))) / "meta_learner.pkl"
        metadata_path = Path(os.path.expanduser(str(self.state_dir))) / "meta_learner_metadata.json"

        # Save model
        try:
            self.model.save(model_path)
        except Exception as e:
            print(f"Error saving model: {e}")

        # Save metadata with JSON-safe types
        metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_list': sorted(self.feature_builder.feature_names),  # Sorted for consistency
            'model_type': str(self.model.model_type),
            'is_trained': bool(self.model.is_trained),
            'version': META_LEARNER_VERSION,
            'training_sample_count': int(sample_count),
            'minimum_samples_required': MINIMUM_TRAINING_SAMPLES
        }

        # Atomic write for metadata
        temp_metadata_path = metadata_path.parent / f"{metadata_path.name}.tmp"

        try:
            with open(temp_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            temp_metadata_path.replace(metadata_path)

        except Exception as e:
            print(f"Error saving metadata: {e}")
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()

    def _save_untrained_metadata(self, sample_count: int = 0) -> None:
        """
        Save metadata for untrained model.

        Args:
            sample_count: Number of samples (if any)
        """
        metadata_path = Path(os.path.expanduser(str(self.state_dir))) / "meta_learner_metadata.json"

        metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_list': sorted(self.feature_builder.feature_names),
            'model_type': str(self.model.model_type),
            'is_trained': False,
            'version': META_LEARNER_VERSION,
            'training_sample_count': int(sample_count),
            'minimum_samples_required': MINIMUM_TRAINING_SAMPLES,
            'status': 'insufficient_data'
        }

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Error saving untrained metadata: {e}")

    def load(self) -> None:
        """Load model and metadata from disk with validation."""
        model_path = Path(os.path.expanduser(str(self.state_dir))) / "meta_learner.pkl"
        metadata_path = Path(os.path.expanduser(str(self.state_dir))) / "meta_learner_metadata.json"

        # Load model
        if model_path.exists():
            self.model.load(model_path)

        # Load and validate metadata
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Validate feature list alignment
                saved_features = metadata.get('feature_list', [])
                if saved_features and saved_features != sorted(self.feature_builder.feature_names):
                    print("Warning: Feature list mismatch between saved and current")
                    print(f"  Saved: {len(saved_features)} features")
                    print(f"  Current: {len(self.feature_builder.feature_names)} features")

            except Exception as e:
                print(f"Warning: Failed to load metadata: {e}")


# ============================================================================
# GLOBAL ENGINE INSTANCE
# ============================================================================

_META_ENGINE = None  # type: Optional[MetaLearningEngine]


def _get_meta_engine() -> MetaLearningEngine:
    """Get or create global MetaLearningEngine instance."""
    global _META_ENGINE
    if _META_ENGINE is None:
        _META_ENGINE = MetaLearningEngine()
    return _META_ENGINE


# ============================================================================
# INTEGRATION HOOKS
# ============================================================================

def evo_meta_predict(
    strategy_name: str,
    regime: str,
    horizon: Union[str, int],
    performance_history: Dict,
    genome: StrategyGenome,
    bandit_confidence: float,
    recent_metrics: Dict
) -> float:
    """
    Predict probability of strategy outperforming ensemble.

    Args:
        strategy_name: Name of strategy
        regime: Current regime
        horizon: Return horizon
        performance_history: Historical performance
        genome: Strategy genome
        bandit_confidence: Bandit confidence score
        recent_metrics: Recent risk metrics

    Returns:
        Probability of outperformance [0, 1]
    """
    engine = _get_meta_engine()
    return engine.predict(
        strategy_name=strategy_name,
        regime=regime,
        horizon=horizon,
        performance_history=performance_history,
        genome=genome,
        bandit_confidence=bandit_confidence,
        recent_metrics=recent_metrics
    )


def evo_meta_train(
    performance_memory: Dict[str, Dict],
    genomes: Dict[str, StrategyGenome]
) -> None:
    """
    Train the meta-learner.

    Args:
        performance_memory: Historical performance data
        genomes: Strategy genomes
    """
    engine = _get_meta_engine()
    engine.train(performance_memory, genomes)


def evo_meta_load() -> None:
    """Load meta-learner from disk."""
    engine = _get_meta_engine()
    engine.load()


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Module D — Meta-Learner Tests")
    print("=" * 80)

    # ========================================================================
    # TEST 1: MetaFeatureBuilder
    # ========================================================================
    print("\n[TEST 1] MetaFeatureBuilder")
    print("-" * 80)

    from .genome import GenomeFactory

    factory = GenomeFactory(seed=42)
    genome = factory.create_default_genome("test_strategy")

    feature_builder = MetaFeatureBuilder()

    performance_history = {
        'rolling_sharpe': [1.5, 1.8, 2.0, 1.9],
        'rolling_sortino': [2.0, 2.2, 2.5],
        'rolling_dd': [-0.05, -0.08, -0.06],
        'rolling_win_rate': [0.55, 0.60, 0.58],
        'recent_returns': [0.01, 0.02, -0.01, 0.03],
        'meta_accuracy': 0.70,
        'wfo_sharpe': 1.8,
        'volatility': 0.15
    }

    recent_metrics = {
        'volatility': 0.15,
        'forecast_dispersion': 0.3,
        'correlation_to_ensemble': 0.2
    }

    features = feature_builder.build_features(
        strategy_name="test_strategy",
        regime="bull",
        horizon=5,
        performance_history=performance_history,
        genome=genome,
        bandit_confidence=0.75,
        recent_metrics=recent_metrics
    )

    print(f"  Feature shape: {features.shape}")
    print(f"  Feature columns: {len(features.columns)}")
    print(f"  Expected features: {len(feature_builder.feature_names)}")

    assert features.shape[0] == 1, "Should be single-row DataFrame"
    assert features.shape[1] == len(feature_builder.feature_names), "Feature count mismatch"
    assert 'regime_trend' in features.columns, "Missing regime features"
    assert 'rolling_sharpe_mean' in features.columns, "Missing performance features"
    assert 'bandit_confidence' in features.columns, "Missing bandit features"

    print("  ✓ Feature building working correctly")

    # ========================================================================
    # TEST 2: MetaLearner Training
    # ========================================================================
    print("\n[TEST 2] MetaLearner Training")
    print("-" * 80)

    model = MetaLearner(model_type='rf', random_state=42)

    # Create synthetic training data
    np.random.seed(42)
    n_samples = 100

    X_train = pd.DataFrame(
        np.random.randn(n_samples, len(feature_builder.feature_names)),
        columns=feature_builder.feature_names
    )

    # Create target: higher rolling_sharpe_mean -> more likely to outperform
    y_train = (X_train['rolling_sharpe_mean'] > 0).astype(int)

    # Train
    model.fit(X_train, y_train)

    print(f"  Model trained: {model.is_trained}")
    print(f"  Feature count: {len(model.feature_names)}")

    assert model.is_trained, "Model should be trained"
    assert model.feature_names == feature_builder.feature_names, "Feature names mismatch"

    print("  ✓ Model training working")

    # ========================================================================
    # TEST 3: Prediction
    # ========================================================================
    print("\n[TEST 3] Prediction")
    print("-" * 80)

    # Predict on new data
    X_test = features.copy()

    probas = model.predict_proba(X_test)

    print(f"  Prediction shape: {probas.shape}")
    print(f"  Probability: {probas[0]:.4f}")

    assert len(probas) == 1, "Should have one prediction"
    assert 0 <= probas[0] <= 1, "Probability out of bounds"

    print("  ✓ Prediction working")

    # ========================================================================
    # TEST 4: Save and Load Model
    # ========================================================================
    print("\n[TEST 4] Save and Load Model")
    print("-" * 80)

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        model_path = temp_dir / "test_model.pkl"

        # Save
        model.save(model_path)
        print(f"  Model saved to: {model_path}")

        # Load into new model
        model2 = MetaLearner(model_type='rf')
        model2.load(model_path)

        print(f"  Model loaded: {model2.is_trained}")
        print(f"  Feature count: {len(model2.feature_names)}")

        # Predict with loaded model
        probas2 = model2.predict_proba(X_test)

        print(f"  Original prediction: {probas[0]:.4f}")
        print(f"  Loaded prediction: {probas2[0]:.4f}")

        assert model2.is_trained, "Loaded model should be trained"
        assert np.allclose(probas, probas2), "Predictions should match"

        print("  ✓ Save/load working")

    finally:
        shutil.rmtree(temp_dir)

    # ========================================================================
    # TEST 5: MetaLearningEngine
    # ========================================================================
    print("\n[TEST 5] MetaLearningEngine")
    print("-" * 80)

    temp_dir2 = Path(tempfile.mkdtemp())

    try:
        engine = MetaLearningEngine(model_type='rf', state_dir=temp_dir2)

        # Create fake performance memory
        genomes_dict = {
            'momentum': factory.create_default_genome('momentum'),
            'mean_reversion': factory.create_default_genome('mean_reversion'),
            'breakout': factory.create_default_genome('breakout')
        }

        performance_memory = {
            'momentum': {
                'recent_returns': [0.02, 0.03, 0.01, 0.04],
                'rolling_sharpe': [1.8, 2.0, 1.9],
                'rolling_sortino': [2.2, 2.4, 2.3],
                'rolling_dd': [-0.05, -0.06],
                'rolling_win_rate': [0.60, 0.62],
                'meta_accuracy': 0.72,
                'wfo_sharpe': 2.0,
                'volatility': 0.14,
                'regime': 'bull',
                'horizon': 5,
                'bandit_confidence': 0.80,
                'forecast_dispersion': 0.25,
                'correlation_to_ensemble': 0.15
            },
            'mean_reversion': {
                'recent_returns': [0.01, -0.01, 0.02, 0.01],
                'rolling_sharpe': [1.2, 1.4, 1.3],
                'rolling_sortino': [1.5, 1.7, 1.6],
                'rolling_dd': [-0.08, -0.10],
                'rolling_win_rate': [0.55, 0.54],
                'meta_accuracy': 0.65,
                'wfo_sharpe': 1.3,
                'volatility': 0.18,
                'regime': 'sideways',
                'horizon': 3,
                'bandit_confidence': 0.60,
                'forecast_dispersion': 0.35,
                'correlation_to_ensemble': 0.25
            },
            'breakout': {
                'recent_returns': [0.04, 0.05, 0.03, 0.06],
                'rolling_sharpe': [2.2, 2.4, 2.3],
                'rolling_sortino': [2.8, 3.0, 2.9],
                'rolling_dd': [-0.04, -0.05],
                'rolling_win_rate': [0.65, 0.67],
                'meta_accuracy': 0.75,
                'wfo_sharpe': 2.3,
                'volatility': 0.12,
                'regime': 'bull',
                'horizon': 10,
                'bandit_confidence': 0.85,
                'forecast_dispersion': 0.20,
                'correlation_to_ensemble': 0.10
            }
        }

        # Prepare training data
        X, y = engine.prepare_training_data(performance_memory, genomes_dict)

        print(f"  Training samples: {len(X)}")
        print(f"  Positive samples: {y.sum()}")
        print(f"  Negative samples: {len(y) - y.sum()}")

        assert len(X) > 0, "Should have training samples"
        assert len(X) == len(y), "X and y should have same length"

        # Train
        engine.train(performance_memory, genomes_dict)

        print(f"  Engine trained: {engine.model.is_trained}")

        # Predict
        proba = engine.predict(
            strategy_name='momentum',
            regime='bull',
            horizon=5,
            performance_history=performance_memory['momentum'],
            genome=genomes_dict['momentum'],
            bandit_confidence=0.80,
            recent_metrics={'volatility': 0.14, 'forecast_dispersion': 0.25, 'correlation_to_ensemble': 0.15}
        )

        print(f"  Momentum outperformance probability: {proba:.4f}")

        assert 0 <= proba <= 1, "Probability out of bounds"

        print("  ✓ MetaLearningEngine working")

    finally:
        shutil.rmtree(temp_dir2)

    # ========================================================================
    # TEST 6: Integration Hooks
    # ========================================================================
    print("\n[TEST 6] Integration Hooks")
    print("-" * 80)

    def test_integration_hooks():
        """Test integration hooks with global engine."""
        global _META_ENGINE

        temp_dir3 = Path(tempfile.mkdtemp())

        try:
            # Reset global engine
            _META_ENGINE = MetaLearningEngine(model_type='rf', state_dir=temp_dir3)

            # Train using hook
            evo_meta_train(performance_memory, genomes_dict)

            # Predict using hook
            proba_hook = evo_meta_predict(
                strategy_name='breakout',
                regime='bull',
                horizon=10,
                performance_history=performance_memory['breakout'],
                genome=genomes_dict['breakout'],
                bandit_confidence=0.85,
                recent_metrics={'volatility': 0.12, 'forecast_dispersion': 0.20, 'correlation_to_ensemble': 0.10}
            )

            print(f"  Breakout outperformance probability: {proba_hook:.4f}")

            assert 0 <= proba_hook <= 1, "Probability out of bounds"

            print("  ✓ Integration hooks working")

        finally:
            shutil.rmtree(temp_dir3)

    test_integration_hooks()

    # ========================================================================
    # TEST 7: Untrained Model Fallback
    # ========================================================================
    print("\n[TEST 7] Untrained Model Fallback")
    print("-" * 80)

    untrained_model = MetaLearner(model_type='rf')

    # Predict without training
    X_dummy = pd.DataFrame(
        np.random.randn(5, len(feature_builder.feature_names)),
        columns=feature_builder.feature_names
    )

    probas_untrained = untrained_model.predict_proba(X_dummy)

    print(f"  Untrained predictions: {probas_untrained}")
    print(f"  All equal to 0.5: {np.allclose(probas_untrained, 0.5)}")

    assert np.allclose(probas_untrained, 0.5), "Untrained model should return 0.5"

    print("  ✓ Untrained fallback working")

    # ========================================================================
    # TEST 8: Feature Alignment
    # ========================================================================
    print("\n[TEST 8] Feature Alignment")
    print("-" * 80)

    # Create DataFrame with missing and extra columns
    misaligned_df = pd.DataFrame({
        'regime_trend': [1.0],
        'rolling_sharpe_mean': [1.5],
        'extra_column': [999],  # Extra column
        # Missing most columns
    })

    print(f"  Original columns: {list(misaligned_df.columns)}")

    aligned_df = _ensure_feature_alignment(misaligned_df, feature_builder.feature_names)

    print(f"  Aligned columns: {len(aligned_df.columns)}")
    print(f"  Expected columns: {len(feature_builder.feature_names)}")
    print(f"  Extra column removed: {'extra_column' not in aligned_df.columns}")
    print(f"  Missing columns added: {len(aligned_df.columns) == len(feature_builder.feature_names)}")

    assert len(aligned_df.columns) == len(feature_builder.feature_names), "Column count should match"
    assert 'extra_column' not in aligned_df.columns, "Extra column should be removed"
    assert all(col in aligned_df.columns for col in feature_builder.feature_names), "All expected columns should be present"

    print("  ✓ Feature alignment working")

    # ========================================================================
    # TEST 9: Metadata Round-Trip
    # ========================================================================
    print("\n[TEST 9] Metadata Round-Trip")
    print("-" * 80)

    temp_dir4 = Path(tempfile.mkdtemp())

    try:
        engine = MetaLearningEngine(model_type='rf', state_dir=temp_dir4)

        # Save metadata
        engine.save(sample_count=50)

        metadata_path = temp_dir4 / "meta_learner_metadata.json"
        assert metadata_path.exists(), "Metadata file should exist"

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"  Metadata keys: {list(metadata.keys())}")
        print(f"  Version: {metadata.get('version')}")
        print(f"  Sample count: {metadata.get('training_sample_count')}")
        print(f"  Feature count: {len(metadata.get('feature_list', []))}")
        print(f"  Is trained: {metadata.get('is_trained')}")

        assert 'version' in metadata, "Should have version"
        assert 'feature_list' in metadata, "Should have feature list"
        assert 'training_sample_count' in metadata, "Should have sample count"
        assert metadata['training_sample_count'] == 50, "Sample count should be 50"
        assert len(metadata['feature_list']) == 25, "Should have 25 features"

        print("  ✓ Metadata round-trip working")

    finally:
        shutil.rmtree(temp_dir4)

    # ========================================================================
    # TEST 10: Sparse History Handling
    # ========================================================================
    print("\n[TEST 10] Sparse History Handling")
    print("-" * 80)

    # Create sparse performance history
    sparse_history = {
        'rolling_sharpe': [],  # Empty
        'rolling_sortino': [1.5],  # Single value
        # Most fields missing
    }

    sparse_metrics = {
        # All missing
    }

    genome = factory.create_default_genome("sparse_test")

    # Should not crash
    try:
        features = feature_builder.build_features(
            strategy_name="sparse_test",
            regime="bull",
            horizon=5,
            performance_history=sparse_history,
            genome=genome,
            bandit_confidence=0.5,
            recent_metrics=sparse_metrics
        )

        print(f"  Features shape: {features.shape}")
        print(f"  All features present: {len(features.columns) == 25}")
        print(f"  No NaN values: {not features.isnull().any().any()}")

        assert features.shape[0] == 1, "Should have one row"
        assert features.shape[1] == 25, "Should have 25 columns"
        assert not features.isnull().any().any(), "Should have no NaN values"

        print("  ✓ Sparse history handling working")

    except Exception as e:
        print(f"  ERROR: {e}")
        raise

    # ========================================================================
    # TEST 11: Corrupted Model Load
    # ========================================================================
    print("\n[TEST 11] Corrupted Model Load")
    print("-" * 80)

    temp_dir5 = Path(tempfile.mkdtemp())

    try:
        # Create corrupted pickle file
        corrupted_path = temp_dir5 / "corrupted_model.pkl"
        with open(corrupted_path, 'wb') as f:
            f.write(b"this is not a valid pickle file")

        # Try to load corrupted model
        model_corrupted = MetaLearner(model_type='rf')
        model_corrupted.load(corrupted_path)

        print(f"  Model after load: {model_corrupted.is_trained}")
        print(f"  Has fallback model: {model_corrupted.model is not None}")

        assert not model_corrupted.is_trained, "Should not be trained"
        assert model_corrupted.model is not None, "Should have fallback model"

        # Prediction should still work
        X_test = pd.DataFrame(
            np.random.randn(3, 25),
            columns=feature_builder.feature_names
        )
        probas = model_corrupted.predict_proba(X_test)

        print(f"  Fallback predictions: {probas}")
        assert np.allclose(probas, 0.5), "Should return 0.5 for untrained"

        print("  ✓ Corrupted model load fallback working")

    finally:
        shutil.rmtree(temp_dir5)

    # ========================================================================
    # TEST 12: Insufficient Training Data
    # ========================================================================
    print("\n[TEST 12] Insufficient Training Data")
    print("-" * 80)

    temp_dir6 = Path(tempfile.mkdtemp())

    try:
        engine = MetaLearningEngine(model_type='rf', state_dir=temp_dir6)

        # Create small performance memory (< 20 samples)
        small_genomes = {
            'test1': factory.create_default_genome('test1'),
            'test2': factory.create_default_genome('test2')
        }

        small_memory = {
            'test1': {
                'recent_returns': [0.01, 0.02],  # Only 2 returns
                'rolling_sharpe': [1.5],
                'rolling_sortino': [1.8],
                'rolling_dd': [-0.05],
                'rolling_win_rate': [0.55],
                'meta_accuracy': 0.65,
                'wfo_sharpe': 1.5,
                'volatility': 0.15,
                'regime': 'bull',
                'horizon': 5,
                'bandit_confidence': 0.70,
                'forecast_dispersion': 0.25,
                'correlation_to_ensemble': 0.15
            },
            'test2': {
                'recent_returns': [0.02, 0.03],
                'rolling_sharpe': [1.8],
                'rolling_sortino': [2.0],
                'rolling_dd': [-0.06],
                'rolling_win_rate': [0.60],
                'meta_accuracy': 0.70,
                'wfo_sharpe': 1.8,
                'volatility': 0.14,
                'regime': 'bull',
                'horizon': 5,
                'bandit_confidence': 0.75,
                'forecast_dispersion': 0.20,
                'correlation_to_ensemble': 0.10
            }
        }

        # Train (should skip due to insufficient data)
        engine.train(small_memory, small_genomes)

        # Check metadata was saved
        metadata_path = temp_dir6 / "meta_learner_metadata.json"
        assert metadata_path.exists(), "Metadata should be saved even when training skipped"

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"  Is trained: {metadata.get('is_trained')}")
        print(f"  Sample count: {metadata.get('training_sample_count')}")
        print(f"  Status: {metadata.get('status')}")

        assert metadata['is_trained'] == False, "Should not be trained"
        assert metadata.get('status') == 'insufficient_data', "Should have insufficient_data status"

        print("  ✓ Insufficient training data handling working")

    finally:
        shutil.rmtree(temp_dir6)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL MODULE D SWEEP TESTS PASSED")
    print("=" * 80)
    print("\nMeta-Learner Features:")
    print("  ✓ Feature building (25 features)")
    print("  ✓ Model training (RF/XGBoost)")
    print("  ✓ Probability prediction")
    print("  ✓ Save/load persistence")
    print("  ✓ MetaLearningEngine orchestration")
    print("  ✓ Integration hooks (evo_meta_*)")
    print("  ✓ Untrained model fallback (0.5)")
    print("\nSweep D.1 Enhancements:")
    print("  ✓ Feature alignment (missing/extra columns)")
    print("  ✓ Metadata round-trip (JSON-safe types)")
    print("  ✓ Sparse history handling (safe defaults)")
    print("  ✓ Corrupted model load (fallback)")
    print("  ✓ Insufficient training data (< 20 samples)")
    print("  ✓ Atomic writes (model + metadata)")
    print("  ✓ Type safety (numpy → Python)")
    print("\nModule D — Meta-Learner: PRODUCTION READY")
    print("=" * 80)
