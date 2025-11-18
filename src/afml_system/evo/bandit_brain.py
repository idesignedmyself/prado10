"""
PRADO9_EVO Module A — Bandit Brain (Hierarchical Thompson Sampling)

A three-level adaptive bandit system for continuous learning:
1. Strategy Selection Bandit - learns which strategy works in which regime
2. Hyperparameter Bandit - learns which parameter sets produce highest Sharpe
3. Regime Confidence Bandit - learns regime classifier reliability

Author: PRADO9_EVO Builder
Date: 2025-01-16
Revised: Sweep A.1
"""

import os
import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def _encode_dataclass(obj: Any) -> Dict:
    """
    Encode dataclass to JSON-serializable dict.

    Handles:
    - datetime → ISO string
    - None values
    - float preservation
    """
    if hasattr(obj, '__dataclass_fields__'):
        data = asdict(obj)
        # Convert datetime to ISO string
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    return obj


def _decode_bandit_state(data: Dict) -> 'BanditState':
    """Reconstruct BanditState from dict."""
    return BanditState(
        strategy=data['strategy'],
        regime=data['regime'],
        alpha=float(data.get('alpha', 1.0)),
        beta=float(data.get('beta', 1.0)),
        total_rewards=float(data.get('total_rewards', 0.0)),
        observations=int(data.get('observations', 0)),
        last_updated=data.get('last_updated')
    )


def _decode_hyperparam_state(data: Dict) -> 'HyperparameterBanditState':
    """Reconstruct HyperparameterBanditState from dict."""
    return HyperparameterBanditState(
        strategy=data['strategy'],
        config_id=data['config_id'],
        alpha=float(data.get('alpha', 1.0)),
        beta=float(data.get('beta', 1.0)),
        reward_sum=float(data.get('reward_sum', 0.0)),
        observations=int(data.get('observations', 0)),
        last_updated=data.get('last_updated')
    )


def _decode_regime_state(data: Dict) -> 'RegimeConfidenceState':
    """Reconstruct RegimeConfidenceState from dict."""
    return RegimeConfidenceState(
        regime=data['regime'],
        alpha=float(data.get('alpha', 1.0)),
        beta=float(data.get('beta', 1.0)),
        accuracy=float(data.get('accuracy', 0.5)),
        observations=int(data.get('observations', 0)),
        last_updated=data.get('last_updated')
    )


# ============================================================================
# BANDIT STATE DATACLASSES
# ============================================================================

@dataclass
class BanditState:
    """State for strategy-regime bandit arm."""
    strategy: str
    regime: str
    alpha: float = 1.0  # successes (minimum prior = 1)
    beta: float = 1.0   # failures (minimum prior = 1)
    total_rewards: float = 0.0
    observations: int = 0
    last_updated: Optional[str] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()
        # Enforce minimum priors
        self.alpha = max(1.0, self.alpha)
        self.beta = max(1.0, self.beta)


@dataclass
class HyperparameterBanditState:
    """State for hyperparameter configuration bandit arm."""
    strategy: str
    config_id: str
    alpha: float = 1.0
    beta: float = 1.0
    reward_sum: float = 0.0
    observations: int = 0
    last_updated: Optional[str] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()
        # Enforce minimum priors
        self.alpha = max(1.0, self.alpha)
        self.beta = max(1.0, self.beta)


@dataclass
class RegimeConfidenceState:
    """State for regime confidence tracking."""
    regime: str
    alpha: float = 1.0  # correct predictions
    beta: float = 1.0   # incorrect predictions
    accuracy: float = 0.5
    observations: int = 0
    last_updated: Optional[str] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()
        # Enforce minimum priors
        self.alpha = max(1.0, self.alpha)
        self.beta = max(1.0, self.beta)


# ============================================================================
# THOMPSON SAMPLING UTILITIES
# ============================================================================

def _safe_beta_sample(alpha: float, beta: float, epsilon: float = 1e-6) -> float:
    """
    Safe Thompson Sampling from Beta distribution.

    Prevents:
    - Beta(0, 0) failures
    - NaN/inf values
    - Division by zero

    Args:
        alpha: Success count (min 1.0)
        beta: Failure count (min 1.0)
        epsilon: Smoothing constant

    Returns:
        float: Sample in [0, 1], fallback to 0.5 on error
    """
    try:
        # Enforce minimum values with epsilon smoothing
        alpha = max(1.0, alpha + epsilon)
        beta = max(1.0, beta + epsilon)

        # Sample from Beta distribution
        sample = np.random.beta(alpha, beta)

        # Validate result
        if np.isnan(sample) or np.isinf(sample):
            return 0.5

        return float(sample)

    except Exception:
        # Fallback to uniform
        return 0.5


# ============================================================================
# LEVEL 1 — STRATEGY SELECTION BANDIT
# ============================================================================

class StrategySelectionBandit:
    """
    Thompson Sampling bandit for strategy selection per regime.

    Learns which strategies perform best in which regime through
    Beta-distributed Thompson Sampling with adaptive reward shaping.
    """

    def __init__(self):
        self.states: Dict[Tuple[str, str], BanditState] = {}

    def _get_key(self, strategy: str, regime: str) -> Tuple[str, str]:
        """Generate composite key for strategy-regime pair."""
        return (strategy, regime)

    def _ensure_state(self, strategy: str, regime: str) -> BanditState:
        """Ensure bandit state exists for strategy-regime pair."""
        key = self._get_key(strategy, regime)
        if key not in self.states:
            self.states[key] = BanditState(strategy=strategy, regime=regime)
        return self.states[key]

    def sample(self, strategy: str, regime: str) -> float:
        """
        Draw Thompson sample for strategy-regime pair.

        Returns:
            float: Beta-distributed sample in [0, 1]
        """
        state = self._ensure_state(strategy, regime)
        return _safe_beta_sample(state.alpha, state.beta)

    def update(self, strategy: str, regime: str, reward: float):
        """
        Update bandit state based on observed reward.

        Reward shaping:
        - reward > 0: alpha += max(0.1, reward)
        - reward < 0: beta += max(0.1, abs(reward))
        - reward == 0: slight exploration (alpha+=0.01, beta+=0.01)
        """
        state = self._ensure_state(strategy, regime)

        if reward > 0:
            # Positive reward increases alpha (success)
            increment = max(0.1, reward)
            state.alpha += increment
        elif reward < 0:
            # Negative reward increases beta (failure)
            increment = max(0.1, abs(reward))
            state.beta += increment
        else:
            # Zero reward: exploration pressure
            state.alpha += 0.01
            state.beta += 0.01

        # Update tracking metrics
        state.total_rewards += reward
        state.observations += 1
        state.last_updated = datetime.now().isoformat()

    def top_strategies(self, regime: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N strategies for regime based on expected value.

        Returns:
            List of (strategy, expected_value) tuples, sorted descending
        """
        regime_states = [
            (s.strategy, s.alpha / (s.alpha + s.beta))
            for s in self.states.values()
            if s.regime == regime
        ]

        if not regime_states:
            return []

        # Sort by expected value (mean of Beta distribution)
        regime_states.sort(key=lambda x: x[1], reverse=True)
        return regime_states[:n]

    def get_state(self) -> Dict[str, Dict]:
        """Export complete bandit state."""
        return {
            f"{s.strategy}_{s.regime}": _encode_dataclass(s)
            for s in self.states.values()
        }

    def save_state(self, filepath: Path):
        """Save bandit state to JSON."""
        filepath = Path(os.path.expanduser(str(filepath)))
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.get_state(), f, indent=2)

    def load_state(self, filepath: Path):
        """Load bandit state from JSON."""
        filepath = Path(os.path.expanduser(str(filepath)))

        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.states = {}
        for key, state_dict in data.items():
            state = _decode_bandit_state(state_dict)
            self.states[(state.strategy, state.regime)] = state


# ============================================================================
# LEVEL 2 — HYPERPARAMETER BANDIT
# ============================================================================

class HyperparameterBandit:
    """
    Thompson Sampling bandit for hyperparameter configuration selection.

    Learns which hyperparameter configs produce highest Sharpe for each strategy.
    """

    def __init__(self):
        self.states: Dict[Tuple[str, str], HyperparameterBanditState] = {}

    def _get_key(self, strategy: str, config_id: str) -> Tuple[str, str]:
        """Generate composite key for strategy-config pair."""
        return (strategy, config_id)

    def _ensure_state(self, strategy: str, config_id: str) -> HyperparameterBanditState:
        """Ensure bandit state exists for strategy-config pair."""
        key = self._get_key(strategy, config_id)
        if key not in self.states:
            self.states[key] = HyperparameterBanditState(
                strategy=strategy,
                config_id=config_id
            )
        return self.states[key]

    def sample(self, strategy: str, config_id: str) -> float:
        """
        Draw Thompson sample for strategy-config pair.

        Returns:
            float: Beta-distributed sample in [0, 1]
        """
        state = self._ensure_state(strategy, config_id)
        return _safe_beta_sample(state.alpha, state.beta)

    def update(self, strategy: str, config_id: str, reward: float):
        """
        Update bandit state based on WFO Sharpe or forward performance.

        Reward shaping:
        - reward > 0: alpha += max(0.1, reward)
        - reward < 0: beta += max(0.1, abs(reward))
        - reward == 0: exploration (alpha+=0.01, beta+=0.01)
        """
        state = self._ensure_state(strategy, config_id)

        if reward > 0:
            increment = max(0.1, reward)
            state.alpha += increment
        elif reward < 0:
            increment = max(0.1, abs(reward))
            state.beta += increment
        else:
            state.alpha += 0.01
            state.beta += 0.01

        # Update tracking
        state.reward_sum += reward
        state.observations += 1
        state.last_updated = datetime.now().isoformat()

    def top_configs(self, strategy: str, n: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N configs for strategy based on expected value.

        Returns:
            List of (config_id, expected_value) tuples, sorted descending
        """
        strategy_states = [
            (s.config_id, s.alpha / (s.alpha + s.beta))
            for s in self.states.values()
            if s.strategy == strategy
        ]

        if not strategy_states:
            return []

        strategy_states.sort(key=lambda x: x[1], reverse=True)
        return strategy_states[:n]

    def all_configs(self, strategy: str) -> List[str]:
        """
        Get all config IDs for a strategy.

        Returns:
            List of config_id strings
        """
        return [
            s.config_id
            for s in self.states.values()
            if s.strategy == strategy
        ]

    def get_state(self) -> Dict[str, Dict]:
        """Export complete bandit state."""
        return {
            f"{s.strategy}_{s.config_id}": _encode_dataclass(s)
            for s in self.states.values()
        }

    def save_state(self, filepath: Path):
        """Save bandit state to JSON."""
        filepath = Path(os.path.expanduser(str(filepath)))
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.get_state(), f, indent=2)

    def load_state(self, filepath: Path):
        """Load bandit state from JSON."""
        filepath = Path(os.path.expanduser(str(filepath)))

        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.states = {}
        for key, state_dict in data.items():
            state = _decode_hyperparam_state(state_dict)
            self.states[(state.strategy, state.config_id)] = state


# ============================================================================
# LEVEL 3 — REGIME CONFIDENCE BANDIT
# ============================================================================

class RegimeConfidenceBandit:
    """
    Thompson Sampling bandit for regime classifier confidence.

    Learns how reliable the regime classifier is for each regime.
    """

    def __init__(self):
        self.states: Dict[str, RegimeConfidenceState] = {}

    def _ensure_state(self, regime: str) -> RegimeConfidenceState:
        """Ensure state exists for regime."""
        if regime not in self.states:
            self.states[regime] = RegimeConfidenceState(regime=regime)
        return self.states[regime]

    def sample(self, regime: str) -> float:
        """
        Draw Thompson sample for regime confidence.

        Returns:
            float: Beta-distributed sample representing confidence
        """
        state = self._ensure_state(regime)
        return _safe_beta_sample(state.alpha, state.beta)

    def update(self, regime: str, correct: bool):
        """
        Update regime confidence based on classification accuracy.

        Args:
            regime: Regime that was predicted
            correct: True if prediction correct, False otherwise
        """
        state = self._ensure_state(regime)

        # Binary reward: correct = alpha++, incorrect = beta++
        if correct:
            state.alpha += 1.0
        else:
            state.beta += 1.0

        # Update accuracy estimate (mean of Beta distribution)
        state.observations += 1
        state.accuracy = state.alpha / (state.alpha + state.beta)
        state.last_updated = datetime.now().isoformat()

    def confidence(self, regime: str) -> float:
        """
        Get confidence score for regime.

        Returns:
            float: Expected accuracy in [0, 1], defaults to 0.5
        """
        if regime not in self.states:
            return 0.5

        state = self.states[regime]
        return state.accuracy

    def get_state(self) -> Dict[str, Dict]:
        """Export complete bandit state."""
        return {
            regime: _encode_dataclass(state)
            for regime, state in self.states.items()
        }

    def save_state(self, filepath: Path):
        """Save bandit state to JSON."""
        filepath = Path(os.path.expanduser(str(filepath)))
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.get_state(), f, indent=2)

    def load_state(self, filepath: Path):
        """Load bandit state from JSON."""
        filepath = Path(os.path.expanduser(str(filepath)))

        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.states = {}
        for regime, state_dict in data.items():
            self.states[regime] = _decode_regime_state(state_dict)


# ============================================================================
# BANDIT BRAIN — HIERARCHICAL ORCHESTRATOR
# ============================================================================

class BanditBrain:
    """
    Unified hierarchical Thompson Sampling brain.

    Combines three levels of adaptive learning:
    1. Strategy selection per regime
    2. Hyperparameter selection per strategy
    3. Regime confidence tracking

    This is PRADO9_EVO's adaptive intelligence core.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize BanditBrain.

        Args:
            state_dir: Directory for persisting bandit states
                      Defaults to ~/.prado/evo/
        """
        if state_dir is None:
            state_dir = Path.home() / ".prado" / "evo"

        # Expand tilde and ensure absolute path
        self.state_dir = Path(os.path.expanduser(str(state_dir)))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize three-level bandit system
        self.strategy_bandit = StrategySelectionBandit()
        self.hyperparam_bandit = HyperparameterBandit()
        self.regime_bandit = RegimeConfidenceBandit()

        # Load existing states if available
        self.load_all_states()

    # ========================================================================
    # SELECTION METHODS
    # ========================================================================

    def select_strategy(self, regime: str, candidates: List[str]) -> str:
        """
        Select best strategy for regime using Thompson Sampling.

        Args:
            regime: Current market regime
            candidates: List of candidate strategies

        Returns:
            str: Selected strategy name
        """
        if not candidates:
            raise ValueError("No candidate strategies provided")

        # Sample from each candidate
        samples = {
            strategy: self.strategy_bandit.sample(strategy, regime)
            for strategy in candidates
        }

        # Return strategy with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def select_hyperparameter(self, strategy: str, config_ids: List[str]) -> str:
        """
        Select best hyperparameter config for strategy.

        Args:
            strategy: Strategy name
            config_ids: List of config IDs to choose from

        Returns:
            str: Selected config_id
        """
        if not config_ids:
            raise ValueError("No config IDs provided")

        # Sample from each config
        samples = {
            config_id: self.hyperparam_bandit.sample(strategy, config_id)
            for config_id in config_ids
        }

        # Return config with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def regime_confidence(self, regime: str) -> float:
        """
        Get confidence score for regime classification.

        Args:
            regime: Regime name

        Returns:
            float: Confidence in [0, 1], defaults to 0.5
        """
        return self.regime_bandit.confidence(regime)

    # ========================================================================
    # UPDATE METHODS
    # ========================================================================

    def update_strategy_reward(self, strategy: str, regime: str, reward: float):
        """
        Update strategy performance for regime.

        Args:
            strategy: Strategy name
            regime: Regime name
            reward: Performance metric (Sharpe, return, etc.)
        """
        self.strategy_bandit.update(strategy, regime, reward)

    def update_config_reward(self, strategy: str, config_id: str, reward: float):
        """
        Update hyperparameter config performance.

        Args:
            strategy: Strategy name
            config_id: Config identifier
            reward: Performance metric (typically Sharpe)
        """
        self.hyperparam_bandit.update(strategy, config_id, reward)

    def update_regime_accuracy(self, regime: str, correct: bool):
        """
        Update regime classification accuracy.

        Args:
            regime: Regime that was predicted
            correct: Whether prediction was correct
        """
        self.regime_bandit.update(regime, correct)

    # ========================================================================
    # PERSISTENCE METHODS
    # ========================================================================

    def save_all_states(self):
        """Save all bandit states to disk."""
        self.strategy_bandit.save_state(self.state_dir / "strategy_bandit.json")
        self.hyperparam_bandit.save_state(self.state_dir / "hyperparam_bandit.json")
        self.regime_bandit.save_state(self.state_dir / "regime_bandit.json")

    def load_all_states(self):
        """Load all bandit states from disk."""
        self.strategy_bandit.load_state(self.state_dir / "strategy_bandit.json")
        self.hyperparam_bandit.load_state(self.state_dir / "hyperparam_bandit.json")
        self.regime_bandit.load_state(self.state_dir / "regime_bandit.json")

    # ========================================================================
    # ANALYTICS METHODS
    # ========================================================================

    def get_top_strategies(self, regime: str, n: int = 5) -> List[Tuple[str, float]]:
        """Get top performing strategies for regime."""
        return self.strategy_bandit.top_strategies(regime, n)

    def get_top_configs(self, strategy: str, n: int = 3) -> List[Tuple[str, float]]:
        """Get top performing configs for strategy."""
        return self.hyperparam_bandit.top_configs(strategy, n)

    def get_all_configs(self, strategy: str) -> List[str]:
        """Get all configs for strategy."""
        return self.hyperparam_bandit.all_configs(strategy)

    def get_regime_confidences(self) -> Dict[str, float]:
        """Get confidence scores for all regimes."""
        return {
            regime: state.accuracy
            for regime, state in self.regime_bandit.states.items()
        }


# ============================================================================
# INTEGRATION HOOK FUNCTIONS
# ============================================================================

# Global singleton instance
_BRAIN_INSTANCE = None  # type: Optional[BanditBrain]


def _get_brain() -> BanditBrain:
    """Get or create global BanditBrain instance."""
    global _BRAIN_INSTANCE
    if _BRAIN_INSTANCE is None:
        _BRAIN_INSTANCE = BanditBrain()
    return _BRAIN_INSTANCE


def evo_select_strategy(regime: str, strategy_list: List[str]) -> str:
    """
    EVO integration hook: Select best strategy for regime.

    Args:
        regime: Current market regime
        strategy_list: Available strategies

    Returns:
        str: Selected strategy name
    """
    brain = _get_brain()
    return brain.select_strategy(regime, strategy_list)


def evo_update_strategy(strategy: str, regime: str, reward: float):
    """
    EVO integration hook: Update strategy performance.

    Args:
        strategy: Strategy name
        regime: Regime name
        reward: Performance reward
    """
    brain = _get_brain()
    brain.update_strategy_reward(strategy, regime, reward)
    brain.save_all_states()  # Persist after each update


def evo_select_config(strategy: str, config_list: List[str]) -> str:
    """
    EVO integration hook: Select best hyperparameter config.

    Args:
        strategy: Strategy name
        config_list: Available config IDs

    Returns:
        str: Selected config ID
    """
    brain = _get_brain()
    return brain.select_hyperparameter(strategy, config_list)


def evo_update_config(strategy: str, config_id: str, reward: float):
    """
    EVO integration hook: Update config performance.

    Args:
        strategy: Strategy name
        config_id: Config identifier
        reward: Performance reward (typically Sharpe)
    """
    brain = _get_brain()
    brain.update_config_reward(strategy, config_id, reward)
    brain.save_all_states()


def evo_regime_confidence(regime: str) -> float:
    """
    EVO integration hook: Get regime classification confidence.

    Args:
        regime: Regime name

    Returns:
        float: Confidence score in [0, 1]
    """
    brain = _get_brain()
    return brain.regime_confidence(regime)


def evo_update_regime(regime: str, correct: bool):
    """
    EVO integration hook: Update regime accuracy.

    Args:
        regime: Regime that was predicted
        correct: Whether prediction was correct
    """
    brain = _get_brain()
    brain.update_regime_accuracy(regime, correct)
    brain.save_all_states()


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Module A — Bandit Brain Tests (Sweep A.1)")
    print("=" * 80)

    # ========================================================================
    # TEST 1: Thompson Sampling Stability
    # ========================================================================
    print("\n[TEST 1] Thompson Sampling Stability")
    print("-" * 80)

    # Test safe beta sampling
    print("\nTesting _safe_beta_sample():")

    # Normal case
    sample = _safe_beta_sample(5.0, 2.0)
    print(f"  Beta(5, 2) sample: {sample:.4f}")
    assert 0.0 <= sample <= 1.0, "Sample out of bounds"

    # Edge case: minimum priors
    sample = _safe_beta_sample(1.0, 1.0)
    print(f"  Beta(1, 1) sample: {sample:.4f}")
    assert 0.0 <= sample <= 1.0, "Sample out of bounds"

    # Edge case: very small values (should enforce minimum)
    sample = _safe_beta_sample(0.001, 0.001)
    print(f"  Beta(0.001, 0.001) sample (enforced min): {sample:.4f}")
    assert 0.0 <= sample <= 1.0, "Sample out of bounds"

    # Stability test: 1000 samples should all be valid
    samples = [_safe_beta_sample(2.0, 3.0) for _ in range(1000)]
    assert all(0.0 <= s <= 1.0 for s in samples), "Invalid samples detected"
    assert not any(np.isnan(s) or np.isinf(s) for s in samples), "NaN/inf detected"
    print(f"  1000 samples: mean={np.mean(samples):.4f}, std={np.std(samples):.4f}")

    print("  ✓ Thompson Sampling stable")

    # ========================================================================
    # TEST 2: Reward Shaping
    # ========================================================================
    print("\n[TEST 2] Reward Shaping Logic")
    print("-" * 80)

    strat_bandit = StrategySelectionBandit()
    strategy = 'momentum'
    regime = 'trending'

    # Test positive reward
    initial_alpha = strat_bandit._ensure_state(strategy, regime).alpha
    strat_bandit.update(strategy, regime, 1.5)
    new_alpha = strat_bandit._ensure_state(strategy, regime).alpha
    print(f"  Positive reward (1.5): alpha {initial_alpha:.2f} → {new_alpha:.2f}")
    assert new_alpha > initial_alpha, "Alpha should increase"

    # Test negative reward
    initial_beta = strat_bandit._ensure_state(strategy, regime).beta
    strat_bandit.update(strategy, regime, -0.8)
    new_beta = strat_bandit._ensure_state(strategy, regime).beta
    print(f"  Negative reward (-0.8): beta {initial_beta:.2f} → {new_beta:.2f}")
    assert new_beta > initial_beta, "Beta should increase"

    # Test zero reward
    state_before = strat_bandit._ensure_state(strategy, regime)
    alpha_before, beta_before = state_before.alpha, state_before.beta
    strat_bandit.update(strategy, regime, 0.0)
    state_after = strat_bandit._ensure_state(strategy, regime)
    print(f"  Zero reward: alpha {alpha_before:.2f} → {state_after.alpha:.2f}")
    print(f"                beta {beta_before:.2f} → {state_after.beta:.2f}")
    assert state_after.alpha > alpha_before, "Alpha should increase slightly"
    assert state_after.beta > beta_before, "Beta should increase slightly"

    print("  ✓ Reward shaping working correctly")

    # ========================================================================
    # TEST 3: State Persistence
    # ========================================================================
    print("\n[TEST 3] State Persistence & Reload")
    print("-" * 80)

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create brain and train
        brain1 = BanditBrain(state_dir=temp_dir)

        # Add some learning
        for i in range(10):
            brain1.update_strategy_reward('momentum', 'trending', 1.0)
            brain1.update_strategy_reward('mean_reversion', 'trending', -0.5)
            brain1.update_regime_accuracy('trending', i < 8)  # 80% accuracy

        brain1.save_all_states()

        # Check files exist
        assert (temp_dir / "strategy_bandit.json").exists(), "Strategy state not saved"
        assert (temp_dir / "regime_bandit.json").exists(), "Regime state not saved"
        print(f"  State files saved to {temp_dir}")

        # Create new brain and load
        brain2 = BanditBrain(state_dir=temp_dir)

        # Verify state loaded correctly
        top_strats = brain2.get_top_strategies('trending', n=2)
        print(f"  Loaded top strategies: {top_strats}")
        assert len(top_strats) > 0, "No strategies loaded"
        assert top_strats[0][0] == 'momentum', "Top strategy should be momentum"

        conf = brain2.regime_confidence('trending')
        print(f"  Loaded regime confidence: {conf:.3f}")
        assert 0.7 < conf < 0.9, "Regime confidence not preserved"

        print("  ✓ State persistence working")

    finally:
        shutil.rmtree(temp_dir)

    # ========================================================================
    # TEST 4: Zero → Non-Zero Transitions
    # ========================================================================
    print("\n[TEST 4] Zero → Non-Zero Transitions")
    print("-" * 80)

    hp_bandit = HyperparameterBandit()

    # Start with fresh state
    config_id = 'config_test'
    strategy = 'breakout'

    # Initial state (untouched)
    sample1 = hp_bandit.sample(strategy, config_id)
    print(f"  Initial sample (no data): {sample1:.4f}")

    # Add one positive observation
    hp_bandit.update(strategy, config_id, 1.2)
    sample2 = hp_bandit.sample(strategy, config_id)
    print(f"  After positive reward: {sample2:.4f}")

    # Add negative observation
    hp_bandit.update(strategy, config_id, -0.8)
    sample3 = hp_bandit.sample(strategy, config_id)
    print(f"  After negative reward: {sample3:.4f}")

    # Verify all samples valid
    assert all(0.0 <= s <= 1.0 for s in [sample1, sample2, sample3]), "Invalid samples"
    print("  ✓ Zero → non-zero transitions stable")

    # ========================================================================
    # TEST 5: Integration Hooks Validation
    # ========================================================================
    print("\n[TEST 5] Integration Hooks Validation")
    print("-" * 80)

    def test_hooks():
        """Test integration hooks with proper scoping."""
        global _BRAIN_INSTANCE

        temp_dir2 = Path(tempfile.mkdtemp())
        _BRAIN_INSTANCE = BanditBrain(state_dir=temp_dir2)

        try:
            regime = 'ranging'
            strategies = ['mean_reversion', 'pairs', 'scalping']

            # Test strategy selection
            selected = evo_select_strategy(regime, strategies)
            print(f"  Selected strategy: {selected}")
            assert selected in strategies, "Invalid selection"

            # Test strategy update
            evo_update_strategy(selected, regime, 1.5)

            # Test config selection
            configs = ['cfg_a', 'cfg_b', 'cfg_c']
            selected_cfg = evo_select_config(selected, configs)
            print(f"  Selected config: {selected_cfg}")
            assert selected_cfg in configs, "Invalid config selection"

            # Test config update
            evo_update_config(selected, selected_cfg, 2.0)

            # Test regime confidence
            evo_update_regime(regime, True)
            evo_update_regime(regime, True)
            evo_update_regime(regime, False)
            conf = evo_regime_confidence(regime)
            print(f"  Regime confidence: {conf:.3f}")
            assert 0.0 <= conf <= 1.0, "Invalid confidence"

            # Verify state files created
            assert (temp_dir2 / "strategy_bandit.json").exists(), "State not saved"

            print("  ✓ All integration hooks working")

        finally:
            shutil.rmtree(temp_dir2)
            _BRAIN_INSTANCE = None

    test_hooks()

    # ========================================================================
    # TEST 6: Regime Confidence Logic
    # ========================================================================
    print("\n[TEST 6] Regime Confidence Logic")
    print("-" * 80)

    regime_bandit = RegimeConfidenceBandit()

    # Test: 7 correct, 3 incorrect → 70% accuracy
    for i in range(10):
        regime_bandit.update('test_regime', i < 7)

    conf = regime_bandit.confidence('test_regime')
    print(f"  7/10 correct → confidence: {conf:.3f}")
    assert 0.6 < conf < 0.8, "Confidence calculation incorrect"

    # Test: untouched regime → 0.5 default
    conf_unknown = regime_bandit.confidence('unknown_regime')
    print(f"  Unknown regime → confidence: {conf_unknown:.3f}")
    assert conf_unknown == 0.5, "Default confidence should be 0.5"

    print("  ✓ Regime confidence logic correct")

    # ========================================================================
    # TEST 7: HyperparameterBandit all_configs()
    # ========================================================================
    print("\n[TEST 7] HyperparameterBandit.all_configs()")
    print("-" * 80)

    hp_bandit2 = HyperparameterBandit()
    strat = 'momentum'

    # Add some configs
    for cfg in ['cfg_1', 'cfg_2', 'cfg_3']:
        hp_bandit2.update(strat, cfg, 1.0)

    all_cfgs = hp_bandit2.all_configs(strat)
    print(f"  All configs for {strat}: {all_cfgs}")
    assert len(all_cfgs) == 3, "Should have 3 configs"
    assert 'cfg_1' in all_cfgs, "cfg_1 missing"

    print("  ✓ all_configs() working")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL SWEEP A.1 TESTS PASSED")
    print("=" * 80)
    print("\nFixes Applied:")
    print("  ✓ Path handling with tilde expansion")
    print("  ✓ JSON serialization/deserialization")
    print("  ✓ Thompson Sampling stability (no NaN/inf)")
    print("  ✓ Enhanced reward shaping (positive/negative/zero)")
    print("  ✓ HyperparameterBandit all_configs() method")
    print("  ✓ RegimeConfidenceBandit accuracy logic")
    print("  ✓ Integration hooks validated")
    print("  ✓ Type safety and imports")
    print("  ✓ Comprehensive inline testing")
    print("\nModule A — Bandit Brain: PRODUCTION READY")
    print("=" * 80)
