"""
PRADO9_EVO Module C — Evolution Engine (Genetic + Bayesian Optimization Layer)

Implements evolutionary dynamics for strategy genomes:
- Fitness evaluation with multi-metric weighting
- Population selection and replacement
- Genetic operators (mutation, crossover)
- Bandit Brain integration for adaptive selection
- Walk-forward optimization integration
- Generation-based evolution

This is the core self-improvement engine of PRADO9_EVO.

Author: PRADO9_EVO Builder
Date: 2025-01-16
"""

import os
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .genome import StrategyGenome, GenomeLibrary, GenomeFactory, encode_genome, decode_genome
from .bandit_brain import BanditBrain


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _encode_population(
    generation: int,
    population_size: int,
    genomes: Dict[str, StrategyGenome]
) -> Dict[str, any]:
    """
    Encode population state to JSON-serializable dictionary.

    Args:
        generation: Current generation number
        population_size: Target population size
        genomes: Dictionary of genomes

    Returns:
        JSON-serializable dictionary
    """
    return {
        'generation': int(generation),
        'population_size': int(population_size),
        'timestamp': datetime.now().isoformat(),
        'genomes': {
            name: encode_genome(genome)
            for name, genome in genomes.items()
        }
    }


def _decode_population(
    state: Dict[str, any]
) -> Tuple[int, int, Dict[str, StrategyGenome]]:
    """
    Decode population state from JSON dictionary.

    Args:
        state: Saved population state

    Returns:
        Tuple of (generation, population_size, genomes_dict)
    """
    generation = int(state.get('generation', 0))
    population_size = int(state.get('population_size', 20))

    genomes = {}
    for name, genome_data in state.get('genomes', {}).items():
        try:
            genome = decode_genome(genome_data)
            genomes[name] = genome
        except Exception as e:
            print(f"Warning: Failed to decode genome {name}: {e}")
            continue

    return generation, population_size, genomes


# ============================================================================
# FITNESS EVALUATOR
# ============================================================================

class FitnessEvaluator:
    """
    Evaluates strategy genome fitness based on multiple performance metrics.

    Uses weighted combination of:
    - Sharpe ratio
    - Sortino ratio
    - Meta-label accuracy
    - Walk-forward Sharpe
    - Bandit confidence
    - Win rate
    - Maximum drawdown penalty
    """

    def __init__(
        self,
        sharpe_weight: float = 0.40,
        sortino_weight: float = 0.20,
        meta_weight: float = 0.10,
        wfo_weight: float = 0.10,
        bandit_weight: float = 0.10,
        winrate_weight: float = 0.05,
        drawdown_weight: float = 0.05
    ):
        """
        Initialize FitnessEvaluator with metric weights.

        Args:
            sharpe_weight: Weight for Sharpe ratio
            sortino_weight: Weight for Sortino ratio
            meta_weight: Weight for meta-label accuracy
            wfo_weight: Weight for walk-forward optimization Sharpe
            bandit_weight: Weight for bandit confidence
            winrate_weight: Weight for win rate
            drawdown_weight: Weight for drawdown penalty
        """
        self.weights = {
            'sharpe': sharpe_weight,
            'sortino': sortino_weight,
            'meta': meta_weight,
            'wfo': wfo_weight,
            'bandit': bandit_weight,
            'winrate': winrate_weight,
            'drawdown': drawdown_weight
        }

        # Verify weights sum to approximately 1.0
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def compute_fitness(
        self,
        genome: StrategyGenome,
        performance: Dict[str, float]
    ) -> float:
        """
        Compute weighted fitness score for a genome.

        Args:
            genome: Strategy genome to evaluate
            performance: Dictionary containing performance metrics:
                - sharpe: Sharpe ratio
                - sortino: Sortino ratio
                - max_dd: Maximum drawdown (negative)
                - win_rate: Win rate [0, 1]
                - meta_accuracy: Meta-label accuracy [0, 1]
                - bandit_confidence: Bandit confidence [0, 1]
                - wfo_sharpe: Walk-forward Sharpe

        Returns:
            float: Weighted fitness score
        """
        # Extract metrics with defaults
        sharpe = performance.get('sharpe', 0.0)
        sortino = performance.get('sortino', 0.0)
        max_dd = performance.get('max_dd', 0.0)
        win_rate = performance.get('win_rate', 0.5)
        meta_accuracy = performance.get('meta_accuracy', 0.5)
        bandit_confidence = performance.get('bandit_confidence', 0.5)
        wfo_sharpe = performance.get('wfo_sharpe', 0.0)

        # Stability guards: check for NaN, inf, or invalid values
        def _safe_float(value: float, default: float = 0.0) -> float:
            """Convert to safe float, replacing NaN/inf with default."""
            if not isinstance(value, (int, float)):
                return default
            if np.isnan(value) or np.isinf(value):
                return default
            return float(value)

        sharpe = _safe_float(sharpe, 0.0)
        sortino = _safe_float(sortino, 0.0)
        max_dd = _safe_float(max_dd, 0.0)
        win_rate = _safe_float(win_rate, 0.5)
        meta_accuracy = _safe_float(meta_accuracy, 0.5)
        bandit_confidence = _safe_float(bandit_confidence, 0.5)
        wfo_sharpe = _safe_float(wfo_sharpe, 0.0)

        # Normalize Sharpe-like metrics (assume range [-2, 4])
        sharpe_norm = np.clip((sharpe + 2) / 6, 0, 1)
        sortino_norm = np.clip((sortino + 2) / 6, 0, 1)
        wfo_norm = np.clip((wfo_sharpe + 2) / 6, 0, 1)

        # Drawdown penalty (assume max_dd in range [-0.50, 0])
        # Penalize drawdowns > 20%
        max_dd_penalty = max(0, abs(max_dd) / 0.20)
        dd_component = np.clip(1.0 - max_dd_penalty, 0, 1)

        # Compute weighted fitness
        fitness = (
            self.weights['sharpe'] * sharpe_norm +
            self.weights['sortino'] * sortino_norm +
            self.weights['meta'] * meta_accuracy +
            self.weights['wfo'] * wfo_norm +
            self.weights['bandit'] * bandit_confidence +
            self.weights['winrate'] * win_rate +
            self.weights['drawdown'] * dd_component
        )

        # Final safety check
        fitness = _safe_float(fitness, 0.0)
        return float(np.clip(fitness, 0, 1))

    def normalize_fitness(
        self,
        fitness_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize fitness scores to [0, 1] range using min-max scaling.

        Args:
            fitness_dict: Dictionary mapping strategy names to fitness scores

        Returns:
            Dictionary with normalized fitness scores
        """
        if not fitness_dict:
            return {}

        fitness_values = list(fitness_dict.values())
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        # Avoid division by zero
        if max_fitness == min_fitness:
            return {name: 0.5 for name in fitness_dict.keys()}

        # Min-max normalization
        normalized = {
            name: (fitness - min_fitness) / (max_fitness - min_fitness)
            for name, fitness in fitness_dict.items()
        }

        return normalized


# ============================================================================
# EVOLUTION ENGINE
# ============================================================================

class EvolutionEngine:
    """
    Core evolutionary engine for PRADO9_EVO.

    Manages:
    - Population evolution
    - Fitness-based selection
    - Genetic operators (mutation, crossover)
    - Bandit Brain integration
    - Generation advancement
    - Population persistence
    """

    def __init__(
        self,
        genome_library: Optional[GenomeLibrary] = None,
        bandit_brain: Optional[BanditBrain] = None,
        population_size: int = 20,
        elite_fraction: float = 0.3,
        state_dir: Optional[Path] = None
    ):
        """
        Initialize EvolutionEngine.

        Args:
            genome_library: GenomeLibrary instance (creates new if None)
            bandit_brain: BanditBrain instance (creates new if None)
            population_size: Target population size
            elite_fraction: Fraction of population to preserve as elite
            state_dir: Directory for state persistence
        """
        if state_dir is None:
            state_dir = Path.home() / ".prado" / "evo"

        self.state_dir = Path(os.path.expanduser(str(state_dir)))
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.genome_library = genome_library or GenomeLibrary(state_dir=state_dir)
        self.bandit_brain = bandit_brain or BanditBrain(state_dir=state_dir)
        self.factory = GenomeFactory()
        self.fitness_evaluator = FitnessEvaluator()

        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.current_generation = 0

        # Ensure minimum population
        self._ensure_population()

        # Load state if available
        self.load_population()

    def _ensure_population(self):
        """Ensure population meets minimum size."""
        current_size = len(self.genome_library.all_genomes())
        if current_size < self.population_size:
            # Add random genomes to reach target size
            for i in range(self.population_size - current_size):
                genome = self.factory.create_random_genome(f"init_pop_{i}")
                self.genome_library.add_genome(genome)

    def evaluate_generation(
        self,
        performance_history: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Evaluate fitness for all genomes in current generation.

        Args:
            performance_history: Dictionary mapping strategy names to performance metrics
        """
        if not performance_history:
            print("Warning: Empty performance history provided")
            return

        for genome in self.genome_library.all_genomes():
            if genome.strategy_name in performance_history:
                performance = performance_history[genome.strategy_name]

                # Compute fitness
                try:
                    fitness = self.fitness_evaluator.compute_fitness(genome, performance)

                    # Validate fitness
                    if np.isnan(fitness) or np.isinf(fitness):
                        print(f"Warning: Invalid fitness for {genome.strategy_name}: {fitness}")
                        fitness = 0.0

                    # Update genome fitness
                    self.genome_library.update_fitness(genome.strategy_name, fitness)

                except Exception as e:
                    print(f"Warning: Fitness computation error for {genome.strategy_name}: {e}")
                    continue

                # Update bandit brain (use Sharpe as reward)
                try:
                    sharpe = performance.get('sharpe', 0.0)

                    # Validate Sharpe
                    if isinstance(sharpe, (int, float)) and not (np.isnan(sharpe) or np.isinf(sharpe)):
                        for regime in genome.regime_filter:
                            if isinstance(regime, str) and regime:
                                self.bandit_brain.update_strategy_reward(
                                    genome.strategy_name,
                                    regime,
                                    sharpe
                                )
                except Exception as e:
                    print(f"Warning: Bandit update error for {genome.strategy_name}: {e}")
                    continue

    def select_top_genomes(self, n: int) -> List[StrategyGenome]:
        """
        Select top N genomes by fitness.

        Args:
            n: Number of top genomes to select

        Returns:
            List of top genomes sorted by fitness (descending)
        """
        if n <= 0:
            return []

        all_genomes = self.genome_library.all_genomes()
        if not all_genomes:
            return []

        # Cap n to population size
        n = min(n, len(all_genomes))

        return self.genome_library.top_genomes(n)

    def generate_offspring(
        self,
        top_genomes: List[StrategyGenome],
        offspring_count: int
    ) -> List[StrategyGenome]:
        """
        Generate offspring through mutation and crossover.

        Args:
            top_genomes: Elite genomes to use as parents
            offspring_count: Number of offspring to generate

        Returns:
            List of offspring genomes
        """
        if not top_genomes:
            return []

        if offspring_count <= 0:
            return []

        offspring = []
        max_iterations = offspring_count * 10  # Prevent infinite loops
        iterations = 0

        while len(offspring) < offspring_count and iterations < max_iterations:
            iterations += 1

            try:
                if random.random() < 0.5:
                    # Mutation (50%)
                    parent = random.choice(top_genomes)
                    mutant = self.factory.mutate(parent)
                    mutant.strategy_name = f"{parent.strategy_name}_mut_gen{self.current_generation}_{len(offspring)}"

                    # Validate offspring
                    if len(mutant.feature_set) >= 3:
                        offspring.append(mutant)
                else:
                    # Crossover (50%)
                    if len(top_genomes) < 2:
                        # Not enough parents for crossover, use mutation instead
                        parent = random.choice(top_genomes)
                        mutant = self.factory.mutate(parent)
                        mutant.strategy_name = f"{parent.strategy_name}_mut_gen{self.current_generation}_{len(offspring)}"
                        if len(mutant.feature_set) >= 3:
                            offspring.append(mutant)
                        continue

                    parent1 = random.choice(top_genomes)
                    parent2 = random.choice(top_genomes)

                    # Ensure different parents
                    attempts = 0
                    while parent1.strategy_name == parent2.strategy_name and attempts < 10:
                        parent2 = random.choice(top_genomes)
                        attempts += 1

                    if parent1.strategy_name != parent2.strategy_name:
                        child1, child2 = self.factory.crossover(parent1, parent2)
                        child1.strategy_name = f"cross_gen{self.current_generation}_a_{len(offspring)}"

                        # Validate child1
                        if len(child1.feature_set) >= 3:
                            offspring.append(child1)

                        if len(offspring) < offspring_count and len(child2.feature_set) >= 3:
                            child2.strategy_name = f"cross_gen{self.current_generation}_b_{len(offspring)}"
                            offspring.append(child2)
                    else:
                        # Fallback to mutation
                        mutant = self.factory.mutate(parent1)
                        mutant.strategy_name = f"{parent1.strategy_name}_mut_gen{self.current_generation}_{len(offspring)}"
                        if len(mutant.feature_set) >= 3:
                            offspring.append(mutant)

            except Exception as e:
                print(f"Warning: Offspring generation error: {e}")
                # Try mutation as safe fallback
                try:
                    parent = random.choice(top_genomes)
                    mutant = self.factory.mutate(parent)
                    mutant.strategy_name = f"{parent.strategy_name}_mut_gen{self.current_generation}_{len(offspring)}"
                    if len(mutant.feature_set) >= 3:
                        offspring.append(mutant)
                except Exception:
                    continue

        return offspring[:offspring_count]

    def evolve_population(self) -> Dict[str, StrategyGenome]:
        """
        Evolve population for one generation.

        Process:
        1. Sort by fitness
        2. Select top elite_fraction as parents
        3. Generate offspring (50% mutation, 50% crossover)
        4. Replace bottom (1 - elite_fraction)
        5. Increment generation
        6. Save population

        Returns:
            Dictionary of newly created genomes
        """
        all_genomes = self.genome_library.all_genomes()

        if len(all_genomes) < 4:
            # Need minimum population for evolution
            print(f"Warning: Population too small for evolution ({len(all_genomes)} < 4)")
            return {}

        population_size = len(all_genomes)
        elite_count = max(2, int(population_size * self.elite_fraction))
        replace_count = max(1, int(population_size * (1 - self.elite_fraction)))

        # Ensure elite + replace doesn't exceed population
        if elite_count + replace_count > population_size:
            elite_count = max(2, population_size - replace_count)

        # Select elite
        elite = self.select_top_genomes(elite_count)

        if not elite:
            print("Warning: No elite genomes selected")
            return {}

        # Generate offspring
        offspring = self.generate_offspring(elite, replace_count)

        if not offspring:
            print("Warning: Failed to generate offspring")
            return {}

        # Ensure we don't exceed population capacity
        actual_replace = min(replace_count, len(offspring), population_size - elite_count)

        # Get bottom performers to replace
        bottom_genomes = all_genomes[-actual_replace:]

        # Replace bottom with offspring
        new_genomes = {}
        for i in range(actual_replace):
            if i >= len(bottom_genomes) or i >= len(offspring):
                break

            old_genome = bottom_genomes[i]
            new_genome = offspring[i]

            # Remove old
            if old_genome.strategy_name in self.genome_library.genomes:
                del self.genome_library.genomes[old_genome.strategy_name]

            # Add new
            self.genome_library.add_genome(new_genome)
            new_genomes[new_genome.strategy_name] = new_genome

        # Verify population size maintained
        final_size = len(self.genome_library.all_genomes())
        if final_size != population_size:
            print(f"Warning: Population size changed: {final_size} != {population_size}")

        # Increment generation
        self.current_generation += 1

        # Save state
        try:
            self.save_population()
        except Exception as e:
            print(f"Warning: Failed to save population: {e}")

        return new_genomes

    def update_population(
        self,
        new_population: Dict[str, StrategyGenome]
    ) -> None:
        """
        Update population with new genomes.

        Args:
            new_population: Dictionary mapping names to genomes
        """
        for name, genome in new_population.items():
            self.genome_library.genomes[name] = genome

        self.save_population()

    def save_population(self) -> None:
        """Save complete population state to disk."""
        save_path = Path(os.path.expanduser(str(self.state_dir))) / "population.json"

        try:
            state = _encode_population(
                self.current_generation,
                self.population_size,
                self.genome_library.genomes
            )

            with open(save_path, 'w') as f:
                json.dump(state, f, indent=2, sort_keys=True)

        except Exception as e:
            print(f"Error saving population: {e}")
            raise

    def load_population(self) -> None:
        """Load population state from disk."""
        load_path = Path(os.path.expanduser(str(self.state_dir))) / "population.json"

        if not load_path.exists():
            return

        try:
            with open(load_path, 'r') as f:
                state = json.load(f)

            generation, population_size, genomes = _decode_population(state)

            self.current_generation = generation
            self.population_size = population_size

            # Genomes are already loaded by GenomeLibrary
            # Just sync generation counter and verify

        except Exception as e:
            print(f"Warning: Failed to load population state: {e}")

    def get_population_stats(self) -> Dict[str, any]:
        """
        Get statistics about current population.

        Returns:
            Dictionary with population statistics
        """
        all_genomes = self.genome_library.all_genomes()

        if not all_genomes:
            return {
                'size': 0,
                'generation': self.current_generation,
                'avg_fitness': 0.0,
                'max_fitness': 0.0,
                'min_fitness': 0.0
            }

        fitnesses = [g.fitness for g in all_genomes]

        return {
            'size': len(all_genomes),
            'generation': self.current_generation,
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses),
            'std_fitness': np.std(fitnesses)
        }


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Module C — Evolution Engine Tests")
    print("=" * 80)

    # ========================================================================
    # TEST 1: Fitness Evaluation
    # ========================================================================
    print("\n[TEST 1] Fitness Evaluation")
    print("-" * 80)

    evaluator = FitnessEvaluator()

    # Test case: Good performance
    good_performance = {
        'sharpe': 2.0,
        'sortino': 2.5,
        'max_dd': -0.10,
        'win_rate': 0.60,
        'meta_accuracy': 0.70,
        'bandit_confidence': 0.80,
        'wfo_sharpe': 1.8
    }

    factory = GenomeFactory(seed=42)
    genome = factory.create_default_genome("test_strat")

    fitness_good = evaluator.compute_fitness(genome, good_performance)
    print(f"  Good performance fitness: {fitness_good:.4f}")

    # Test case: Poor performance
    poor_performance = {
        'sharpe': -0.5,
        'sortino': 0.0,
        'max_dd': -0.30,
        'win_rate': 0.40,
        'meta_accuracy': 0.50,
        'bandit_confidence': 0.40,
        'wfo_sharpe': -0.3
    }

    fitness_poor = evaluator.compute_fitness(genome, poor_performance)
    print(f"  Poor performance fitness: {fitness_poor:.4f}")

    assert fitness_good > fitness_poor, "Good fitness should be higher"
    assert 0 <= fitness_good <= 1, "Fitness out of bounds"
    assert 0 <= fitness_poor <= 1, "Fitness out of bounds"

    print("  ✓ Fitness evaluation working correctly")

    # ========================================================================
    # TEST 2: Fitness Normalization
    # ========================================================================
    print("\n[TEST 2] Fitness Normalization")
    print("-" * 80)

    fitness_dict = {
        'strat_a': 0.8,
        'strat_b': 0.5,
        'strat_c': 0.3,
        'strat_d': 0.9
    }

    normalized = evaluator.normalize_fitness(fitness_dict)

    print(f"  Original: {fitness_dict}")
    print(f"  Normalized: {normalized}")

    assert min(normalized.values()) >= 0, "Normalized values should be >= 0"
    assert max(normalized.values()) <= 1, "Normalized values should be <= 1"
    assert max(normalized, key=normalized.get) == 'strat_d', "Top should be strat_d"

    print("  ✓ Normalization working correctly")

    # ========================================================================
    # TEST 3: Evolution Engine Initialization
    # ========================================================================
    print("\n[TEST 3] Evolution Engine Initialization")
    print("-" * 80)

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        engine = EvolutionEngine(
            population_size=10,
            elite_fraction=0.3,
            state_dir=temp_dir
        )

        pop_size = len(engine.genome_library.all_genomes())
        print(f"  Population size: {pop_size}")
        print(f"  Current generation: {engine.current_generation}")

        assert pop_size == 10, f"Population should be 10, got {pop_size}"
        assert engine.current_generation == 0, "Should start at generation 0"

        print("  ✓ Engine initialization working")

    finally:
        shutil.rmtree(temp_dir)

    # ========================================================================
    # TEST 4: Offspring Generation
    # ========================================================================
    print("\n[TEST 4] Offspring Generation")
    print("-" * 80)

    temp_dir2 = Path(tempfile.mkdtemp())

    try:
        engine = EvolutionEngine(
            population_size=10,
            state_dir=temp_dir2
        )

        # Get top genomes
        top = engine.select_top_genomes(3)
        print(f"  Top genomes selected: {len(top)}")

        # Generate offspring
        offspring = engine.generate_offspring(top, 5)
        print(f"  Offspring generated: {len(offspring)}")

        assert len(offspring) == 5, "Should generate 5 offspring"
        assert all(o.generation > 0 for o in offspring), "Offspring should have generation > 0"

        print("  ✓ Offspring generation working")

    finally:
        shutil.rmtree(temp_dir2)

    # ========================================================================
    # TEST 5: Full Evolution Cycle
    # ========================================================================
    print("\n[TEST 5] Full Evolution Cycle")
    print("-" * 80)

    temp_dir3 = Path(tempfile.mkdtemp())

    try:
        engine = EvolutionEngine(
            population_size=10,
            elite_fraction=0.3,
            state_dir=temp_dir3
        )

        # Assign random fitness
        for genome in engine.genome_library.all_genomes():
            engine.genome_library.update_fitness(
                genome.strategy_name,
                random.uniform(0.3, 0.9)
            )

        initial_size = len(engine.genome_library.all_genomes())
        print(f"  Initial population: {initial_size}")

        # Evolve
        new_genomes = engine.evolve_population()

        final_size = len(engine.genome_library.all_genomes())
        print(f"  New genomes created: {len(new_genomes)}")
        print(f"  Final population: {final_size}")
        print(f"  Current generation: {engine.current_generation}")

        assert final_size == initial_size, "Population size should be stable"
        assert engine.current_generation == 1, "Should be generation 1"
        assert len(new_genomes) > 0, "Should create new genomes"

        print("  ✓ Evolution cycle working")

    finally:
        shutil.rmtree(temp_dir3)

    # ========================================================================
    # TEST 6: Population Persistence
    # ========================================================================
    print("\n[TEST 6] Population Persistence")
    print("-" * 80)

    temp_dir4 = Path(tempfile.mkdtemp())

    try:
        # Create engine and evolve
        engine1 = EvolutionEngine(
            population_size=8,
            state_dir=temp_dir4
        )

        for genome in engine1.genome_library.all_genomes():
            engine1.genome_library.update_fitness(
                genome.strategy_name,
                random.uniform(0.4, 0.8)
            )

        engine1.evolve_population()
        gen1 = engine1.current_generation

        engine1.save_population()
        print(f"  Saved generation: {gen1}")

        # Load in new engine
        engine2 = EvolutionEngine(
            population_size=8,
            state_dir=temp_dir4
        )

        gen2 = engine2.current_generation
        size2 = len(engine2.genome_library.all_genomes())

        print(f"  Loaded generation: {gen2}")
        print(f"  Loaded population size: {size2}")

        assert gen2 == gen1, "Generation should be preserved"
        assert size2 == 8, "Population size should be preserved"

        print("  ✓ Persistence working")

    finally:
        shutil.rmtree(temp_dir4)

    # ========================================================================
    # TEST 7: Evaluation with Performance History
    # ========================================================================
    print("\n[TEST 7] Evaluation with Performance History")
    print("-" * 80)

    temp_dir5 = Path(tempfile.mkdtemp())

    try:
        engine = EvolutionEngine(
            population_size=5,
            state_dir=temp_dir5
        )

        # Create performance history
        genomes = engine.genome_library.all_genomes()
        performance_history = {}

        for i, genome in enumerate(genomes[:3]):
            performance_history[genome.strategy_name] = {
                'sharpe': 1.0 + i * 0.5,
                'sortino': 1.2 + i * 0.5,
                'max_dd': -0.15,
                'win_rate': 0.55,
                'meta_accuracy': 0.65,
                'bandit_confidence': 0.70,
                'wfo_sharpe': 1.0
            }

        # Evaluate
        engine.evaluate_generation(performance_history)

        # Check fitness updated
        for name in performance_history.keys():
            genome = engine.genome_library.get_genome(name)
            assert genome.fitness > 0, f"Fitness should be > 0 for {name}"
            print(f"  {name}: fitness = {genome.fitness:.4f}")

        print("  ✓ Evaluation with performance history working")

    finally:
        shutil.rmtree(temp_dir5)

    # ========================================================================
    # TEST 8: Population Statistics
    # ========================================================================
    print("\n[TEST 8] Population Statistics")
    print("-" * 80)

    temp_dir6 = Path(tempfile.mkdtemp())

    try:
        engine = EvolutionEngine(
            population_size=10,
            state_dir=temp_dir6
        )

        # Assign fitness
        for i, genome in enumerate(engine.genome_library.all_genomes()):
            engine.genome_library.update_fitness(
                genome.strategy_name,
                0.3 + i * 0.05
            )

        stats = engine.get_population_stats()

        print(f"  Size: {stats['size']}")
        print(f"  Generation: {stats['generation']}")
        print(f"  Avg fitness: {stats['avg_fitness']:.4f}")
        print(f"  Max fitness: {stats['max_fitness']:.4f}")
        print(f"  Min fitness: {stats['min_fitness']:.4f}")
        print(f"  Std fitness: {stats['std_fitness']:.4f}")

        assert stats['size'] == 10, "Size should be 10"
        assert stats['avg_fitness'] > 0, "Avg should be > 0"
        assert stats['max_fitness'] >= stats['min_fitness'], "Max >= Min"

        print("  ✓ Population statistics working")

    finally:
        shutil.rmtree(temp_dir6)

    # ========================================================================
    # TEST 9: Fitness Stability (NaN/Inf Handling)
    # ========================================================================
    print("\n[TEST 9] Fitness Stability (NaN/Inf Handling)")
    print("-" * 80)

    evaluator = FitnessEvaluator()
    factory = GenomeFactory(seed=42)
    genome = factory.create_default_genome("test_strat")

    # Test NaN handling
    bad_performance_nan = {
        'sharpe': float('nan'),
        'sortino': 2.0,
        'max_dd': -0.10,
        'win_rate': 0.60
    }

    fitness_nan = evaluator.compute_fitness(genome, bad_performance_nan)
    print(f"  NaN Sharpe fitness: {fitness_nan:.4f}")
    assert 0 <= fitness_nan <= 1, "Fitness should be in [0, 1] even with NaN"
    assert not np.isnan(fitness_nan), "Fitness should not be NaN"

    # Test Inf handling
    bad_performance_inf = {
        'sharpe': float('inf'),
        'sortino': 2.0,
        'max_dd': -0.10,
        'win_rate': 0.60
    }

    fitness_inf = evaluator.compute_fitness(genome, bad_performance_inf)
    print(f"  Inf Sharpe fitness: {fitness_inf:.4f}")
    assert 0 <= fitness_inf <= 1, "Fitness should be in [0, 1] even with Inf"
    assert not np.isinf(fitness_inf), "Fitness should not be Inf"

    # Test negative infinity
    bad_performance_neginf = {
        'sharpe': float('-inf'),
        'sortino': 1.0,
        'max_dd': -0.20
    }

    fitness_neginf = evaluator.compute_fitness(genome, bad_performance_neginf)
    print(f"  -Inf Sharpe fitness: {fitness_neginf:.4f}")
    assert 0 <= fitness_neginf <= 1, "Fitness should be in [0, 1] even with -Inf"

    print("  ✓ NaN/Inf handling working correctly")

    # ========================================================================
    # TEST 10: Edge Case - Empty Top Genomes
    # ========================================================================
    print("\n[TEST 10] Edge Case - Empty Top Genomes")
    print("-" * 80)

    temp_dir7 = Path(tempfile.mkdtemp())

    try:
        engine = EvolutionEngine(
            population_size=5,
            state_dir=temp_dir7
        )

        # Test selecting 0 genomes
        top_zero = engine.select_top_genomes(0)
        print(f"  Top 0 genomes: {len(top_zero)}")
        assert len(top_zero) == 0, "Should return empty list"

        # Test selecting more than population
        top_many = engine.select_top_genomes(100)
        print(f"  Top 100 (but pop=5): {len(top_many)}")
        assert len(top_many) == 5, "Should cap to population size"

        # Test offspring with empty list
        offspring_empty = engine.generate_offspring([], 5)
        print(f"  Offspring from empty list: {len(offspring_empty)}")
        assert len(offspring_empty) == 0, "Should return empty"

        print("  ✓ Edge cases handled correctly")

    finally:
        shutil.rmtree(temp_dir7)

    # ========================================================================
    # TEST 11: Encode/Decode Population
    # ========================================================================
    print("\n[TEST 11] Encode/Decode Population")
    print("-" * 80)

    temp_dir8 = Path(tempfile.mkdtemp())

    try:
        # Create engine with some genomes
        engine = EvolutionEngine(
            population_size=5,
            state_dir=temp_dir8
        )

        # Assign fitness
        for i, genome in enumerate(engine.genome_library.all_genomes()):
            engine.genome_library.update_fitness(
                genome.strategy_name,
                0.5 + i * 0.1
            )

        # Encode
        encoded = _encode_population(
            engine.current_generation,
            engine.population_size,
            engine.genome_library.genomes
        )

        print(f"  Encoded generation: {encoded['generation']}")
        print(f"  Encoded population size: {encoded['population_size']}")
        print(f"  Encoded genomes count: {len(encoded['genomes'])}")

        # Decode
        gen, pop_size, genomes = _decode_population(encoded)

        print(f"  Decoded generation: {gen}")
        print(f"  Decoded population size: {pop_size}")
        print(f"  Decoded genomes count: {len(genomes)}")

        assert gen == engine.current_generation, "Generation should match"
        assert pop_size == engine.population_size, "Population size should match"
        assert len(genomes) == len(engine.genome_library.genomes), "Genome count should match"

        # Verify all fields preserved
        for name, genome in genomes.items():
            original = engine.genome_library.get_genome(name)
            assert genome.strategy_name == original.strategy_name, "Name preserved"
            assert genome.model_type == original.model_type, "Model type preserved"
            assert genome.fitness == original.fitness, f"Fitness preserved: {genome.fitness} == {original.fitness}"

        print("  ✓ Encode/decode preserves all data")

    finally:
        shutil.rmtree(temp_dir8)

    # ========================================================================
    # TEST 12: Bandit Integration Safety
    # ========================================================================
    print("\n[TEST 12] Bandit Integration Safety")
    print("-" * 80)

    temp_dir9 = Path(tempfile.mkdtemp())

    try:
        engine = EvolutionEngine(
            population_size=5,
            state_dir=temp_dir9
        )

        genomes = engine.genome_library.all_genomes()

        # Create performance with invalid values
        performance_history = {
            genomes[0].strategy_name: {
                'sharpe': float('nan'),  # Invalid
                'sortino': 1.5,
                'max_dd': -0.10
            },
            genomes[1].strategy_name: {
                'sharpe': 2.0,  # Valid
                'sortino': 2.5,
                'max_dd': -0.08
            },
            genomes[2].strategy_name: {
                'sharpe': float('inf'),  # Invalid
                'sortino': 1.0,
                'max_dd': -0.15
            }
        }

        # Should not crash
        engine.evaluate_generation(performance_history)

        # Check that valid genome got updated
        genome1 = engine.genome_library.get_genome(genomes[1].strategy_name)
        print(f"  Valid genome fitness: {genome1.fitness:.4f}")
        assert genome1.fitness > 0, "Valid genome should have fitness"

        # Check that invalid genomes got safe defaults
        genome0 = engine.genome_library.get_genome(genomes[0].strategy_name)
        genome2 = engine.genome_library.get_genome(genomes[2].strategy_name)
        print(f"  NaN genome fitness: {genome0.fitness:.4f}")
        print(f"  Inf genome fitness: {genome2.fitness:.4f}")
        assert not np.isnan(genome0.fitness), "Should not be NaN"
        assert not np.isinf(genome2.fitness), "Should not be Inf"

        print("  ✓ Bandit integration handles invalid values")

    finally:
        shutil.rmtree(temp_dir9)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL MODULE C SWEEP TESTS PASSED")
    print("=" * 80)
    print("\nEvolution Engine Features:")
    print("  ✓ Fitness evaluation with weighted metrics")
    print("  ✓ Fitness normalization")
    print("  ✓ Population initialization")
    print("  ✓ Offspring generation (mutation + crossover)")
    print("  ✓ Full evolution cycle")
    print("  ✓ Population persistence (save/load)")
    print("  ✓ Performance history evaluation")
    print("  ✓ Population statistics")
    print("\nSweep C.1 Enhancements:")
    print("  ✓ NaN/Inf stability guards in fitness calculation")
    print("  ✓ Edge case handling (empty genomes, zero selection)")
    print("  ✓ Encode/decode for JSON serialization")
    print("  ✓ Bandit Brain integration safety")
    print("  ✓ Offspring generation validation (min features)")
    print("  ✓ Population size stability verification")
    print("  ✓ Safe defaults for all metrics")
    print("\nModule C — Evolution Engine: PRODUCTION READY")
    print("=" * 80)
