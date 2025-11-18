"""
PRADO9_EVO Evolution Engine
Module A - Bandit Brain (Hierarchical Thompson Sampling)
Module B - Strategy Genome (Evolution DNA Layer)
Module C - Evolution Engine (Genetic + Bayesian Optimization)
Module D - Meta-Learner (Strategy Performance Predictor)
Module E - Performance Memory (High-Resolution Quant Memory)
Module F - Correlation Cluster Engine (Strategy Diversity Layer)
Module G - Evolutionary Allocator (Adaptive Hybrid Alpha Blender)
"""

from .bandit_brain import (
    BanditBrain,
    StrategySelectionBandit,
    HyperparameterBandit,
    RegimeConfidenceBandit,
    evo_select_strategy,
    evo_update_strategy,
    evo_select_config,
    evo_update_config,
    evo_regime_confidence,
    evo_update_regime,
)

from .genome import (
    StrategyGenome,
    GenomeFactory,
    GenomeLibrary,
    save_genomes,
    load_genomes,
    evo_get_genome,
    evo_update_fitness,
    evo_evolve_generation,
    evo_all_genomes,
)

from .evolution_engine import (
    FitnessEvaluator,
    EvolutionEngine,
)

from .meta_learner import (
    MetaFeatureBuilder,
    MetaLearner,
    MetaLearningEngine,
    evo_meta_predict,
    evo_meta_train,
    evo_meta_load,
)

from .performance_memory import (
    PerformanceRecord,
    PerformanceMemory,
    evo_perf_add,
    evo_perf_get,
    evo_perf_rolling,
    evo_perf_summary,
    evo_perf_save,
    evo_perf_load,
)

from .correlation_engine import (
    CorrelationMatrixBuilder,
    ClusterEngine,
    CorrelationClusterEngine,
    evo_corr_update,
    evo_corr_get_clusters,
    evo_corr_get_penalties,
    evo_corr_get_uniqueness,
)

from .evolutionary_allocator import (
    StrategySignal,
    AllocationWeights,
    ConflictEngine,
    AllocationDecision,
    EvolutionaryAllocator,
    evo_allocate,
)

__all__ = [
    # Module A - Bandit Brain
    'BanditBrain',
    'StrategySelectionBandit',
    'HyperparameterBandit',
    'RegimeConfidenceBandit',
    'evo_select_strategy',
    'evo_update_strategy',
    'evo_select_config',
    'evo_update_config',
    'evo_regime_confidence',
    'evo_update_regime',
    # Module B - Genome
    'StrategyGenome',
    'GenomeFactory',
    'GenomeLibrary',
    'save_genomes',
    'load_genomes',
    'evo_get_genome',
    'evo_update_fitness',
    'evo_evolve_generation',
    'evo_all_genomes',
    # Module C - Evolution Engine
    'FitnessEvaluator',
    'EvolutionEngine',
    # Module D - Meta-Learner
    'MetaFeatureBuilder',
    'MetaLearner',
    'MetaLearningEngine',
    'evo_meta_predict',
    'evo_meta_train',
    'evo_meta_load',
    # Module E - Performance Memory
    'PerformanceRecord',
    'PerformanceMemory',
    'evo_perf_add',
    'evo_perf_get',
    'evo_perf_rolling',
    'evo_perf_summary',
    'evo_perf_save',
    'evo_perf_load',
    # Module F - Correlation Cluster Engine
    'CorrelationMatrixBuilder',
    'ClusterEngine',
    'CorrelationClusterEngine',
    'evo_corr_update',
    'evo_corr_get_clusters',
    'evo_corr_get_penalties',
    'evo_corr_get_uniqueness',
    # Module G - Evolutionary Allocator
    'StrategySignal',
    'AllocationWeights',
    'ConflictEngine',
    'AllocationDecision',
    'EvolutionaryAllocator',
    'evo_allocate',
]
