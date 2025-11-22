"""
Hybrid ML Fusion Engine

Fuses multiple signal sources:
1. Rule-based strategy ensemble (11 strategies)
2. ML horizon models (1d, 3d, 5d, 10d)
3. ML regime-specific models (per regime per horizon)
4. Confidence-weighted blending

Output: Final position signal with diagnostic breakdown
"""

import numpy as np
from typing import Dict, Tuple


class HybridMLFusion:
    """
    Fuses:
        1. Rule-based PRADO9_EVO signal
        2. ML Horizon model signal
        3. ML Regime-specific model signal

    Ensures final output stays stable, bounded, and meaningful.
    """

    def __init__(self):
        pass

    def fuse(
        self,
        rule_signal: float,
        ml_horizon_signal: float,
        ml_regime_signal: float,
        ml_horizon_conf: float,
        ml_regime_conf: float,
        ml_weight: float = 0.25
    ) -> Tuple[float, Dict]:
        """
        Fuse rule and ML signals with confidence weighting.

        Args:
            rule_signal: Signal from rule-based ensemble [-1, 1]
            ml_horizon_signal: Signal from horizon ML model
            ml_regime_signal: Signal from regime ML model
            ml_horizon_conf: Confidence of horizon prediction [0, 1]
            ml_regime_conf: Confidence of regime prediction [0, 1]
            ml_weight: Weight given to ML signals (default 0.25)

        Returns:
            final_signal: Combined signal in [-1, 1]
            diagnostics: Dict with signal breakdown
        """
        # ML weighted vote
        ml_vote = (
            0.6 * ml_horizon_signal * ml_horizon_conf +
            0.4 * ml_regime_signal * ml_regime_conf
        )

        ml_vote *= ml_weight  # scale ML impact

        # Blend rule-based and ML signals
        blended = rule_signal + ml_vote

        # Final stabilized output
        final_signal = float(np.tanh(blended))

        diagnostics = {
            "rule_signal": float(rule_signal),
            "ml_horizon_signal": float(ml_horizon_signal),
            "ml_regime_signal": float(ml_regime_signal),
            "ml_horizon_conf": float(ml_horizon_conf),
            "ml_regime_conf": float(ml_regime_conf),
            "ml_vote": float(ml_vote),
            "ml_contribution": float(ml_vote),
            "final_signal": final_signal
        }

        return final_signal, diagnostics
