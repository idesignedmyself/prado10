"""
PRADO9_EVO Test Suite — Module Y: Position Scaling Engine

Comprehensive validation of confidence-based position scaling with:
- Meta-learner confidence scaling
- Bandit exploration/exploitation
- Regime-based aggression
- Pyramiding logic
- Correlation penalty
- Determinism verification

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from src.afml_system.risk import PositionScaler, ScalingFactors


class TestPositionScaler:
    """Test suite for Module Y: Position Scaling Engine."""

    def setup_method(self):
        """Initialize scaler for each test."""
        self.scaler = PositionScaler(
            meta_confidence_range=(0.5, 1.5),
            bandit_min_scale=0.2,
            max_position=3.0
        )

    def test_meta_confidence_scaling(self):
        """
        Test 1: Meta-learner confidence scaling
        - 0% confidence (prob=0.0) → 0.5x scaling
        - 50% confidence (prob=0.5) → 1.0x scaling
        - 100% confidence (prob=1.0) → 1.5x scaling
        """
        position = 1.0
        regime = 'NORMAL'
        bandit_weight = 1.0

        # Low confidence → scale down
        scaled_low = self.scaler.scale(position, meta_prob=0.0, bandit_weight=bandit_weight, regime=regime)
        assert 0.4 < scaled_low < 0.6, f"Expected ~0.5x for 0% confidence, got {scaled_low:.3f}x"

        # Neutral confidence → no change
        scaled_neutral = self.scaler.scale(position, meta_prob=0.5, bandit_weight=bandit_weight, regime=regime)
        assert 0.9 < scaled_neutral < 1.1, f"Expected ~1.0x for 50% confidence, got {scaled_neutral:.3f}x"

        # High confidence → scale up
        scaled_high = self.scaler.scale(position, meta_prob=1.0, bandit_weight=bandit_weight, regime=regime)
        assert 1.4 < scaled_high < 1.6, f"Expected ~1.5x for 100% confidence, got {scaled_high:.3f}x"

        print("✅ Test 1: Meta-learner confidence scaling works correctly")

    def test_bandit_exploration_exploitation(self):
        """
        Test 2: Bandit weight scaling
        - Exploration (weight=0.1) → reduced to min 0.2x
        - Exploitation (weight=1.0) → full 1.0x
        """
        position = 1.0
        meta_prob = 0.5
        regime = 'NORMAL'

        # Exploration → minimum scale
        scaled_explore = self.scaler.scale(position, meta_prob=meta_prob, bandit_weight=0.1, regime=regime)
        assert scaled_explore < 0.3, f"Expected exploration to reduce position, got {scaled_explore:.3f}x"

        # Exploitation → full scale
        scaled_exploit = self.scaler.scale(position, meta_prob=meta_prob, bandit_weight=1.0, regime=regime)
        assert 0.9 < scaled_exploit < 1.1, f"Expected exploitation at full scale, got {scaled_exploit:.3f}x"

        # Verify exploration is significantly smaller
        assert scaled_explore < scaled_exploit * 0.5, "Exploration should be <50% of exploitation"

        print("✅ Test 2: Bandit exploration/exploitation scaling works correctly")

    def test_regime_based_scaling(self):
        """
        Test 3: Regime-based aggression adjustments
        - TRENDING → 1.4x (aggressive)
        - HIGH_VOL → 0.7x (conservative)
        - LOW_VOL → 1.2x (moderately aggressive)
        - NORMAL → 1.0x (neutral)
        """
        position = 1.0
        meta_prob = 0.5
        bandit_weight = 1.0

        regimes = {
            'TRENDING': (1.3, 1.5, "aggressive"),
            'HIGH_VOL': (0.6, 0.8, "conservative"),
            'LOW_VOL': (1.1, 1.3, "moderately aggressive"),
            'NORMAL': (0.9, 1.1, "neutral")
        }

        for regime, (min_expected, max_expected, description) in regimes.items():
            scaled = self.scaler.scale(position, meta_prob=meta_prob, bandit_weight=bandit_weight, regime=regime)
            assert min_expected < scaled < max_expected, \
                f"{regime} should be {description} ({min_expected:.1f}-{max_expected:.1f}x), got {scaled:.3f}x"

        print("✅ Test 3: Regime-based scaling adjustments work correctly")

    def test_combined_scaling_pipeline(self):
        """
        Test 4: Combined scaling pipeline
        Verify all factors multiply correctly:
        position × meta_factor × bandit_factor × regime_factor
        """
        position = 1.0

        # High confidence + exploitation + trending → maximum aggression
        scaled_max = self.scaler.scale(
            position=position,
            meta_prob=0.8,  # 1.3x
            bandit_weight=1.0,  # 1.0x
            regime='TRENDING'  # 1.4x
        )
        # Expected: 1.0 × 1.3 × 1.0 × 1.4 = 1.82x
        assert 1.7 < scaled_max < 2.0, f"Expected ~1.82x for max aggression, got {scaled_max:.3f}x"

        # Low confidence + exploration + high vol → minimum aggression
        scaled_min = self.scaler.scale(
            position=position,
            meta_prob=0.2,  # 0.7x
            bandit_weight=0.3,  # 0.3x
            regime='HIGH_VOL'  # 0.7x
        )
        # Expected: 1.0 × 0.7 × 0.3 × 0.7 = 0.147x
        assert 0.1 < scaled_min < 0.2, f"Expected ~0.15x for min aggression, got {scaled_min:.3f}x"

        # Verify max > min
        assert scaled_max > scaled_min * 10, "Maximum aggression should be >10x minimum"

        print("✅ Test 4: Combined scaling pipeline works correctly")

    def test_position_capping(self):
        """
        Test 5: Position capping at safety limits
        - No position should exceed ±3.0x (max_position)
        """
        # Extreme bullish scenario
        scaled_max = self.scaler.scale(
            position=5.0,  # Large input
            meta_prob=1.0,  # Max confidence
            bandit_weight=1.0,
            regime='TRENDING'
        )
        assert scaled_max <= 3.0, f"Position should be capped at 3.0x, got {scaled_max:.3f}x"

        # Extreme bearish scenario
        scaled_min = self.scaler.scale(
            position=-5.0,  # Large negative
            meta_prob=1.0,
            bandit_weight=1.0,
            regime='TRENDING'
        )
        assert scaled_min >= -3.0, f"Position should be capped at -3.0x, got {scaled_min:.3f}x"

        print("✅ Test 5: Position capping at ±3.0x works correctly")

    def test_pyramid_winners_losers(self):
        """
        Test 6: Pyramiding logic - add to winners, reduce losers
        - Winning trade (+5% P&L) → increase position
        - Losing trade (-3% P&L) → decrease position
        """
        position = 1.0
        meta_prob = 0.7

        # Winning position → pyramid up
        pyramided_winner = self.scaler.pyramid_winners(
            position=position,
            meta_prob=meta_prob,
            current_pnl=0.05,  # +5% profit
            pnl_threshold=0.02
        )
        assert pyramided_winner > position, f"Winning position should increase, got {pyramided_winner:.3f}x"

        # Losing position → cut down
        pyramided_loser = self.scaler.pyramid_winners(
            position=position,
            meta_prob=meta_prob,
            current_pnl=-0.03,  # -3% loss
            pnl_threshold=0.02
        )
        assert pyramided_loser < position, f"Losing position should decrease, got {pyramided_loser:.3f}x"

        # Neutral position → no change
        pyramided_neutral = self.scaler.pyramid_winners(
            position=position,
            meta_prob=meta_prob,
            current_pnl=0.01,  # Within threshold
            pnl_threshold=0.02
        )
        assert 0.95 < pyramided_neutral < 1.05, f"Neutral position should stay same, got {pyramided_neutral:.3f}x"

        print("✅ Test 6: Pyramiding logic works correctly")

    def test_correlation_penalty(self):
        """
        Test 7: Correlation penalty reduces position size
        - 0% penalty → no reduction
        - 50% penalty → ~50% reduction
        - 80% penalty → ~80% reduction (floor at 20%)
        """
        position = 1.0
        meta_prob = 0.5
        bandit_weight = 1.0
        regime = 'NORMAL'

        # No penalty
        scaled_no_penalty = self.scaler.scale(
            position, meta_prob, bandit_weight, regime, correlation_penalty=0.0
        )

        # 50% penalty
        scaled_50_penalty = self.scaler.scale(
            position, meta_prob, bandit_weight, regime, correlation_penalty=0.5
        )

        # 80% penalty (should floor at 20%)
        scaled_80_penalty = self.scaler.scale(
            position, meta_prob, bandit_weight, regime, correlation_penalty=0.8
        )

        assert scaled_50_penalty < scaled_no_penalty, "50% penalty should reduce position"
        assert scaled_80_penalty < scaled_50_penalty, "80% penalty should reduce more than 50%"
        assert scaled_80_penalty >= 0.15, "80% penalty should floor at ~20% of base"

        print("✅ Test 7: Correlation penalty works correctly")

    def test_determinism(self):
        """
        Test 8: Determinism - same inputs produce identical outputs
        """
        position = 1.0
        meta_prob = 0.65
        bandit_weight = 0.8
        regime = 'TRENDING'

        # Run 5 times with identical inputs
        results = []
        for _ in range(5):
            scaled = self.scaler.scale(position, meta_prob, bandit_weight, regime)
            results.append(scaled)

        # All results should be identical
        assert len(set(results)) == 1, f"Results should be deterministic, got {results}"
        print(f"   Deterministic output: {results[0]:.6f}x (5/5 runs identical)")

        print("✅ Test 8: Determinism verified")

    def test_scaling_factors_breakdown(self):
        """
        Test 9: ScalingFactors return provides transparency
        """
        factors = self.scaler.scale(
            position=1.0,
            meta_prob=0.7,
            bandit_weight=0.8,
            regime='TRENDING',
            correlation_penalty=0.2,
            return_factors=True
        )

        assert isinstance(factors, ScalingFactors), "Should return ScalingFactors object"
        assert hasattr(factors, 'meta_confidence_factor'), "Should have meta_confidence_factor"
        assert hasattr(factors, 'bandit_factor'), "Should have bandit_factor"
        assert hasattr(factors, 'regime_factor'), "Should have regime_factor"
        assert hasattr(factors, 'correlation_factor'), "Should have correlation_factor"
        assert hasattr(factors, 'final_scale'), "Should have final_scale"
        assert hasattr(factors, 'scaled_position'), "Should have scaled_position"

        # Verify factors are reasonable
        assert 1.0 < factors.meta_confidence_factor < 1.5, "Meta factor should be in range"
        assert 0.8 <= factors.bandit_factor <= 1.0, "Bandit factor should be in range"
        assert factors.regime_factor == 1.4, "TRENDING regime should be 1.4x"
        assert 0.7 < factors.correlation_factor < 1.0, "Correlation factor should be reduced"

        print("✅ Test 9: ScalingFactors breakdown works correctly")
        print(f"   Meta: {factors.meta_confidence_factor:.3f}x")
        print(f"   Bandit: {factors.bandit_factor:.3f}x")
        print(f"   Regime: {factors.regime_factor:.3f}x")
        print(f"   Correlation: {factors.correlation_factor:.3f}x")
        print(f"   Final: {factors.scaled_position:.3f}x")

    def test_batch_scaling(self):
        """
        Test 10: Vectorized batch scaling for backtest performance
        """
        # Create sample data
        n = 100
        positions = pd.Series([1.0] * n)
        meta_probs = pd.Series(np.random.uniform(0.3, 0.8, n))
        bandit_weights = pd.Series(np.random.uniform(0.5, 1.0, n))
        regimes = pd.Series(['NORMAL'] * 50 + ['TRENDING'] * 50)

        # Scale batch
        scaled = self.scaler.scale_batch(
            positions=positions,
            meta_probs=meta_probs,
            bandit_weights=bandit_weights,
            regimes=regimes
        )

        assert len(scaled) == n, f"Should return {n} results"
        assert isinstance(scaled, pd.Series), "Should return pandas Series"
        assert all(scaled <= 3.0), "All positions should be capped at 3.0x"
        assert all(scaled >= -3.0), "All positions should be capped at -3.0x"

        # Verify TRENDING regime has higher average scaling
        normal_mean = scaled[:50].mean()
        trending_mean = scaled[50:].mean()
        assert trending_mean > normal_mean, "TRENDING should have higher average scaling"

        print("✅ Test 10: Batch scaling works correctly")
        print(f"   NORMAL mean: {normal_mean:.3f}x")
        print(f"   TRENDING mean: {trending_mean:.3f}x")

    def test_regime_volatility_adjustment(self):
        """
        Test 11: Regime + volatility combined adjustment
        """
        position = 1.0
        regime = 'HIGH_VOL'
        current_vol = 0.30  # 30% volatility
        baseline_vol = 0.15  # 15% baseline

        scaled = self.scaler.scale_by_regime_volatility(
            position=position,
            regime=regime,
            current_vol=current_vol,
            baseline_vol=baseline_vol
        )

        # HIGH_VOL regime (0.7x) × high volatility adjustment → very conservative
        # Expected: 1.0 × 0.7 × (0.15/0.30) = 0.35x
        assert 0.3 < scaled < 0.4, f"High vol in HIGH_VOL regime should be very conservative, got {scaled:.3f}x"

        # Test TRENDING with low volatility → aggressive
        scaled_aggressive = self.scaler.scale_by_regime_volatility(
            position=position,
            regime='TRENDING',
            current_vol=0.10,  # Low volatility
            baseline_vol=baseline_vol
        )

        # TRENDING regime (1.4x) × low volatility adjustment → aggressive
        # Expected: 1.0 × 1.4 × (0.15/0.10) = 2.1x (capped at 2.0x by vol_adjustment cap)
        assert scaled_aggressive > scaled * 3, "TRENDING with low vol should be much more aggressive"

        print("✅ Test 11: Regime + volatility adjustment works correctly")


def run_manual_tests():
    """
    Standalone manual test runner (no pytest required).
    Run with: python tests/test_position_scaler.py
    """
    print("\n" + "="*70)
    print("PRADO9_EVO — Module Y Position Scaler Validation")
    print("="*70 + "\n")

    test_suite = TestPositionScaler()
    tests = [
        test_suite.test_meta_confidence_scaling,
        test_suite.test_bandit_exploration_exploitation,
        test_suite.test_regime_based_scaling,
        test_suite.test_combined_scaling_pipeline,
        test_suite.test_position_capping,
        test_suite.test_pyramid_winners_losers,
        test_suite.test_correlation_penalty,
        test_suite.test_determinism,
        test_suite.test_scaling_factors_breakdown,
        test_suite.test_batch_scaling,
        test_suite.test_regime_volatility_adjustment
    ]

    passed = 0
    failed = 0

    for test in tests:
        test_suite.setup_method()
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed}/{passed + failed} passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED - Module Y is production-ready")
    else:
        print(f"❌ {failed} tests failed")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_manual_tests()
    exit(0 if success else 1)
