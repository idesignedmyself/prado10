"""
PRADO9_EVO SWEEP Y.1 — Risk Scaling Validation

Validates that Module Y: Position Scaling Engine behaves correctly:
1. Trend regimes increase position size (1.4x)
2. High-vol regimes shrink size (0.7x)
3. MetaProbability scaling increases size with confidence
4. Bandit weight reduces size for weak strategies
5. Deterministic behavior across multiple signals
6. No exposure runaway ("position explosion" test)

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from src.afml_system.risk import PositionScaler


class TestSweepY1RiskScaling:
    """SWEEP Y.1: Comprehensive risk scaling validation."""

    def setup_method(self):
        """Initialize scaler for each test."""
        self.scaler = PositionScaler(
            meta_confidence_range=(0.5, 1.5),
            bandit_min_scale=0.2,
            max_position=3.0
        )
        self.base_position = 1.0

    def test_1_trend_regime_increases_position(self):
        """
        Test 1: Trend regimes increase position size (1.4x)

        Validates that TRENDING regime applies 1.4x multiplier
        with neutral confidence and full exploitation.
        """
        print("\n" + "="*70)
        print("Test 1: Trend Regime Increases Position Size")
        print("="*70)

        # Neutral confidence, full exploitation, TRENDING regime
        scaled_trending = self.scaler.scale(
            position=self.base_position,
            meta_prob=0.5,      # Neutral confidence
            bandit_weight=1.0,  # Full exploitation
            regime='TRENDING'
        )

        # Neutral confidence, full exploitation, NORMAL regime (baseline)
        scaled_normal = self.scaler.scale(
            position=self.base_position,
            meta_prob=0.5,
            bandit_weight=1.0,
            regime='NORMAL'
        )

        print(f"NORMAL regime:   {scaled_normal:.4f}x")
        print(f"TRENDING regime: {scaled_trending:.4f}x")
        print(f"Increase factor: {scaled_trending / scaled_normal:.4f}x")

        # TRENDING should be 1.4x of NORMAL
        expected_ratio = 1.4
        actual_ratio = scaled_trending / scaled_normal

        assert 1.35 < actual_ratio < 1.45, \
            f"TRENDING should be ~1.4x NORMAL, got {actual_ratio:.4f}x"

        print("✅ PASS: TRENDING regime increases position by 1.4x")
        return True

    def test_2_high_vol_regime_shrinks_position(self):
        """
        Test 2: High-vol regimes shrink size (0.7x)

        Validates that HIGH_VOL regime applies 0.7x multiplier
        to protect capital during volatile periods.
        """
        print("\n" + "="*70)
        print("Test 2: High-Vol Regime Shrinks Position Size")
        print("="*70)

        # Neutral confidence, full exploitation, HIGH_VOL regime
        scaled_high_vol = self.scaler.scale(
            position=self.base_position,
            meta_prob=0.5,
            bandit_weight=1.0,
            regime='HIGH_VOL'
        )

        # Neutral confidence, full exploitation, NORMAL regime (baseline)
        scaled_normal = self.scaler.scale(
            position=self.base_position,
            meta_prob=0.5,
            bandit_weight=1.0,
            regime='NORMAL'
        )

        print(f"NORMAL regime:   {scaled_normal:.4f}x")
        print(f"HIGH_VOL regime: {scaled_high_vol:.4f}x")
        print(f"Reduction factor: {scaled_high_vol / scaled_normal:.4f}x")

        # HIGH_VOL should be 0.7x of NORMAL
        expected_ratio = 0.7
        actual_ratio = scaled_high_vol / scaled_normal

        assert 0.65 < actual_ratio < 0.75, \
            f"HIGH_VOL should be ~0.7x NORMAL, got {actual_ratio:.4f}x"

        print("✅ PASS: HIGH_VOL regime reduces position by 0.7x")
        return True

    def test_3_meta_probability_increases_with_confidence(self):
        """
        Test 3: MetaProbability scaling increases size with confidence

        Validates that higher meta-learner confidence leads to larger positions:
        - Low confidence (0.2) → scale down
        - Medium confidence (0.5) → neutral
        - High confidence (0.8) → scale up
        """
        print("\n" + "="*70)
        print("Test 3: Meta Probability Scales with Confidence")
        print("="*70)

        test_cases = [
            (0.2, "low"),
            (0.5, "medium"),
            (0.8, "high")
        ]

        results = []
        for meta_prob, label in test_cases:
            scaled = self.scaler.scale(
                position=self.base_position,
                meta_prob=meta_prob,
                bandit_weight=1.0,
                regime='NORMAL'
            )
            results.append((label, meta_prob, scaled))
            print(f"{label.capitalize()} confidence (prob={meta_prob}): {scaled:.4f}x")

        # Extract scaled values
        low_scaled = results[0][2]
        medium_scaled = results[1][2]
        high_scaled = results[2][2]

        # Validate ordering: low < medium < high
        assert low_scaled < medium_scaled < high_scaled, \
            f"Scaling should increase with confidence: {low_scaled:.4f} < {medium_scaled:.4f} < {high_scaled:.4f}"

        # Validate specific ranges
        # Low (0.2): 0.5 + (1.5-0.5)*0.2 = 0.7x
        assert 0.65 < low_scaled < 0.75, f"Low confidence should be ~0.7x, got {low_scaled:.4f}x"

        # Medium (0.5): 0.5 + (1.5-0.5)*0.5 = 1.0x
        assert 0.95 < medium_scaled < 1.05, f"Medium confidence should be ~1.0x, got {medium_scaled:.4f}x"

        # High (0.8): 0.5 + (1.5-0.5)*0.8 = 1.3x
        assert 1.25 < high_scaled < 1.35, f"High confidence should be ~1.3x, got {high_scaled:.4f}x"

        print("✅ PASS: Meta probability correctly scales with confidence")
        return True

    def test_4_bandit_weight_reduces_weak_strategies(self):
        """
        Test 4: Bandit weight reduces size for weak strategies

        Validates that low bandit weights (exploration) reduce position size:
        - High weight (1.0) → full exposure
        - Medium weight (0.5) → reduced exposure
        - Low weight (0.2) → minimal exposure (floor at 0.2x)
        """
        print("\n" + "="*70)
        print("Test 4: Bandit Weight Reduces Weak Strategies")
        print("="*70)

        test_cases = [
            (1.0, "high"),
            (0.5, "medium"),
            (0.2, "low")
        ]

        results = []
        for bandit_weight, label in test_cases:
            scaled = self.scaler.scale(
                position=self.base_position,
                meta_prob=0.5,
                bandit_weight=bandit_weight,
                regime='NORMAL'
            )
            results.append((label, bandit_weight, scaled))
            print(f"{label.capitalize()} bandit weight ({bandit_weight}): {scaled:.4f}x")

        # Extract scaled values
        high_scaled = results[0][2]
        medium_scaled = results[1][2]
        low_scaled = results[2][2]

        # Validate ordering: low < medium < high
        assert low_scaled < medium_scaled < high_scaled, \
            f"Scaling should increase with bandit weight: {low_scaled:.4f} < {medium_scaled:.4f} < {high_scaled:.4f}"

        # Validate that high weight keeps position near 1.0x
        assert 0.95 < high_scaled < 1.05, f"High bandit weight should be ~1.0x, got {high_scaled:.4f}x"

        # Validate that low weight reduces significantly
        assert low_scaled < 0.3, f"Low bandit weight should reduce significantly, got {low_scaled:.4f}x"

        print("✅ PASS: Bandit weight correctly reduces weak strategies")
        return True

    def test_5_deterministic_behavior(self):
        """
        Test 5: Deterministic behavior across multiple signals

        Validates that identical inputs produce identical outputs
        across multiple runs.
        """
        print("\n" + "="*70)
        print("Test 5: Deterministic Behavior")
        print("="*70)

        # Test parameters
        test_params = {
            'position': 1.0,
            'meta_prob': 0.65,
            'bandit_weight': 0.8,
            'regime': 'TRENDING'
        }

        # Run 10 times with identical inputs
        results = []
        for i in range(10):
            scaled = self.scaler.scale(**test_params)
            results.append(scaled)

        print(f"Test parameters: {test_params}")
        print(f"Run 1:  {results[0]:.8f}x")
        print(f"Run 2:  {results[1]:.8f}x")
        print(f"Run 10: {results[9]:.8f}x")

        # All results should be identical
        unique_results = set(results)
        assert len(unique_results) == 1, \
            f"Expected 1 unique result, got {len(unique_results)}: {unique_results}"

        print(f"✅ PASS: All 10 runs produced identical output: {results[0]:.8f}x")
        return True

    def test_6_no_position_explosion(self):
        """
        Test 6: No exposure runaway ("position explosion" test)

        Validates that even with extreme inputs, positions are capped
        at safety limits (±3.0x) to prevent runaway exposure.
        """
        print("\n" + "="*70)
        print("Test 6: No Position Explosion")
        print("="*70)

        # Extreme bullish scenario
        extreme_long = self.scaler.scale(
            position=10.0,       # Very large input
            meta_prob=1.0,       # Maximum confidence
            bandit_weight=1.0,   # Full exploitation
            regime='TRENDING'    # Aggressive regime (1.4x)
        )

        # Extreme bearish scenario
        extreme_short = self.scaler.scale(
            position=-10.0,      # Very large negative
            meta_prob=1.0,
            bandit_weight=1.0,
            regime='TRENDING'
        )

        print(f"Input:  +10.0x position")
        print(f"Output: {extreme_long:+.4f}x (capped at +3.0x)")
        print(f"Input:  -10.0x position")
        print(f"Output: {extreme_short:+.4f}x (capped at -3.0x)")

        # Validate capping
        assert extreme_long <= 3.0, \
            f"Long position should be capped at 3.0x, got {extreme_long:.4f}x"
        assert extreme_short >= -3.0, \
            f"Short position should be capped at -3.0x, got {extreme_short:.4f}x"

        # Test multiple extreme scenarios
        print("\nTesting 100 random extreme inputs...")
        violations = 0
        for _ in range(100):
            random_position = np.random.uniform(-50.0, 50.0)
            random_meta = np.random.uniform(0.0, 1.0)
            random_bandit = np.random.uniform(0.0, 1.0)
            random_regime = np.random.choice(['TRENDING', 'HIGH_VOL', 'LOW_VOL', 'NORMAL'])

            scaled = self.scaler.scale(
                position=random_position,
                meta_prob=random_meta,
                bandit_weight=random_bandit,
                regime=random_regime
            )

            if scaled > 3.0 or scaled < -3.0:
                violations += 1

        print(f"Violations: {violations}/100 (should be 0)")

        assert violations == 0, \
            f"Found {violations} position explosions exceeding ±3.0x limit"

        print("✅ PASS: All positions capped at ±3.0x, no explosions detected")
        return True


def run_sweep_y1():
    """
    Run SWEEP Y.1 validation suite.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n" + "="*70)
    print("PRADO9_EVO — SWEEP Y.1: Risk Scaling Validation")
    print("="*70)
    print("\nValidating Module Y: Position Scaling Engine")
    print("Testing: regime scaling, confidence adjustments, determinism, safety")
    print("")

    test_suite = TestSweepY1RiskScaling()

    tests = [
        ("Test 1: Trend Regime Increases Position", test_suite.test_1_trend_regime_increases_position),
        ("Test 2: High-Vol Regime Shrinks Position", test_suite.test_2_high_vol_regime_shrinks_position),
        ("Test 3: Meta Probability Scaling", test_suite.test_3_meta_probability_increases_with_confidence),
        ("Test 4: Bandit Weight Reduces Weak Strategies", test_suite.test_4_bandit_weight_reduces_weak_strategies),
        ("Test 5: Deterministic Behavior", test_suite.test_5_deterministic_behavior),
        ("Test 6: No Position Explosion", test_suite.test_6_no_position_explosion)
    ]

    passed = 0
    failed = 0
    failed_tests = []

    for test_name, test_func in tests:
        test_suite.setup_method()
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                failed_tests.append(test_name)
        except AssertionError as e:
            print(f"\n❌ FAIL: {test_name}")
            print(f"   Error: {e}")
            failed += 1
            failed_tests.append(test_name)
        except Exception as e:
            print(f"\n❌ ERROR: {test_name}")
            print(f"   Exception: {e}")
            failed += 1
            failed_tests.append(test_name)

    # Final summary
    print("\n" + "="*70)
    print("SWEEP Y.1 Results")
    print("="*70)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")
    print("")

    if failed == 0:
        print("✅ ALL TESTS PASSED - Module Y risk scaling validated")
        print("\nValidated:")
        print("  • Trend regimes increase position (1.4x)")
        print("  • High-vol regimes shrink position (0.7x)")
        print("  • Meta probability scales with confidence")
        print("  • Bandit weight reduces weak strategies")
        print("  • Deterministic behavior confirmed")
        print("  • Position explosion prevention (±3.0x cap)")
    else:
        print(f"❌ {failed} TESTS FAILED")
        print("\nFailed tests:")
        for test_name in failed_tests:
            print(f"  • {test_name}")

    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_sweep_y1()
    exit(0 if success else 1)
