"""
SWEEP V.1 — Volatility Strategy Engine Validation

Tests for Module V: Volatility-based trading strategies
"""

import sys
sys.path.insert(0, '/Users/darraykennedy/Desktop/python_pro/prado_evo/src')

from afml_system.volatility import VolatilityStrategies


def test_vol_breakout_signals():
    """Test 1: vol_breakout generates correct sides for different volatility levels."""
    print("\n=== Test 1: vol_breakout Signal Generation ===")
    vol_strats = VolatilityStrategies()

    # High volatility -> Long
    features_high = {"volatility": 0.025, "momentum": 0.01}
    signal = vol_strats.vol_breakout(features_high, "HIGH_VOL", "1D")
    assert signal.side == 1, f"Expected side=1 for high vol, got {signal.side}"
    assert signal.strategy_name == "vol_breakout"
    print(f"✓ High vol (2.5%): side={signal.side} (Long), prob={signal.probability}")

    # Low volatility -> Short
    features_low = {"volatility": 0.015, "momentum": 0.01}
    signal = vol_strats.vol_breakout(features_low, "HIGH_VOL", "1D")
    assert signal.side == -1, f"Expected side=-1 for low vol, got {signal.side}"
    print(f"✓ Low vol (1.5%): side={signal.side} (Short), prob={signal.probability}")

    print("✅ Test 1 PASSED")


def test_vol_spike_fade_signals():
    """Test 2: vol_spike_fade fades extreme spikes."""
    print("\n=== Test 2: vol_spike_fade Signal Generation ===")
    vol_strats = VolatilityStrategies()

    # Extreme spike -> Fade (Short)
    features_spike = {"volatility": 0.04, "momentum": 0.01}
    signal = vol_strats.vol_spike_fade(features_spike, "HIGH_VOL", "1D")
    assert signal.side == -1, f"Expected side=-1 for spike, got {signal.side}"
    print(f"✓ Spike (4.0%): side={signal.side} (Fade/Short), prob={signal.probability}")

    # Normal volatility -> Long
    features_normal = {"volatility": 0.02, "momentum": 0.01}
    signal = vol_strats.vol_spike_fade(features_normal, "HIGH_VOL", "1D")
    assert signal.side == 1, f"Expected side=1 for normal vol, got {signal.side}"
    print(f"✓ Normal (2.0%): side={signal.side} (Long), prob={signal.probability}")

    print("✅ Test 2 PASSED")


def test_vol_compression_signals():
    """Test 3: vol_compression anticipates breakout after compression."""
    print("\n=== Test 3: vol_compression Signal Generation ===")
    vol_strats = VolatilityStrategies()

    # Compressed volatility -> Long (anticipate expansion)
    features_compressed = {"volatility": 0.008, "momentum": 0.01}
    signal = vol_strats.vol_compression(features_compressed, "LOW_VOL", "1D")
    assert signal.side == 1, f"Expected side=1 for compression, got {signal.side}"
    print(f"✓ Compressed (0.8%): side={signal.side} (Long), prob={signal.probability}")

    # Normal volatility -> Short
    features_normal = {"volatility": 0.02, "momentum": 0.01}
    signal = vol_strats.vol_compression(features_normal, "LOW_VOL", "1D")
    assert signal.side == -1, f"Expected side=-1 for normal vol, got {signal.side}"
    print(f"✓ Normal (2.0%): side={signal.side} (Short), prob={signal.probability}")

    print("✅ Test 3 PASSED")


def test_vol_mean_revert_signals():
    """Test 4: vol_mean_revert mean-reverts volatility."""
    print("\n=== Test 4: vol_mean_revert Signal Generation ===")
    vol_strats = VolatilityStrategies()

    # High volatility -> Fade (Short)
    features_high = {"volatility": 0.03, "momentum": 0.01}
    signal = vol_strats.vol_mean_revert(features_high, "MEAN_REVERTING", "1D")
    assert signal.side == -1, f"Expected side=-1 for high vol, got {signal.side}"
    print(f"✓ High vol (3.0%): side={signal.side} (Short), prob={signal.probability}")

    # Low volatility -> Long
    features_low = {"volatility": 0.01, "momentum": 0.01}
    signal = vol_strats.vol_mean_revert(features_low, "MEAN_REVERTING", "1D")
    assert signal.side == 1, f"Expected side=1 for low vol, got {signal.side}"
    print(f"✓ Low vol (1.0%): side={signal.side} (Long), prob={signal.probability}")

    print("✅ Test 4 PASSED")


def test_trend_breakout_signals():
    """Test 5: trend_breakout trades strong momentum."""
    print("\n=== Test 5: trend_breakout Signal Generation ===")
    vol_strats = VolatilityStrategies()

    # Strong positive momentum -> Long
    features_bullish = {"volatility": 0.02, "momentum": 0.03}
    signal = vol_strats.trend_breakout(features_bullish, "TRENDING", "1D")
    assert signal.side == 1, f"Expected side=1 for positive momentum, got {signal.side}"
    print(f"✓ Bullish (3.0%): side={signal.side} (Long), prob={signal.probability}")

    # Strong negative momentum -> Short
    features_bearish = {"volatility": 0.02, "momentum": -0.03}
    signal = vol_strats.trend_breakout(features_bearish, "TRENDING", "1D")
    assert signal.side == -1, f"Expected side=-1 for negative momentum, got {signal.side}"
    print(f"✓ Bearish (-3.0%): side={signal.side} (Short), prob={signal.probability}")

    # Weak momentum -> Neutral
    features_neutral = {"volatility": 0.02, "momentum": 0.005}
    signal = vol_strats.trend_breakout(features_neutral, "TRENDING", "1D")
    assert signal.side == 0, f"Expected side=0 for weak momentum, got {signal.side}"
    print(f"✓ Neutral (0.5%): side={signal.side} (Neutral), prob={signal.probability}")

    print("✅ Test 5 PASSED")


def test_signal_structure():
    """Test 6: All signals have required fields."""
    print("\n=== Test 6: Signal Structure Validation ===")
    vol_strats = VolatilityStrategies()

    features = {"volatility": 0.02, "momentum": 0.01}

    strategies = [
        ("vol_breakout", lambda: vol_strats.vol_breakout(features, "HIGH_VOL", "1D")),
        ("vol_spike_fade", lambda: vol_strats.vol_spike_fade(features, "HIGH_VOL", "1D")),
        ("vol_compression", lambda: vol_strats.vol_compression(features, "LOW_VOL", "1D")),
        ("vol_mean_revert", lambda: vol_strats.vol_mean_revert(features, "MEAN_REVERTING", "1D")),
        ("trend_breakout", lambda: vol_strats.trend_breakout(features, "TRENDING", "1D")),
    ]

    for name, strat_func in strategies:
        signal = strat_func()

        assert hasattr(signal, 'strategy_name')
        assert hasattr(signal, 'regime')
        assert hasattr(signal, 'horizon')
        assert hasattr(signal, 'side')
        assert hasattr(signal, 'probability')
        assert hasattr(signal, 'meta_probability')
        assert hasattr(signal, 'forecast_return')
        assert hasattr(signal, 'volatility_forecast')
        assert hasattr(signal, 'bandit_weight')
        assert hasattr(signal, 'uniqueness')
        assert hasattr(signal, 'correlation_penalty')

        assert signal.strategy_name == name
        assert 0.0 <= signal.probability <= 1.0
        assert 0.0 <= signal.meta_probability <= 1.0

        print(f"✓ {name}: all fields present and valid")

    print("✅ Test 6 PASSED")


def test_determinism():
    """Test 7: Same inputs produce same signals."""
    print("\n=== Test 7: Determinism ===")
    vol_strats1 = VolatilityStrategies()
    vol_strats2 = VolatilityStrategies()

    features = {"volatility": 0.025, "momentum": 0.02}

    signal1 = vol_strats1.vol_breakout(features, "HIGH_VOL", "1D")
    signal2 = vol_strats2.vol_breakout(features, "HIGH_VOL", "1D")

    assert signal1.side == signal2.side
    assert signal1.probability == signal2.probability
    assert signal1.forecast_return == signal2.forecast_return
    assert signal1.volatility_forecast == signal2.volatility_forecast

    print(f"✓ vol_breakout: deterministic (side={signal1.side})")

    signal1 = vol_strats1.vol_compression(features, "LOW_VOL", "1D")
    signal2 = vol_strats2.vol_compression(features, "LOW_VOL", "1D")

    assert signal1.side == signal2.side
    assert signal1.probability == signal2.probability

    print(f"✓ vol_compression: deterministic (side={signal1.side})")

    print("✅ Test 7 PASSED")


def test_probability_ranges():
    """Test 8: Probabilities are reasonable and in valid range."""
    print("\n=== Test 8: Probability Ranges ===")
    vol_strats = VolatilityStrategies()

    features_list = [
        {"volatility": 0.01, "momentum": 0.01},
        {"volatility": 0.02, "momentum": 0.02},
        {"volatility": 0.03, "momentum": 0.03},
        {"volatility": 0.04, "momentum": -0.02},
    ]

    for features in features_list:
        signal = vol_strats.vol_breakout(features, "HIGH_VOL", "1D")
        assert 0.5 <= signal.probability <= 0.7, \
            f"vol_breakout probability {signal.probability} out of expected range [0.5, 0.7]"

        signal = vol_strats.vol_spike_fade(features, "HIGH_VOL", "1D")
        assert 0.5 <= signal.probability <= 0.7, \
            f"vol_spike_fade probability {signal.probability} out of expected range"

        signal = vol_strats.vol_compression(features, "LOW_VOL", "1D")
        assert 0.5 <= signal.probability <= 0.7, \
            f"vol_compression probability {signal.probability} out of expected range"

    print("✓ All probabilities in valid range [0.5, 0.7]")
    print("✅ Test 8 PASSED")


def test_uniqueness_scores():
    """Test 9: Uniqueness scores reflect strategy independence."""
    print("\n=== Test 9: Uniqueness Scores ===")
    vol_strats = VolatilityStrategies()

    features = {"volatility": 0.02, "momentum": 0.01}

    strategies = [
        vol_strats.vol_breakout(features, "HIGH_VOL", "1D"),
        vol_strats.vol_spike_fade(features, "HIGH_VOL", "1D"),
        vol_strats.vol_compression(features, "LOW_VOL", "1D"),
        vol_strats.vol_mean_revert(features, "MEAN_REVERTING", "1D"),
        vol_strats.trend_breakout(features, "TRENDING", "1D"),
    ]

    for signal in strategies:
        assert 0.0 <= signal.uniqueness <= 1.0, \
            f"{signal.strategy_name} uniqueness {signal.uniqueness} out of range [0, 1]"
        assert signal.uniqueness >= 0.5, \
            f"{signal.strategy_name} uniqueness {signal.uniqueness} too low (should be >= 0.5)"
        print(f"✓ {signal.strategy_name}: uniqueness={signal.uniqueness:.2f}")

    print("✅ Test 9 PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("SWEEP V.1 — Volatility Strategy Engine Tests")
    print("=" * 60)

    try:
        test_vol_breakout_signals()
        test_vol_spike_fade_signals()
        test_vol_compression_signals()
        test_vol_mean_revert_signals()
        test_trend_breakout_signals()
        test_signal_structure()
        test_determinism()
        test_probability_ranges()
        test_uniqueness_scores()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Module V Working Correctly")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
