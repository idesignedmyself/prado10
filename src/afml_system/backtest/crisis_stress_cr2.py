"""
PRADO9_EVO Module CR2 — Crisis Backtest Expansion

Enhanced crisis period detection and synthetic crisis generation.

Features:
1. Multi-crisis regime detector (recognizes 2008, 2020, 2022 patterns)
2. Synthetic crisis generator (creates realistic crash scenarios)
3. Pattern-based crisis classification
4. Volatility signature matching

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


# ============================================================================
# CONSTANTS
# ============================================================================

CR2_VERSION = '1.0.0'
EPSILON = 1e-12


# ============================================================================
# ENUMS
# ============================================================================

class CrisisType(Enum):
    """Types of financial crises."""
    FLASH_CRASH = "FLASH_CRASH"           # Rapid single-day crash (2010-style)
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"  # Credit freeze (2008-style)
    PANDEMIC_SHOCK = "PANDEMIC_SHOCK"     # Sudden global shock (2020-style)
    BEAR_MARKET = "BEAR_MARKET"           # Slow grind down (2022-style)
    DOT_COM_BURST = "DOT_COM_BURST"       # Tech bubble pop (2000-style)
    SOVEREIGN_DEBT = "SOVEREIGN_DEBT"     # European debt crisis (2011-style)
    UNKNOWN = "UNKNOWN"                   # Unclassified crisis


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CrisisSignature:
    """Statistical signature of a crisis type."""
    crisis_type: CrisisType
    duration_days: Tuple[int, int]  # (min, max) duration
    peak_vol_mult: Tuple[float, float]  # (min, max) volatility multiplier
    drawdown_range: Tuple[float, float]  # (min, max) drawdown
    recovery_days: Tuple[int, int]  # (min, max) recovery time
    correlation_breakdown: bool  # Whether correlations break down
    liquidity_dry_up: bool  # Whether liquidity disappears


# Known crisis signatures for pattern matching
CRISIS_SIGNATURES = {
    CrisisType.FLASH_CRASH: CrisisSignature(
        crisis_type=CrisisType.FLASH_CRASH,
        duration_days=(1, 5),
        peak_vol_mult=(5.0, 10.0),
        drawdown_range=(-0.10, -0.05),
        recovery_days=(5, 20),
        correlation_breakdown=False,
        liquidity_dry_up=True
    ),
    CrisisType.LIQUIDITY_CRISIS: CrisisSignature(
        crisis_type=CrisisType.LIQUIDITY_CRISIS,
        duration_days=(90, 180),
        peak_vol_mult=(3.0, 5.0),
        drawdown_range=(-0.55, -0.35),
        recovery_days=(180, 540),
        correlation_breakdown=True,
        liquidity_dry_up=True
    ),
    CrisisType.PANDEMIC_SHOCK: CrisisSignature(
        crisis_type=CrisisType.PANDEMIC_SHOCK,
        duration_days=(30, 90),
        peak_vol_mult=(4.0, 8.0),
        drawdown_range=(-0.40, -0.25),
        recovery_days=(90, 180),
        correlation_breakdown=True,
        liquidity_dry_up=True
    ),
    CrisisType.BEAR_MARKET: CrisisSignature(
        crisis_type=CrisisType.BEAR_MARKET,
        duration_days=(180, 365),
        peak_vol_mult=(1.5, 2.5),
        drawdown_range=(-0.30, -0.15),
        recovery_days=(180, 365),
        correlation_breakdown=False,
        liquidity_dry_up=False
    ),
}


@dataclass
class DetectedCrisis:
    """A detected crisis period with classification."""
    name: str
    crisis_type: CrisisType
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    duration_days: int
    max_drawdown: float
    peak_volatility: float
    vol_multiplier: float
    recovery_days: int
    match_confidence: float  # 0-1 confidence in classification


# ============================================================================
# MULTI-CRISIS REGIME DETECTOR
# ============================================================================

class MultiCrisisDetector:
    """
    Enhanced crisis detector that recognizes multiple crisis types.

    Detects and classifies crises by matching volatility and drawdown
    patterns against known crisis signatures (2008, 2020, 2022, etc.).

    Example:
        >>> detector = MultiCrisisDetector()
        >>> crises = detector.detect_crises(df)
        >>> for crisis in crises:
        ...     print(f"{crisis.name}: {crisis.crisis_type.value}")
    """

    def __init__(
        self,
        vol_threshold_multiplier: float = 2.0,
        min_crisis_duration: int = 5,
        max_crises_to_detect: int = 10
    ):
        """
        Initialize Multi-Crisis Detector.

        Args:
            vol_threshold_multiplier: Volatility must be this many times median
            min_crisis_duration: Minimum days for crisis (filter noise)
            max_crises_to_detect: Maximum number of crises to return
        """
        self.vol_threshold_multiplier = vol_threshold_multiplier
        self.min_crisis_duration = min_crisis_duration
        self.max_crises_to_detect = max_crises_to_detect

    def detect_crises(self, df: pd.DataFrame) -> List[DetectedCrisis]:
        """
        Detect and classify crisis periods from OHLCV data.

        Args:
            df: OHLCV DataFrame with datetime index

        Returns:
            List of DetectedCrisis objects
        """
        # Compute returns and volatility
        returns = df['close'].pct_change(fill_method=None).fillna(0)
        vol_window = 20
        rolling_vol = returns.rolling(window=vol_window).std()

        # Compute baseline volatility (median)
        baseline_vol = rolling_vol.median()

        # Find high-volatility periods
        high_vol_threshold = self.vol_threshold_multiplier * baseline_vol
        is_high_vol = rolling_vol > high_vol_threshold

        # Find contiguous crisis periods
        crisis_periods = self._find_contiguous_periods(df, is_high_vol)

        # Classify each crisis
        detected_crises = []
        for i, (start_idx, end_idx) in enumerate(crisis_periods):
            crisis = self._classify_crisis(
                df=df,
                start_idx=start_idx,
                end_idx=end_idx,
                crisis_num=i+1,
                baseline_vol=baseline_vol
            )

            if crisis is not None:
                detected_crises.append(crisis)

        # Limit to max_crises_to_detect
        return detected_crises[:self.max_crises_to_detect]

    def _find_contiguous_periods(
        self,
        df: pd.DataFrame,
        condition: pd.Series
    ) -> List[Tuple[int, int]]:
        """
        Find contiguous periods where condition is True.

        Args:
            df: DataFrame
            condition: Boolean series

        Returns:
            List of (start_idx, end_idx) tuples
        """
        periods = []
        in_period = False
        start_idx = None

        for i in range(len(condition)):
            if condition.iloc[i] and not in_period:
                # Period starts
                start_idx = i
                in_period = True
            elif not condition.iloc[i] and in_period:
                # Period ends
                end_idx = i

                # Check minimum duration
                if end_idx - start_idx >= self.min_crisis_duration:
                    periods.append((start_idx, end_idx))

                in_period = False

        # Handle case where period extends to end of data
        if in_period and start_idx is not None:
            end_idx = len(condition)
            if end_idx - start_idx >= self.min_crisis_duration:
                periods.append((start_idx, end_idx))

        return periods

    def _classify_crisis(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        crisis_num: int,
        baseline_vol: float
    ) -> Optional[DetectedCrisis]:
        """
        Classify a crisis period by matching against known signatures.

        Args:
            df: OHLCV DataFrame
            start_idx: Start index of crisis
            end_idx: End index of crisis
            crisis_num: Crisis number (for naming)
            baseline_vol: Baseline volatility for comparison

        Returns:
            DetectedCrisis or None if classification fails
        """
        # Extract crisis data
        crisis_df = df.iloc[start_idx:end_idx]

        if len(crisis_df) < 2:
            return None

        # Compute crisis statistics
        returns = crisis_df['close'].pct_change(fill_method=None).fillna(0)
        crisis_vol = returns.std()
        vol_multiplier = crisis_vol / baseline_vol if baseline_vol > EPSILON else 1.0

        # Compute drawdown
        equity = (1 + returns).cumprod()
        running_max = equity.expanding().max()
        drawdown = ((equity - running_max) / running_max)
        max_drawdown = drawdown.min()

        # Duration
        duration_days = (crisis_df.index[-1] - crisis_df.index[0]).days

        # Estimate recovery time (simplified)
        recovery_days = int(duration_days * 1.5)  # Rough estimate

        # Match against known signatures
        crisis_type, confidence = self._match_signature(
            duration_days=duration_days,
            vol_multiplier=vol_multiplier,
            max_drawdown=max_drawdown,
            recovery_days=recovery_days
        )

        # Generate name based on crisis type and date
        year = crisis_df.index[0].year
        month = crisis_df.index[0].strftime('%b')

        if crisis_type != CrisisType.UNKNOWN:
            name = f"{year} {month} {crisis_type.value.replace('_', ' ').title()}"
        else:
            name = f"Crisis {crisis_num} ({year}-{month})"

        return DetectedCrisis(
            name=name,
            crisis_type=crisis_type,
            start_date=crisis_df.index[0],
            end_date=crisis_df.index[-1],
            duration_days=duration_days,
            max_drawdown=max_drawdown,
            peak_volatility=crisis_vol * np.sqrt(252),  # Annualized
            vol_multiplier=vol_multiplier,
            recovery_days=recovery_days,
            match_confidence=confidence
        )

    def _match_signature(
        self,
        duration_days: int,
        vol_multiplier: float,
        max_drawdown: float,
        recovery_days: int
    ) -> Tuple[CrisisType, float]:
        """
        Match crisis characteristics against known signatures.

        Args:
            duration_days: Crisis duration
            vol_multiplier: Volatility multiplier vs baseline
            max_drawdown: Maximum drawdown
            recovery_days: Estimated recovery time

        Returns:
            (CrisisType, confidence) tuple
        """
        best_match = CrisisType.UNKNOWN
        best_score = 0.0

        for crisis_type, signature in CRISIS_SIGNATURES.items():
            # Score each dimension (0-1)
            duration_score = self._score_range(
                duration_days,
                signature.duration_days[0],
                signature.duration_days[1]
            )

            vol_score = self._score_range(
                vol_multiplier,
                signature.peak_vol_mult[0],
                signature.peak_vol_mult[1]
            )

            dd_score = self._score_range(
                max_drawdown,
                signature.drawdown_range[0],
                signature.drawdown_range[1]
            )

            recovery_score = self._score_range(
                recovery_days,
                signature.recovery_days[0],
                signature.recovery_days[1]
            )

            # Weighted average (vol and drawdown are most important)
            total_score = (
                0.2 * duration_score +
                0.3 * vol_score +
                0.4 * dd_score +
                0.1 * recovery_score
            )

            if total_score > best_score:
                best_score = total_score
                best_match = crisis_type

        # Confidence threshold
        confidence = best_score
        if confidence < 0.5:
            # Poor match - classify as UNKNOWN
            best_match = CrisisType.UNKNOWN
            confidence = 0.0

        return best_match, confidence

    def _score_range(
        self,
        value: float,
        min_val: float,
        max_val: float
    ) -> float:
        """
        Score how well a value fits in a range (0-1).

        Args:
            value: Value to score
            min_val: Minimum of range
            max_val: Maximum of range

        Returns:
            Score from 0 (poor fit) to 1 (perfect fit)
        """
        if min_val <= value <= max_val:
            # Inside range - perfect score
            return 1.0
        elif value < min_val:
            # Below range - score based on distance
            distance = min_val - value
            range_width = max_val - min_val
            return max(0.0, 1.0 - distance / range_width)
        else:
            # Above range - score based on distance
            distance = value - max_val
            range_width = max_val - min_val
            return max(0.0, 1.0 - distance / range_width)


# ============================================================================
# SYNTHETIC CRISIS GENERATOR
# ============================================================================

class SyntheticCrisisGenerator:
    """
    Generate synthetic crisis scenarios for robustness testing.

    Creates realistic OHLCV data that mimics historical crises
    (2008, 2020, 2022) for stress testing strategies.

    Example:
        >>> generator = SyntheticCrisisGenerator(seed=42)
        >>> crisis_df = generator.generate_crisis(
        ...     baseline_df=df,
        ...     crisis_type=CrisisType.PANDEMIC_SHOCK,
        ...     severity=1.0
        ... )
    """

    def __init__(self, seed: int = 42):
        """
        Initialize Synthetic Crisis Generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_crisis(
        self,
        baseline_df: pd.DataFrame,
        crisis_type: CrisisType = CrisisType.PANDEMIC_SHOCK,
        severity: float = 1.0,
        duration_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic crisis data.

        Args:
            baseline_df: Baseline OHLCV data (for realistic scaling)
            crisis_type: Type of crisis to generate
            severity: Crisis severity multiplier (0.5-2.0)
            duration_days: Override default duration (optional)

        Returns:
            OHLCV DataFrame with synthetic crisis
        """
        # Reset seed for reproducibility
        np.random.seed(self.seed)

        # Get crisis signature
        if crisis_type in CRISIS_SIGNATURES:
            signature = CRISIS_SIGNATURES[crisis_type]
        else:
            # Default to pandemic shock
            signature = CRISIS_SIGNATURES[CrisisType.PANDEMIC_SHOCK]

        # Determine duration
        if duration_days is None:
            duration_days = int(np.mean(signature.duration_days))

        # Adjust for severity
        target_drawdown = np.mean(signature.drawdown_range) * severity
        target_vol_mult = np.mean(signature.peak_vol_mult) * severity

        # Compute baseline statistics
        baseline_returns = baseline_df['close'].pct_change(fill_method=None).dropna()
        baseline_vol = baseline_returns.std()
        baseline_mean = baseline_returns.mean()

        # Generate crisis returns
        crisis_returns = self._generate_crisis_returns(
            duration_days=duration_days,
            target_drawdown=target_drawdown,
            target_vol_mult=target_vol_mult,
            baseline_vol=baseline_vol,
            baseline_mean=baseline_mean,
            crisis_type=crisis_type
        )

        # Reconstruct OHLCV data
        crisis_df = self._reconstruct_ohlcv(
            returns=crisis_returns,
            initial_price=baseline_df['close'].iloc[-1],
            start_date=baseline_df.index[-1] + timedelta(days=1)
        )

        return crisis_df

    def _generate_crisis_returns(
        self,
        duration_days: int,
        target_drawdown: float,
        target_vol_mult: float,
        baseline_vol: float,
        baseline_mean: float,
        crisis_type: CrisisType
    ) -> np.ndarray:
        """
        Generate crisis return series.

        Args:
            duration_days: Crisis duration
            target_drawdown: Target max drawdown
            target_vol_mult: Target volatility multiplier
            baseline_vol: Baseline volatility
            baseline_mean: Baseline mean return
            crisis_type: Type of crisis

        Returns:
            Array of daily returns
        """
        # Crisis-specific return patterns
        if crisis_type == CrisisType.FLASH_CRASH:
            # Rapid single-day crash
            returns = self._flash_crash_pattern(
                duration_days, target_drawdown, baseline_vol, target_vol_mult
            )

        elif crisis_type == CrisisType.LIQUIDITY_CRISIS:
            # 2008-style: gradual then accelerating decline
            returns = self._liquidity_crisis_pattern(
                duration_days, target_drawdown, baseline_vol, target_vol_mult
            )

        elif crisis_type == CrisisType.PANDEMIC_SHOCK:
            # 2020-style: sudden sharp drop then volatile recovery
            returns = self._pandemic_shock_pattern(
                duration_days, target_drawdown, baseline_vol, target_vol_mult
            )

        elif crisis_type == CrisisType.BEAR_MARKET:
            # 2022-style: slow grind down with periodic rallies
            returns = self._bear_market_pattern(
                duration_days, target_drawdown, baseline_vol, target_vol_mult
            )

        else:
            # Generic crisis: scaled random walk
            returns = np.random.randn(duration_days) * baseline_vol * target_vol_mult
            returns[0] = target_drawdown / 2  # Initial shock

        return returns

    def _flash_crash_pattern(
        self,
        duration_days: int,
        target_drawdown: float,
        baseline_vol: float,
        vol_mult: float
    ) -> np.ndarray:
        """Generate flash crash return pattern."""
        returns = np.zeros(duration_days)

        # Day 1: Massive drop
        returns[0] = target_drawdown

        # Days 2-5: High volatility recovery
        for i in range(1, min(5, duration_days)):
            returns[i] = np.random.randn() * baseline_vol * vol_mult * 0.5

        # Remaining days: Normal volatility
        if duration_days > 5:
            returns[5:] = np.random.randn(duration_days - 5) * baseline_vol

        return returns

    def _liquidity_crisis_pattern(
        self,
        duration_days: int,
        target_drawdown: float,
        baseline_vol: float,
        vol_mult: float
    ) -> np.ndarray:
        """Generate 2008-style liquidity crisis pattern."""
        returns = np.zeros(duration_days)

        # Phase 1 (first 30%): Gradual decline
        phase1_len = int(duration_days * 0.3)
        phase1_dd = target_drawdown * 0.3
        returns[:phase1_len] = np.linspace(0, phase1_dd, phase1_len)
        returns[:phase1_len] += np.random.randn(phase1_len) * baseline_vol * 1.5

        # Phase 2 (next 40%): Accelerating decline
        phase2_len = int(duration_days * 0.4)
        phase2_dd = target_drawdown * 0.5
        returns[phase1_len:phase1_len+phase2_len] = np.linspace(
            phase1_dd, phase1_dd + phase2_dd, phase2_len
        )
        returns[phase1_len:phase1_len+phase2_len] += (
            np.random.randn(phase2_len) * baseline_vol * vol_mult
        )

        # Phase 3 (remaining): Volatile bottoming
        phase3_start = phase1_len + phase2_len
        if phase3_start < duration_days:
            returns[phase3_start:] = (
                np.random.randn(duration_days - phase3_start) * baseline_vol * vol_mult * 0.7
            )

        return returns

    def _pandemic_shock_pattern(
        self,
        duration_days: int,
        target_drawdown: float,
        baseline_vol: float,
        vol_mult: float
    ) -> np.ndarray:
        """Generate 2020-style pandemic shock pattern."""
        returns = np.zeros(duration_days)

        # Phase 1 (first 20%): Sudden crash
        phase1_len = max(5, int(duration_days * 0.2))
        returns[:phase1_len] = np.linspace(0, target_drawdown, phase1_len)
        returns[:phase1_len] += np.random.randn(phase1_len) * baseline_vol * vol_mult

        # Phase 2 (remaining): Volatile V-shaped recovery
        phase2_start = phase1_len
        if phase2_start < duration_days:
            # Recovery with high volatility
            recovery = np.linspace(target_drawdown, 0, duration_days - phase2_start)
            noise = np.random.randn(duration_days - phase2_start) * baseline_vol * vol_mult * 0.6
            returns[phase2_start:] = recovery + noise

        return returns

    def _bear_market_pattern(
        self,
        duration_days: int,
        target_drawdown: float,
        baseline_vol: float,
        vol_mult: float
    ) -> np.ndarray:
        """Generate 2022-style bear market pattern."""
        returns = np.zeros(duration_days)

        # Slow grind down with periodic relief rallies
        downtrend = np.linspace(0, target_drawdown, duration_days)

        # Add periodic rallies (every ~30 days)
        rally_frequency = 30
        for i in range(0, duration_days, rally_frequency):
            rally_len = min(5, duration_days - i)
            if rally_len > 0:
                # Small relief rally
                downtrend[i:i+rally_len] += np.linspace(0, 0.05, rally_len)

        # Add noise
        noise = np.random.randn(duration_days) * baseline_vol * vol_mult

        returns = downtrend + noise

        return returns

    def _reconstruct_ohlcv(
        self,
        returns: np.ndarray,
        initial_price: float,
        start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Reconstruct OHLCV data from returns.

        Args:
            returns: Daily returns
            initial_price: Starting price
            start_date: Starting date

        Returns:
            OHLCV DataFrame
        """
        # Compute close prices
        price_mult = (1 + returns).cumprod()
        close_prices = initial_price * price_mult

        # Generate OHLV from close
        n = len(returns)
        dates = pd.date_range(start=start_date, periods=n, freq='D')

        # Simple OHLV generation
        high_prices = close_prices * (1 + np.abs(np.random.randn(n) * 0.01))
        low_prices = close_prices * (1 - np.abs(np.random.randn(n) * 0.01))
        open_prices = close_prices * (1 + np.random.randn(n) * 0.005)
        volume = np.random.randint(1000000, 10000000, n)

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)

        return df


# ============================================================================
# ENHANCED CRISIS STRESS ENGINE
# ============================================================================

class EnhancedCrisisStressEngine:
    """
    Enhanced crisis stress engine with multi-crisis detection and
    synthetic crisis generation (Module CR2).

    Example:
        >>> engine = EnhancedCrisisStressEngine()
        >>> result = engine.run_multi_crisis_test(symbol, df)
    """

    def __init__(self, seed: int = 42):
        """
        Initialize Enhanced Crisis Stress Engine.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.detector = MultiCrisisDetector()
        self.generator = SyntheticCrisisGenerator(seed=seed)

    def detect_and_classify_crises(
        self,
        df: pd.DataFrame
    ) -> List[DetectedCrisis]:
        """
        Detect and classify all crises in the data.

        Args:
            df: OHLCV DataFrame

        Returns:
            List of DetectedCrisis objects
        """
        return self.detector.detect_crises(df)

    def generate_synthetic_crisis(
        self,
        baseline_df: pd.DataFrame,
        crisis_type: CrisisType,
        severity: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate synthetic crisis data.

        Args:
            baseline_df: Baseline data for scaling
            crisis_type: Type of crisis to generate
            severity: Severity multiplier

        Returns:
            Synthetic OHLCV DataFrame
        """
        return self.generator.generate_crisis(
            baseline_df=baseline_df,
            crisis_type=crisis_type,
            severity=severity
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"EnhancedCrisisStressEngine(seed={self.seed}, version={CR2_VERSION})"
