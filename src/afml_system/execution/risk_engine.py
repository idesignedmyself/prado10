"""
PRADO9_EVO Module H — Risk Engine

Multi-layered risk firewall:
- Position limits
- Leverage limits
- Daily loss limits
- Volatility kill switch
- Correlation kill switch (Module F integration)
- Regime kill switch
- Large jump protection

Author: PRADO9_EVO Builder
Date: 2025-01-16
Version: 1.1.0 (Sweep H.1 - Institutional Grade)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, date

from .trade_types import TradeIntent, PortfolioState, _safe_float


# ============================================================================
# CONSTANTS
# ============================================================================

RISK_ENGINE_VERSION = '1.1.0'
EPSILON = 1e-12


# ============================================================================
# RISK ENGINE
# ============================================================================

class RiskEngine:
    """
    Multi-layered risk firewall.

    Enforces position limits, leverage limits, daily loss limits,
    and multiple kill switches.

    Sweep H.1 - Enhanced with correlation kill, regime kill, and jump protection.
    """

    def __init__(
        self,
        max_position: float = 1.0,
        max_leverage: float = 1.0,
        max_daily_loss: float = 0.05,
        vol_kill_multiplier: float = 3.0,
        conflict_ratio_threshold: float = 5.0,
        jump_protection_multiplier: float = 8.0
    ):
        """
        Initialize risk engine.

        Args:
            max_position: Maximum absolute position (default: 1.0)
            max_leverage: Maximum leverage ratio (default: 1.0)
            max_daily_loss: Maximum daily loss as fraction of equity (default: 0.05)
            vol_kill_multiplier: Volatility kill threshold multiplier (default: 3.0)
            conflict_ratio_threshold: Conflict ratio kill threshold (default: 5.0)
            jump_protection_multiplier: Price jump threshold multiplier (default: 8.0)
        """
        self.max_position = _safe_float(max_position, 1.0)
        self.max_leverage = _safe_float(max_leverage, 1.0)
        self.max_daily_loss = _safe_float(max_daily_loss, 0.05)
        self.vol_kill_multiplier = _safe_float(vol_kill_multiplier, 3.0)
        self.conflict_ratio_threshold = _safe_float(conflict_ratio_threshold, 5.0)
        self.jump_protection_multiplier = _safe_float(jump_protection_multiplier, 8.0)

        # Track daily P&L
        self.daily_pnl_tracker: Dict[date, float] = {}
        self.last_reset_date: Optional[date] = None

        # Sweep H.1 - Track last price for jump protection
        self.last_price: Optional[float] = None

    def check_position_limit(
        self,
        target_position: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if target position exceeds maximum.

        Args:
            target_position: Target position

        Returns:
            Tuple of (allowed, reason)
        """
        target_position = _safe_float(target_position, 0.0)

        if abs(target_position) > self.max_position:
            return False, f"Position limit exceeded: {abs(target_position):.4f} > {self.max_position:.4f}"

        return True, None

    def check_leverage_limit(
        self,
        target_position: float,
        current_price: float,
        equity: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if target position exceeds leverage limit.

        Args:
            target_position: Target position (shares/contracts)
            current_price: Current market price
            equity: Current equity

        Returns:
            Tuple of (allowed, reason)
        """
        target_position = _safe_float(target_position, 0.0)
        current_price = _safe_float(current_price, 0.0)
        equity = _safe_float(equity, 0.0)

        if equity <= EPSILON:
            return False, "Zero or negative equity"

        # Compute target leverage
        notional = abs(target_position * current_price)
        leverage = notional / equity

        if leverage > self.max_leverage:
            return False, f"Leverage limit exceeded: {leverage:.4f} > {self.max_leverage:.4f}"

        return True, None

    def check_daily_loss(
        self,
        current_date: date,
        portfolio_state: PortfolioState,
        initial_equity: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if daily loss limit is exceeded.

        Args:
            current_date: Current date
            portfolio_state: Current portfolio state
            initial_equity: Initial equity at start of day

        Returns:
            Tuple of (allowed, reason)
        """
        initial_equity = _safe_float(initial_equity, 0.0)

        if initial_equity <= EPSILON:
            return True, None  # Can't compute loss ratio

        # Reset daily tracker if new day
        if self.last_reset_date != current_date:
            self.daily_pnl_tracker[current_date] = 0.0
            self.last_reset_date = current_date

        # Compute daily loss
        current_equity = _safe_float(portfolio_state.equity, initial_equity)
        daily_loss = (initial_equity - current_equity) / initial_equity

        if daily_loss > self.max_daily_loss:
            return False, f"Daily loss limit exceeded: {daily_loss:.4f} > {self.max_daily_loss:.4f}"

        return True, None

    def check_volatility_kill_switch(
        self,
        current_volatility: float,
        long_run_volatility: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if volatility kill switch is triggered.

        Args:
            current_volatility: Current volatility estimate
            long_run_volatility: Long-run average volatility

        Returns:
            Tuple of (allowed, reason)
        """
        current_volatility = _safe_float(current_volatility, 0.0)
        long_run_volatility = _safe_float(long_run_volatility, 0.15)

        if long_run_volatility <= EPSILON:
            long_run_volatility = 0.15  # Default fallback

        threshold = self.vol_kill_multiplier * long_run_volatility

        if current_volatility > threshold:
            return False, f"Volatility kill triggered: {current_volatility:.4f} > {threshold:.4f}"

        return True, None

    def check_correlation_kill_switch(
        self,
        allocator_details: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if correlation kill switch is triggered (Sweep H.1).

        High conflict ratio indicates strategy disagreement.

        Args:
            allocator_details: Allocator diagnostics

        Returns:
            Tuple of (allowed, reason)
        """
        if allocator_details is None:
            return True, None

        conflict_ratio = allocator_details.get('conflict_ratio', 0.0)
        conflict_ratio = _safe_float(conflict_ratio, 0.0)

        if conflict_ratio > self.conflict_ratio_threshold:
            return False, f"High conflict ratio: {conflict_ratio:.4f} > {self.conflict_ratio_threshold:.4f}"

        return True, None

    def check_regime_kill_switch(
        self,
        current_volatility: float,
        long_run_volatility: float,
        regime: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if regime kill switch is triggered (Sweep H.1).

        Blocks trades when entering high volatility regime with elevated vol.

        Args:
            current_volatility: Current volatility estimate
            long_run_volatility: Long-run average volatility
            regime: Current regime (optional)

        Returns:
            Tuple of (allowed, reason)
        """
        current_volatility = _safe_float(current_volatility, 0.0)
        long_run_volatility = _safe_float(long_run_volatility, 0.15)

        if long_run_volatility <= EPSILON:
            long_run_volatility = 0.15

        # Check if in HIGH_VOL regime with vol > 2× long-run
        if regime and 'HIGH' in regime.upper() and 'VOL' in regime.upper():
            threshold = 2.0 * long_run_volatility
            if current_volatility > threshold:
                return False, f"Regime kill switch activated: HIGH_VOL regime with volatility {current_volatility:.4f} > {threshold:.4f}"

        return True, None

    def check_large_jump_protection(
        self,
        current_price: float,
        long_run_volatility: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if large price jump protection is triggered (Sweep H.1).

        Blocks trades when price jumps abnormally.

        Args:
            current_price: Current price
            long_run_volatility: Long-run average volatility

        Returns:
            Tuple of (allowed, reason)
        """
        current_price = _safe_float(current_price, 0.0)
        long_run_volatility = _safe_float(long_run_volatility, 0.15)

        if self.last_price is None:
            # First trade, no jump protection
            self.last_price = current_price
            return True, None

        if self.last_price <= EPSILON:
            # Can't compute percentage change
            self.last_price = current_price
            return True, None

        # Compute price change
        price_change = abs(current_price - self.last_price)
        expected_move = self.jump_protection_multiplier * long_run_volatility * self.last_price

        if price_change > expected_move:
            return False, f"Large price jump detected: {price_change:.4f} > {expected_move:.4f}"

        # Update last price
        self.last_price = current_price
        return True, None

    def enforce(
        self,
        trade_intent: TradeIntent,
        portfolio_state: PortfolioState,
        current_price: float,
        current_volatility: float,
        long_run_volatility: float,
        initial_equity: Optional[float] = None,
        allocator_details: Optional[Dict[str, Any]] = None,
        regime: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Enforce all risk checks (Sweep H.1 - Enhanced).

        Args:
            trade_intent: Desired trade
            portfolio_state: Current portfolio state
            current_price: Current market price
            current_volatility: Current volatility estimate
            long_run_volatility: Long-run average volatility
            initial_equity: Initial equity at start of day (optional)
            allocator_details: Allocator diagnostics (optional)
            regime: Current regime (optional)

        Returns:
            Tuple of (allowed, reason)
        """
        # Sanitize inputs
        current_price = _safe_float(current_price, 0.0)
        current_volatility = _safe_float(current_volatility, 0.0)
        long_run_volatility = _safe_float(long_run_volatility, 0.15)

        if initial_equity is None:
            initial_equity = portfolio_state.equity

        # Check position limit
        allowed, reason = self.check_position_limit(trade_intent.target_position)
        if not allowed:
            return False, reason

        # Check leverage limit
        allowed, reason = self.check_leverage_limit(
            trade_intent.target_position,
            current_price,
            portfolio_state.equity
        )
        if not allowed:
            return False, reason

        # Check daily loss
        current_date = trade_intent.timestamp.date() if isinstance(trade_intent.timestamp, datetime) else date.today()
        allowed, reason = self.check_daily_loss(
            current_date,
            portfolio_state,
            initial_equity
        )
        if not allowed:
            return False, reason

        # Check volatility kill switch
        allowed, reason = self.check_volatility_kill_switch(
            current_volatility,
            long_run_volatility
        )
        if not allowed:
            return False, reason

        # Sweep H.1 - Check correlation kill switch
        if allocator_details is None:
            allocator_details = trade_intent.allocator_details

        allowed, reason = self.check_correlation_kill_switch(allocator_details)
        if not allowed:
            return False, reason

        # Sweep H.1 - Check regime kill switch
        allowed, reason = self.check_regime_kill_switch(
            current_volatility,
            long_run_volatility,
            regime
        )
        if not allowed:
            return False, reason

        # Sweep H.1 - Check large jump protection
        allowed, reason = self.check_large_jump_protection(
            current_price,
            long_run_volatility
        )
        if not allowed:
            return False, reason

        # All checks passed
        return True, None
