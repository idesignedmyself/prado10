"""
PRADO9_EVO Module H — Execution Engine

Market simulator with slippage, commissions, and risk controls.

Components:
- ExecutionEngine: Core execution logic
- ExecutionManager: Full lifecycle manager
- evo_execute: Integration hook

Author: PRADO9_EVO Builder
Date: 2025-01-16
Version: 1.1.0 (Sweep H.1 - Institutional Grade)
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from .trade_types import TradeIntent, TradeFill, PortfolioState, _safe_float
from .risk_engine import RiskEngine


# ============================================================================
# CONSTANTS
# ============================================================================

EXECUTION_ENGINE_VERSION = '1.1.0'
EPSILON = 1e-12


# ============================================================================
# EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """
    Core execution engine (Sweep H.1 - Enhanced).

    Simulates market execution with volatility-adjusted slippage and commissions.
    """

    def __init__(
        self,
        slippage_bps: float = 1.0,
        commission_bps: float = 0.1,
        max_position: float = 1.0,
        max_leverage: float = 1.0
    ):
        """
        Initialize execution engine.

        Args:
            slippage_bps: Base slippage in basis points (default: 1.0)
            commission_bps: Commission in basis points (default: 0.1)
            max_position: Maximum absolute position (default: 1.0)
            max_leverage: Maximum leverage ratio (default: 1.0)
        """
        self.slippage_bps = _safe_float(slippage_bps, 1.0)
        self.commission_bps = _safe_float(commission_bps, 0.1)
        self.max_position = _safe_float(max_position, 1.0)
        self.max_leverage = _safe_float(max_leverage, 1.0)

    def compute_target_notional(
        self,
        target_position: float,
        equity: float
    ) -> float:
        """
        Compute target notional from target position.

        Args:
            target_position: Target position [-1, +1] or absolute
            equity: Current equity

        Returns:
            Target notional value
        """
        target_position = _safe_float(target_position, 0.0)
        equity = _safe_float(equity, 0.0)

        return target_position * equity

    def compute_trade_size(
        self,
        target_position: float,
        current_position: float,
        equity: float,
        price: float
    ) -> float:
        """
        Compute trade size (shares/contracts) needed.

        Args:
            target_position: Target position
            current_position: Current position
            equity: Current equity
            price: Current price

        Returns:
            Trade size (positive = buy, negative = sell)
        """
        target_position = _safe_float(target_position, 0.0)
        current_position = _safe_float(current_position, 0.0)
        equity = _safe_float(equity, 0.0)
        price = _safe_float(price, 0.0)

        if price <= EPSILON:
            return 0.0

        # Target position in shares
        target_shares = (target_position * equity) / price

        # Trade size = change in position
        trade_size = target_shares - current_position

        return trade_size

    def compute_slippage(
        self,
        price: float,
        size: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Compute volatility-adjusted slippage cost (Sweep H.1).

        Args:
            price: Base price
            size: Trade size
            volatility: Current volatility (optional)

        Returns:
            Slippage cost (positive = cost)
        """
        price = _safe_float(price, 0.0)
        size = _safe_float(size, 0.0)

        # Base slippage
        base_slippage = price * (self.slippage_bps / 10000.0)

        # Volatility component (Sweep H.1)
        if volatility is not None:
            volatility = _safe_float(volatility, 0.0)
            vol_component = price * (volatility * 0.1)
        else:
            vol_component = 0.0

        # Total slippage per share
        slippage_per_share = base_slippage + vol_component

        # Total slippage cost
        slippage_cost = abs(size) * slippage_per_share

        # Clip to max 5% of notional (Sweep H.1)
        notional = abs(size * price)
        max_slippage = notional * 0.05
        slippage_cost = min(slippage_cost, max_slippage)

        return slippage_cost

    def compute_commission(
        self,
        size: float,
        price: float
    ) -> float:
        """
        Compute commission cost with min/max caps (Sweep H.1).

        Args:
            size: Trade size
            price: Base price

        Returns:
            Commission cost (positive = cost)
        """
        size = _safe_float(size, 0.0)
        price = _safe_float(price, 0.0)

        # Base commission on notional value
        notional = abs(size * price)

        # Skip commission for zero or near-zero trades
        if notional < EPSILON:
            return 0.0

        commission = notional * (self.commission_bps / 10000.0)

        # Apply min/max caps (Sweep H.1)
        commission = max(commission, 0.01)  # Minimum $0.01
        commission = min(commission, notional * 0.005)  # Cap at 0.5% of notional

        return commission

    def apply_risk_limits(
        self,
        target_position: float,
        current_position: float,
        equity: float,
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Apply risk limits.

        Args:
            target_position: Target position
            current_position: Current position
            equity: Current equity
            price: Current price

        Returns:
            Tuple of (allowed, reason)
        """
        target_position = _safe_float(target_position, 0.0)
        equity = _safe_float(equity, 0.0)
        price = _safe_float(price, 0.0)

        # Check position limit
        if abs(target_position) > self.max_position:
            return False, f"Position limit exceeded: {abs(target_position):.4f} > {self.max_position:.4f}"

        # Check leverage limit
        if equity > EPSILON and price > EPSILON:
            target_shares = (target_position * equity) / price
            notional = abs(target_shares * price)
            leverage = notional / equity

            if leverage > self.max_leverage:
                return False, f"Leverage limit exceeded: {leverage:.4f} > {self.max_leverage:.4f}"

        return True, None

    def execute(
        self,
        trade_intent: TradeIntent,
        portfolio_state: PortfolioState,
        price: float,
        volatility: Optional[float] = None
    ) -> Tuple[TradeFill, PortfolioState]:
        """
        Execute trade (Sweep H.1 - Enhanced).

        Args:
            trade_intent: Desired trade
            portfolio_state: Current portfolio state
            price: Current market price
            volatility: Current volatility (optional, for slippage calculation)

        Returns:
            Tuple of (TradeFill, updated PortfolioState)
        """
        price = _safe_float(price, 0.0)

        # Compute trade size
        trade_size = self.compute_trade_size(
            trade_intent.target_position,
            portfolio_state.position,
            portfolio_state.equity,
            price
        )

        # Determine side
        if trade_size > 0:
            side = 'BUY'
        elif trade_size < 0:
            side = 'SELL'
        else:
            side = 'BUY'  # Default for zero trades

        # Compute slippage and commission (Sweep H.1 - volatility-adjusted)
        slippage = self.compute_slippage(price, trade_size, volatility)
        commission = self.compute_commission(trade_size, price)

        # Final execution price
        if side == 'BUY':
            final_price = price + (slippage / abs(trade_size)) if abs(trade_size) > EPSILON else price
        else:
            final_price = price - (slippage / abs(trade_size)) if abs(trade_size) > EPSILON else price

        final_price = _safe_float(final_price, price)

        # Compute notional
        notional = abs(trade_size * price)

        # Create trade fill (Sweep H.1 - with enhanced diagnostics)
        trade_fill = TradeFill(
            timestamp=trade_intent.timestamp,
            symbol=trade_intent.symbol,
            side=side,
            size=trade_size,
            price=price,
            slippage=slippage,
            commission=commission,
            final_price=final_price,
            notional=notional,
            post_trade_position=0.0,  # Will be updated below
            post_trade_equity=0.0,  # Will be updated below
            pnl_realized=0.0,  # Will be updated below
            pnl_unrealized=0.0,  # Will be updated below
            risk_flags=[]
        )

        # Update portfolio state
        updated_state = self.portfolio_update(
            portfolio_state,
            trade_fill,
            price
        )

        # Populate post-trade diagnostics (Sweep H.1)
        trade_fill.post_trade_position = updated_state.position
        trade_fill.post_trade_equity = updated_state.equity
        trade_fill.pnl_unrealized = updated_state.unrealized_pnl(price)

        # Realized P&L from this trade (commission is always a cost)
        trade_fill.pnl_realized = -commission

        return trade_fill, updated_state

    def portfolio_update(
        self,
        portfolio_state: PortfolioState,
        trade_fill: TradeFill,
        current_price: float
    ) -> PortfolioState:
        """
        Update portfolio state after trade (Sweep H.1 - Enhanced).

        Args:
            portfolio_state: Current portfolio state
            trade_fill: Executed trade
            current_price: Current market price

        Returns:
            Updated portfolio state
        """
        # Create copy
        new_state = portfolio_state.copy()

        # Update position
        new_position = portfolio_state.position + trade_fill.size
        new_state.position = _safe_float(new_position, 0.0)

        # Update cash (cost includes slippage and commission)
        trade_cost = trade_fill.size * trade_fill.final_price + trade_fill.commission
        new_cash = portfolio_state.cash - trade_cost
        new_state.cash = _safe_float(new_cash, 0.0)

        # Update equity
        current_price = _safe_float(current_price, 0.0)
        new_equity = new_state.cash + new_state.position * current_price
        new_state.equity = _safe_float(new_equity, 0.0)

        # Sweep H.1 - Prevent divide-by-zero
        new_state.equity = max(new_state.equity, 1e-6)

        # Sweep H.1 - Check for zero-crossing
        old_position = portfolio_state.position
        crossed_zero = False
        if abs(old_position) > EPSILON and abs(new_position) > EPSILON:
            # Check if sign changed
            if (old_position > 0 and new_position < 0) or (old_position < 0 and new_position > 0):
                crossed_zero = True

        # Update entry price (Sweep H.1 - Enhanced with zero-crossing reset)
        if abs(trade_fill.size) > EPSILON:
            if portfolio_state.entry_price is None or crossed_zero:
                # First trade or crossed zero - reset entry price
                new_state.entry_price = trade_fill.final_price
            else:
                # Weighted average
                old_notional = portfolio_state.position * portfolio_state.entry_price
                new_notional = trade_fill.size * trade_fill.final_price
                total_position = new_state.position

                if abs(total_position) > EPSILON:
                    new_state.entry_price = (old_notional + new_notional) / total_position
                else:
                    new_state.entry_price = None  # Flat position

        # Add to trade history
        new_state.trade_history = portfolio_state.trade_history.copy()
        new_state.trade_history.append(trade_fill)

        # Update timestamp
        new_state.timestamp = trade_fill.timestamp

        return new_state


# ============================================================================
# EXECUTION MANAGER
# ============================================================================

class ExecutionManager:
    """
    Full lifecycle execution manager (Sweep H.1 - Enhanced).

    Combines ExecutionEngine and RiskEngine for complete trade lifecycle.
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        risk_engine: RiskEngine
    ):
        """
        Initialize execution manager.

        Args:
            execution_engine: Execution engine instance
            risk_engine: Risk engine instance
        """
        self.execution_engine = execution_engine
        self.risk_engine = risk_engine

    def process(
        self,
        symbol: str,
        price: float,
        volatility: float,
        trade_intent: TradeIntent,
        portfolio_state: PortfolioState,
        long_run_volatility: Optional[float] = None,
        initial_equity: Optional[float] = None,
        regime: Optional[str] = None
    ) -> Tuple[PortfolioState, TradeFill]:
        """
        Process trade intent through full lifecycle (Sweep H.1 - Enhanced).

        Args:
            symbol: Trading symbol
            price: Current market price
            volatility: Current volatility
            trade_intent: Desired trade
            portfolio_state: Current portfolio state
            long_run_volatility: Long-run volatility (optional)
            initial_equity: Initial equity for daily loss check (optional)
            regime: Current regime (optional, for regime kill switch)

        Returns:
            Tuple of (updated PortfolioState, TradeFill)
        """
        # Sanitize inputs
        price = _safe_float(price, 0.0)
        volatility = _safe_float(volatility, 0.0)

        if long_run_volatility is None:
            long_run_volatility = 0.15  # Default

        if initial_equity is None:
            initial_equity = portfolio_state.equity

        # Validate trade intent
        if not isinstance(trade_intent, TradeIntent):
            # Create zero trade fill
            zero_fill = TradeFill(
                timestamp=datetime.now(),
                symbol=symbol,
                side='BUY',
                size=0.0,
                price=price,
                slippage=0.0,
                commission=0.0,
                final_price=price
            )
            return portfolio_state, zero_fill

        # Sweep H.1 - Pre-validation: Skip micro-trades
        target_pos = _safe_float(trade_intent.target_position, 0.0)
        current_pos_fraction = 0.0
        if portfolio_state.equity > EPSILON and price > EPSILON:
            current_pos_fraction = (portfolio_state.position * price) / portfolio_state.equity

        if abs(target_pos - current_pos_fraction) < 1e-6:
            # Micro-trade - skip execution
            zero_fill = TradeFill(
                timestamp=trade_intent.timestamp,
                symbol=trade_intent.symbol,
                side='BUY',
                size=0.0,
                price=price,
                slippage=0.0,
                commission=0.0,
                final_price=price
            )
            return portfolio_state, zero_fill

        # Enforce risk checks (Sweep H.1 - with regime parameter)
        allowed, reason = self.risk_engine.enforce(
            trade_intent,
            portfolio_state,
            price,
            volatility,
            long_run_volatility,
            initial_equity,
            regime=regime
        )

        if not allowed:
            # Risk check failed - create zero trade fill with risk flag
            zero_fill = TradeFill(
                timestamp=trade_intent.timestamp,
                symbol=trade_intent.symbol,
                side='BUY',
                size=0.0,
                price=price,
                slippage=0.0,
                commission=0.0,
                final_price=price,
                risk_flags=[reason] if reason else []
            )
            # Don't update portfolio state
            return portfolio_state, zero_fill

        # Execute trade (Sweep H.1 - with volatility parameter)
        trade_fill, updated_state = self.execution_engine.execute(
            trade_intent,
            portfolio_state,
            price,
            volatility
        )

        return updated_state, trade_fill


# ============================================================================
# INTEGRATION HOOK
# ============================================================================

def evo_execute(
    trade_intent: TradeIntent,
    portfolio_state: PortfolioState,
    price: float,
    volatility: float,
    risk_engine: Optional[RiskEngine] = None,
    execution_engine: Optional[ExecutionEngine] = None,
    long_run_volatility: Optional[float] = None,
    initial_equity: Optional[float] = None,
    regime: Optional[str] = None
) -> Tuple[PortfolioState, TradeFill]:
    """
    Execute trade with default engines (Sweep H.1 - Enhanced).

    Args:
        trade_intent: Desired trade
        portfolio_state: Current portfolio state
        price: Current market price
        volatility: Current volatility
        risk_engine: Risk engine (optional, creates default if None)
        execution_engine: Execution engine (optional, creates default if None)
        long_run_volatility: Long-run volatility (optional)
        initial_equity: Initial equity (optional)
        regime: Current regime (optional)

    Returns:
        Tuple of (updated PortfolioState, TradeFill)
    """
    # Create default engines if not provided
    if execution_engine is None:
        execution_engine = ExecutionEngine()

    if risk_engine is None:
        risk_engine = RiskEngine()

    # Create manager
    manager = ExecutionManager(execution_engine, risk_engine)

    # Process trade (Sweep H.1 - with regime parameter)
    updated_state, trade_fill = manager.process(
        symbol=trade_intent.symbol,
        price=price,
        volatility=volatility,
        trade_intent=trade_intent,
        portfolio_state=portfolio_state,
        long_run_volatility=long_run_volatility,
        initial_equity=initial_equity,
        regime=regime
    )

    return updated_state, trade_fill


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Module H — Execution Engine Tests")
    print("=" * 80)

    # ========================================================================
    # TEST 1: TradeIntent → TradeFill (deterministic)
    # ========================================================================
    print("\n[TEST 1] TradeIntent → Deterministic TradeFill")
    print("-" * 80)

    from datetime import datetime

    # Create initial portfolio state
    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    # Create trade intent
    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=0.5,  # 50% long
        allocator_details={}
    )

    # Execute
    engine = ExecutionEngine(slippage_bps=1.0, commission_bps=0.1)
    price = 400.0
    fill, new_state = engine.execute(intent, portfolio, price)

    print(f"  Target position: {intent.target_position:.2f}")
    print(f"  Trade size: {fill.size:.2f} shares")
    print(f"  Side: {fill.side}")
    print(f"  Base price: ${fill.price:.2f}")
    print(f"  Final price: ${fill.final_price:.4f}")
    print(f"  Slippage: ${fill.slippage:.2f}")
    print(f"  Commission: ${fill.commission:.2f}")

    assert fill.side == 'BUY', "Should be a buy"
    assert fill.size > 0, "Should have positive size"
    assert fill.slippage > 0, "Should have slippage"
    assert fill.commission > 0, "Should have commission"
    assert fill.final_price > fill.price, "Buy should have higher final price"

    print("  ✓ TradeIntent → TradeFill working")

    # ========================================================================
    # TEST 2: Slippage Calculation
    # ========================================================================
    print("\n[TEST 2] Slippage Calculation")
    print("-" * 80)

    engine = ExecutionEngine(slippage_bps=10.0)  # 10 bps slippage
    price = 100.0
    size = 100.0

    slippage = engine.compute_slippage(price, size)
    expected_slippage = 100.0 * (100.0 * 10.0 / 10000.0)  # 100 shares * $0.10

    print(f"  Price: ${price:.2f}")
    print(f"  Size: {size:.0f} shares")
    print(f"  Slippage BPS: {engine.slippage_bps:.1f}")
    print(f"  Slippage: ${slippage:.2f}")
    print(f"  Expected: ${expected_slippage:.2f}")

    assert abs(slippage - expected_slippage) < 1e-6, "Slippage calculation incorrect"

    print("  ✓ Slippage calculation correct")

    # ========================================================================
    # TEST 3: Commission Calculation
    # ========================================================================
    print("\n[TEST 3] Commission Calculation")
    print("-" * 80)

    engine = ExecutionEngine(commission_bps=5.0)  # 5 bps commission
    price = 100.0
    size = 100.0

    commission = engine.compute_commission(size, price)
    notional = 100.0 * 100.0  # $10,000
    expected_commission = notional * (5.0 / 10000.0)  # $5.00

    print(f"  Price: ${price:.2f}")
    print(f"  Size: {size:.0f} shares")
    print(f"  Notional: ${notional:.2f}")
    print(f"  Commission BPS: {engine.commission_bps:.1f}")
    print(f"  Commission: ${commission:.2f}")
    print(f"  Expected: ${expected_commission:.2f}")

    assert abs(commission - expected_commission) < 1e-6, "Commission calculation incorrect"

    print("  ✓ Commission calculation correct")

    # ========================================================================
    # TEST 4: Position Update
    # ========================================================================
    print("\n[TEST 4] Position Update")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=100.0,  # Start with 100 shares
        cash=50000.0,
        equity=90000.0,
        entry_price=400.0
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=0.8,  # Increase to 80%
        allocator_details={}
    )

    engine = ExecutionEngine()
    price = 400.0
    fill, new_state = engine.execute(intent, portfolio, price)

    print(f"  Initial position: {portfolio.position:.2f} shares")
    print(f"  Trade size: {fill.size:.2f} shares")
    print(f"  New position: {new_state.position:.2f} shares")

    expected_new_position = portfolio.position + fill.size
    assert abs(new_state.position - expected_new_position) < 1e-6, "Position update incorrect"

    print("  ✓ Position update correct")

    # ========================================================================
    # TEST 5: Equity Update
    # ========================================================================
    print("\n[TEST 5] Equity Update")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=0.5,
        allocator_details={}
    )

    engine = ExecutionEngine()
    price = 400.0
    fill, new_state = engine.execute(intent, portfolio, price)

    print(f"  Initial equity: ${portfolio.equity:.2f}")
    print(f"  Trade cost: ${fill.size * fill.final_price + fill.commission:.2f}")
    print(f"  New equity: ${new_state.equity:.2f}")

    # Equity should decrease by trade cost
    expected_equity = portfolio.cash - (fill.size * fill.final_price + fill.commission) + new_state.position * price
    assert abs(new_state.equity - expected_equity) < 1.0, "Equity update within tolerance"

    print("  ✓ Equity update correct")

    # ========================================================================
    # TEST 6: Max Position Limit Block
    # ========================================================================
    print("\n[TEST 6] Max Position Limit Block")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=2.0,  # Exceeds max_position=1.0
        allocator_details={}
    )

    risk_engine = RiskEngine(max_position=1.0)
    execution_engine = ExecutionEngine()
    manager = ExecutionManager(execution_engine, risk_engine)

    new_state, fill = manager.process(
        symbol='SPY',
        price=400.0,
        volatility=0.20,
        trade_intent=intent,
        portfolio_state=portfolio
    )

    print(f"  Target position: {intent.target_position:.2f}")
    print(f"  Max position: {risk_engine.max_position:.2f}")
    print(f"  Trade size: {fill.size:.2f}")

    assert fill.size == 0.0, "Trade should be blocked"
    assert new_state.position == portfolio.position, "Position should not change"

    print("  ✓ Max position limit working")

    # ========================================================================
    # TEST 7: Max Leverage Block
    # ========================================================================
    print("\n[TEST 7] Max Leverage Block")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=10000.0,  # Small cash
        equity=10000.0,
        entry_price=None
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=3.0,  # 3x leverage
        allocator_details={}
    )

    risk_engine = RiskEngine(max_leverage=2.0)  # Only allow 2x
    execution_engine = ExecutionEngine()
    manager = ExecutionManager(execution_engine, risk_engine)

    new_state, fill = manager.process(
        symbol='SPY',
        price=400.0,
        volatility=0.20,
        trade_intent=intent,
        portfolio_state=portfolio
    )

    print(f"  Target position: {intent.target_position:.2f}")
    print(f"  Max leverage: {risk_engine.max_leverage:.2f}")
    print(f"  Trade size: {fill.size:.2f}")

    assert fill.size == 0.0, "Trade should be blocked"

    print("  ✓ Max leverage limit working")

    # ========================================================================
    # TEST 8: Volatility Kill Switch
    # ========================================================================
    print("\n[TEST 8] Volatility Kill Switch")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=0.5,
        allocator_details={}
    )

    risk_engine = RiskEngine(vol_kill_multiplier=2.0)
    execution_engine = ExecutionEngine()
    manager = ExecutionManager(execution_engine, risk_engine)

    # High volatility scenario
    new_state, fill = manager.process(
        symbol='SPY',
        price=400.0,
        volatility=0.50,  # Very high
        trade_intent=intent,
        portfolio_state=portfolio,
        long_run_volatility=0.15  # Normal
    )

    print(f"  Current volatility: {0.50:.2f}")
    print(f"  Long-run volatility: {0.15:.2f}")
    print(f"  Kill multiplier: {risk_engine.vol_kill_multiplier:.2f}")
    print(f"  Threshold: {0.15 * 2.0:.2f}")
    print(f"  Trade size: {fill.size:.2f}")

    assert fill.size == 0.0, "Trade should be blocked by volatility kill"

    print("  ✓ Volatility kill switch working")

    # ========================================================================
    # TEST 9: NaN/Inf Safety
    # ========================================================================
    print("\n[TEST 9] NaN/Inf Safety")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    # Create intent with NaN
    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=np.nan,  # NaN input
        allocator_details={}
    )

    print(f"  Target position (input): {np.nan}")
    print(f"  Target position (sanitized): {intent.target_position:.2f}")

    assert intent.target_position == 0.0, "NaN should be sanitized to 0.0"

    execution_engine = ExecutionEngine()
    fill, new_state = execution_engine.execute(intent, portfolio, 400.0)

    print(f"  Trade size: {fill.size:.2f}")
    print(f"  All outputs finite: {np.isfinite([fill.size, fill.price, new_state.equity]).all()}")

    assert np.isfinite(fill.size), "Fill size should be finite"
    assert np.isfinite(new_state.equity), "Equity should be finite"

    print("  ✓ NaN/Inf safety working")

    # ========================================================================
    # TEST 10: Full Lifecycle Trade
    # ========================================================================
    print("\n[TEST 10] Full Lifecycle Trade")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=0.5,
        allocator_details={'strategy': 'momentum'}
    )

    new_state, fill = evo_execute(
        trade_intent=intent,
        portfolio_state=portfolio,
        price=400.0,
        volatility=0.20
    )

    print(f"  Trade executed: {fill.side} {abs(fill.size):.2f} shares")
    print(f"  Initial equity: ${portfolio.equity:.2f}")
    print(f"  Final equity: ${new_state.equity:.2f}")
    print(f"  Position: {new_state.position:.2f} shares")
    print(f"  Entry price: ${new_state.entry_price:.2f}")

    assert fill.size != 0.0, "Trade should execute"
    assert new_state.position != portfolio.position, "Position should change"
    assert len(new_state.trade_history) == 1, "Trade history should have one entry"

    print("  ✓ Full lifecycle trade working")

    # ========================================================================
    # TEST 11: Multiple Sequential Trades
    # ========================================================================
    print("\n[TEST 11] Multiple Sequential Trades")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    execution_engine = ExecutionEngine()

    # Trade 1: Go 50% long
    intent1 = TradeIntent(datetime.now(), 'SPY', 0.5, {})
    fill1, portfolio = execution_engine.execute(intent1, portfolio, 400.0)

    print(f"  Trade 1: {fill1.side} {abs(fill1.size):.2f} shares")
    print(f"    Position: {portfolio.position:.2f}")

    # Trade 2: Increase to 80% long
    intent2 = TradeIntent(datetime.now(), 'SPY', 0.8, {})
    fill2, portfolio = execution_engine.execute(intent2, portfolio, 400.0)

    print(f"  Trade 2: {fill2.side} {abs(fill2.size):.2f} shares")
    print(f"    Position: {portfolio.position:.2f}")

    # Trade 3: Reduce to 30% long
    intent3 = TradeIntent(datetime.now(), 'SPY', 0.3, {})
    fill3, portfolio = execution_engine.execute(intent3, portfolio, 400.0)

    print(f"  Trade 3: {fill3.side} {abs(fill3.size):.2f} shares")
    print(f"    Position: {portfolio.position:.2f}")

    assert len(portfolio.trade_history) == 3, "Should have 3 trades in history"
    assert portfolio.position != 0.0, "Should have final position"

    print("  ✓ Multiple sequential trades working")

    # ========================================================================
    # TEST 12: Risk Engine Override Logging
    # ========================================================================
    print("\n[TEST 12] Risk Engine Override Logging")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=2.0,  # Exceeds limit
        allocator_details={}
    )

    risk_engine = RiskEngine(max_position=1.0)
    allowed, reason = risk_engine.check_position_limit(intent.target_position)

    print(f"  Target position: {intent.target_position:.2f}")
    print(f"  Allowed: {allowed}")
    print(f"  Reason: {reason}")

    assert not allowed, "Should not be allowed"
    assert reason is not None, "Should have reason"
    assert "Position limit exceeded" in reason, "Reason should mention position limit"

    print("  ✓ Risk engine override logging working")

    # ========================================================================
    # TEST 13: Zero Trade Intent
    # ========================================================================
    print("\n[TEST 13] Zero Trade Intent")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=100.0,
        cash=50000.0,
        equity=90000.0,
        entry_price=400.0
    )

    # Intent matches current position
    current_pos_fraction = (100.0 * 400.0) / 90000.0
    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=current_pos_fraction,  # Same as current
        allocator_details={}
    )

    execution_engine = ExecutionEngine()
    fill, new_state = execution_engine.execute(intent, portfolio, 400.0)

    print(f"  Target position: {intent.target_position:.4f}")
    print(f"  Current position fraction: {current_pos_fraction:.4f}")
    print(f"  Trade size: {fill.size:.6f}")

    assert abs(fill.size) < 1e-6, "Trade size should be near zero"
    assert new_state.position == portfolio.position, "Position should not change"

    print("  ✓ Zero trade intent working")

    # ========================================================================
    # TEST 14: Deterministic Output
    # ========================================================================
    print("\n[TEST 14] Deterministic Output")
    print("-" * 80)

    portfolio1 = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    portfolio2 = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent1 = TradeIntent(datetime.now(), 'SPY', 0.5, {})
    intent2 = TradeIntent(datetime.now(), 'SPY', 0.5, {})

    execution_engine = ExecutionEngine(slippage_bps=1.0, commission_bps=0.1)

    fill1, state1 = execution_engine.execute(intent1, portfolio1, 400.0)
    fill2, state2 = execution_engine.execute(intent2, portfolio2, 400.0)

    print(f"  Fill 1 size: {fill1.size:.6f}")
    print(f"  Fill 2 size: {fill2.size:.6f}")
    print(f"  State 1 equity: ${state1.equity:.2f}")
    print(f"  State 2 equity: ${state2.equity:.2f}")

    assert abs(fill1.size - fill2.size) < 1e-10, "Fills should be identical"
    assert abs(state1.equity - state2.equity) < 1e-6, "Equity should be identical"

    print("  ✓ Deterministic output working")

    # ========================================================================
    # TEST 15: Volatility-Adjusted Slippage (Sweep H.1)
    # ========================================================================
    print("\n[TEST 15] Volatility-Adjusted Slippage (Sweep H.1)")
    print("-" * 80)

    engine = ExecutionEngine(slippage_bps=1.0)
    price = 100.0
    size = 100.0

    # Low volatility
    slippage_low_vol = engine.compute_slippage(price, size, volatility=0.10)
    # High volatility
    slippage_high_vol = engine.compute_slippage(price, size, volatility=0.50)

    print(f"  Price: ${price:.2f}")
    print(f"  Size: {size:.0f} shares")
    print(f"  Slippage (low vol 0.10): ${slippage_low_vol:.2f}")
    print(f"  Slippage (high vol 0.50): ${slippage_high_vol:.2f}")

    assert slippage_high_vol > slippage_low_vol, "High volatility should have higher slippage"

    print("  ✓ Volatility-adjusted slippage working")

    # ========================================================================
    # TEST 16: Commission Min/Max Caps (Sweep H.1)
    # ========================================================================
    print("\n[TEST 16] Commission Min/Max Caps (Sweep H.1)")
    print("-" * 80)

    engine = ExecutionEngine(commission_bps=0.1)  # Very low commission rate

    # Small trade - should hit minimum $0.01
    # Notional = 10 shares × $10 = $100
    # Base commission = $100 × 0.00001 = $0.001
    # Min cap kicks in: max($0.001, $0.01) = $0.01
    # Max cap: $100 × 0.005 = $0.50 (doesn't apply)
    size_small = 10.0
    price_small = 10.0
    commission_small = engine.compute_commission(size_small, price_small)

    print(f"  Small trade: {size_small} shares @ ${price_small:.2f}")
    print(f"  Notional: ${size_small * price_small:.2f}")
    print(f"  Base commission (0.1 bps): ${size_small * price_small * 0.00001:.6f}")
    print(f"  Commission (with min cap): ${commission_small:.4f}")
    print(f"  Minimum cap: $0.01")

    assert commission_small == 0.01, "Commission should hit minimum cap of $0.01"

    # Large trade - should hit maximum (0.5% of notional)
    # Notional = 1,000,000 shares × $100 = $100,000,000
    # Base commission (at 1% or 100 bps) = $100,000,000 × 0.01 = $1,000,000
    # Max cap: $100,000,000 × 0.005 = $500,000
    # min($1,000,000, $500,000) = $500,000 (max cap kicks in)
    size_large = 1000000.0
    price_large = 100.0
    engine_high_comm = ExecutionEngine(commission_bps=100.0)  # 1% commission
    commission_large = engine_high_comm.compute_commission(size_large, price_large)
    max_commission = abs(size_large * price_large) * 0.005

    print(f"  Large trade: {size_large:.0f} shares @ ${price_large:.2f}")
    print(f"  Notional: ${size_large * price_large:.0f}")
    print(f"  Base commission (1% or 100 bps): ${size_large * price_large * 0.01:.2f}")
    print(f"  Commission (with max cap): ${commission_large:.2f}")
    print(f"  Maximum cap (0.5% notional): ${max_commission:.2f}")

    assert commission_large == max_commission, "Commission should hit maximum cap"

    print("  ✓ Commission min/max caps working")

    # ========================================================================
    # TEST 17: Correlation Kill Switch (Sweep H.1)
    # ========================================================================
    print("\n[TEST 17] Correlation Kill Switch (Sweep H.1)")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    # High conflict ratio
    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=0.5,
        allocator_details={'conflict_ratio': 10.0}  # High conflict
    )

    risk_engine = RiskEngine(conflict_ratio_threshold=5.0)
    execution_engine = ExecutionEngine()
    manager = ExecutionManager(execution_engine, risk_engine)

    new_state, fill = manager.process(
        symbol='SPY',
        price=400.0,
        volatility=0.20,
        trade_intent=intent,
        portfolio_state=portfolio
    )

    print(f"  Conflict ratio: 10.0")
    print(f"  Threshold: 5.0")
    print(f"  Trade size: {fill.size:.2f}")

    assert fill.size == 0.0, "Trade should be blocked by correlation kill"
    assert len(fill.risk_flags) > 0, "Should have risk flag"

    print("  ✓ Correlation kill switch working")

    # ========================================================================
    # TEST 18: Regime Kill Switch (Sweep H.1)
    # ========================================================================
    print("\n[TEST 18] Regime Kill Switch (Sweep H.1)")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=0.5,
        allocator_details={}
    )

    risk_engine = RiskEngine()
    execution_engine = ExecutionEngine()
    manager = ExecutionManager(execution_engine, risk_engine)

    # HIGH_VOL regime with high volatility
    new_state, fill = manager.process(
        symbol='SPY',
        price=400.0,
        volatility=0.50,  # High volatility
        trade_intent=intent,
        portfolio_state=portfolio,
        long_run_volatility=0.15,
        regime='HIGH_VOL'
    )

    print(f"  Regime: HIGH_VOL")
    print(f"  Current volatility: 0.50")
    print(f"  Long-run volatility: 0.15")
    print(f"  Threshold (2× long-run): 0.30")
    print(f"  Trade size: {fill.size:.2f}")

    assert fill.size == 0.0, "Trade should be blocked by regime kill"

    print("  ✓ Regime kill switch working")

    # ========================================================================
    # TEST 19: Large Price Jump Kill Switch (Sweep H.1)
    # ========================================================================
    print("\n[TEST 19] Large Price Jump Kill Switch (Sweep H.1)")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent1 = TradeIntent(datetime.now(), 'SPY', 0.5, {})
    intent2 = TradeIntent(datetime.now(), 'SPY', 0.6, {})

    # Use a smaller multiplier so the jump exceeds threshold
    risk_engine = RiskEngine(jump_protection_multiplier=2.0)
    execution_engine = ExecutionEngine()
    manager = ExecutionManager(execution_engine, risk_engine)

    # First trade at normal price
    new_state, fill1 = manager.process(
        symbol='SPY',
        price=400.0,
        volatility=0.15,
        trade_intent=intent1,
        portfolio_state=portfolio,
        long_run_volatility=0.15
    )

    print(f"  Trade 1 price: $400.00")
    print(f"  Trade 1 executed: {fill1.size:.2f} shares")

    # Second trade with large price jump (50% = $200 jump)
    # Expected max move: 2.0 × 0.15 × $400 = $120
    # Actual jump: $200 > $120, so should be blocked
    new_state, fill2 = manager.process(
        symbol='SPY',
        price=600.0,  # 50% jump = $200
        volatility=0.15,
        trade_intent=intent2,
        portfolio_state=new_state,
        long_run_volatility=0.15
    )

    print(f"  Trade 2 price: $600.00 (50% jump = $200)")
    print(f"  Expected max move: ${2.0 * 0.15 * 400.0:.2f}")
    print(f"  Actual jump: $200.00")
    print(f"  Trade 2 size: {fill2.size:.2f}")

    assert fill2.size == 0.0, "Trade should be blocked by jump protection"

    print("  ✓ Large price jump kill switch working")

    # ========================================================================
    # TEST 20: Entry Price Reset on Zero-Crossing (Sweep H.1)
    # ========================================================================
    print("\n[TEST 20] Entry Price Reset on Zero-Crossing (Sweep H.1)")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=100.0,  # Start long
        cash=60000.0,
        equity=100000.0,
        entry_price=400.0
    )

    execution_engine = ExecutionEngine()

    # Go from +100 shares to -50 shares (cross zero)
    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol='SPY',
        target_position=-0.125,  # Will result in negative position
        allocator_details={}
    )

    fill, new_state = execution_engine.execute(intent, portfolio, 420.0, volatility=0.15)

    print(f"  Initial position: {portfolio.position:.2f} shares")
    print(f"  Initial entry price: ${portfolio.entry_price:.2f}")
    print(f"  Trade size: {fill.size:.2f} shares")
    print(f"  New position: {new_state.position:.2f} shares")
    print(f"  New entry price: ${new_state.entry_price:.2f}")

    # Entry price should be reset (close to final_price)
    assert new_state.entry_price != portfolio.entry_price, "Entry price should reset on zero-crossing"

    print("  ✓ Entry price reset on zero-crossing working")

    # ========================================================================
    # TEST 21: Realized P&L Computation (Sweep H.1)
    # ========================================================================
    print("\n[TEST 21] Realized P&L Computation (Sweep H.1)")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(datetime.now(), 'SPY', 0.5, {})
    execution_engine = ExecutionEngine(commission_bps=10.0)  # High commission for visibility

    fill, new_state = execution_engine.execute(intent, portfolio, 400.0, volatility=0.15)

    print(f"  Commission: ${fill.commission:.2f}")
    print(f"  Realized P&L: ${fill.pnl_realized:.2f}")

    assert fill.pnl_realized == -fill.commission, "Realized P&L should be -commission"
    assert fill.pnl_realized < 0, "Realized P&L should be negative (cost)"

    print("  ✓ Realized P&L computation working")

    # ========================================================================
    # TEST 22: Unrealized P&L Computation (Sweep H.1)
    # ========================================================================
    print("\n[TEST 22] Unrealized P&L Computation (Sweep H.1)")
    print("-" * 80)

    portfolio = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent = TradeIntent(datetime.now(), 'SPY', 0.5, {})
    execution_engine = ExecutionEngine()

    fill, new_state = execution_engine.execute(intent, portfolio, 400.0, volatility=0.15)

    print(f"  Position: {new_state.position:.2f} shares")
    print(f"  Entry price: ${new_state.entry_price:.2f}")
    print(f"  Current price: $400.00")
    print(f"  Unrealized P&L: ${fill.pnl_unrealized:.2f}")

    # Unrealized P&L should be negative due to slippage on entry
    # Entry price > current price, so unrealized P&L is negative
    assert fill.pnl_unrealized < 0, "Unrealized P&L should be negative (entry price > current price due to slippage)"
    assert abs(fill.pnl_unrealized) > 0, "Unrealized P&L should be non-zero"

    print("  ✓ Unrealized P&L computation working")

    # ========================================================================
    # TEST 23: Invalid Symbol Sanitization (Sweep H.1)
    # ========================================================================
    print("\n[TEST 23] Invalid Symbol Sanitization (Sweep H.1)")
    print("-" * 80)

    intent = TradeIntent(
        timestamp=datetime.now(),
        symbol=None,  # Invalid symbol
        target_position=0.5,
        allocator_details={}
    )

    print(f"  Symbol (input): None")
    print(f"  Symbol (sanitized): {intent.symbol}")

    assert intent.symbol == "UNKNOWN", "Invalid symbol should be sanitized to UNKNOWN"

    print("  ✓ Invalid symbol sanitization working")

    # ========================================================================
    # TEST 24: Deterministic Volatility-Adjusted Slippage (Sweep H.1)
    # ========================================================================
    print("\n[TEST 24] Deterministic Volatility-Adjusted Slippage (Sweep H.1)")
    print("-" * 80)

    portfolio1 = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    portfolio2 = PortfolioState(
        timestamp=datetime.now(),
        symbol='SPY',
        position=0.0,
        cash=100000.0,
        equity=100000.0,
        entry_price=None
    )

    intent1 = TradeIntent(datetime.now(), 'SPY', 0.5, {})
    intent2 = TradeIntent(datetime.now(), 'SPY', 0.5, {})

    execution_engine = ExecutionEngine()

    fill1, state1 = execution_engine.execute(intent1, portfolio1, 400.0, volatility=0.25)
    fill2, state2 = execution_engine.execute(intent2, portfolio2, 400.0, volatility=0.25)

    print(f"  Fill 1 slippage: ${fill1.slippage:.6f}")
    print(f"  Fill 2 slippage: ${fill2.slippage:.6f}")
    print(f"  Difference: ${abs(fill1.slippage - fill2.slippage):.10f}")

    assert abs(fill1.slippage - fill2.slippage) < 1e-10, "Slippage should be deterministic"

    print("  ✓ Deterministic volatility-adjusted slippage working")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL MODULE H TESTS PASSED (24 TESTS)")
    print("=" * 80)
    print("\nExecution Engine Features (Original):")
    print("  ✓ TradeIntent → TradeFill (deterministic)")
    print("  ✓ Slippage calculation")
    print("  ✓ Commission calculation")
    print("  ✓ Position update")
    print("  ✓ Equity update")
    print("  ✓ Max position limit")
    print("  ✓ Max leverage limit")
    print("  ✓ Volatility kill switch")
    print("  ✓ NaN/Inf safety")
    print("  ✓ Full lifecycle trade")
    print("  ✓ Multiple sequential trades")
    print("  ✓ Risk engine override logging")
    print("  ✓ Zero trade intent")
    print("  ✓ Deterministic output")
    print("\nSweep H.1 Enhancements (Institutional Grade):")
    print("  ✓ Volatility-adjusted slippage model")
    print("  ✓ Commission min/max caps")
    print("  ✓ Correlation kill switch")
    print("  ✓ Regime kill switch")
    print("  ✓ Large price jump protection")
    print("  ✓ Entry price reset on zero-crossing")
    print("  ✓ Realized P&L tracking")
    print("  ✓ Unrealized P&L tracking")
    print("  ✓ Enhanced TradeFill diagnostics")
    print("  ✓ Pre-validation (skip micro-trades)")
    print("\nModule H — Execution Engine: INSTITUTIONAL GRADE (Sweep H.1 Complete)")
    print("=" * 80)
