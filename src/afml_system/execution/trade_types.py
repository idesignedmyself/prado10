"""
PRADO9_EVO Module H — Trade Types

Core data structures for execution:
- TradeIntent: Desired trade from allocator
- TradeFill: Actual executed trade
- PortfolioState: Current portfolio state

Author: PRADO9_EVO Builder
Date: 2025-01-16
Version: 1.1.0 (Sweep H.1 - Institutional Grade)
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """
    Convert value to safe float with fallback.

    Args:
        value: Value to convert
        fallback: Fallback if value is NaN/Inf

    Returns:
        Safe float value
    """
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return float(fallback)
        return val
    except (ValueError, TypeError):
        return float(fallback)


def _sanitize_str(value: Any, fallback: str = "UNKNOWN") -> str:
    """
    Convert value to safe string with fallback.

    Args:
        value: Value to convert
        fallback: Fallback if value is invalid

    Returns:
        Safe string value
    """
    try:
        if value is None:
            return fallback
        return str(value)
    except (ValueError, TypeError):
        return fallback


def _sanitize_dict(value: Any, fallback: Optional[Dict] = None) -> Dict:
    """
    Convert value to safe dict with fallback.

    Args:
        value: Value to convert
        fallback: Fallback if value is invalid

    Returns:
        Safe dict value
    """
    if fallback is None:
        fallback = {}

    if isinstance(value, dict):
        return value
    return fallback


def _sanitize_timestamp(value: Any) -> datetime:
    """
    Convert value to safe datetime.

    Args:
        value: Value to convert

    Returns:
        Safe datetime value
    """
    if isinstance(value, datetime):
        return value
    return datetime.utcnow()


# ============================================================================
# TRADE INTENT
# ============================================================================

@dataclass
class TradeIntent:
    """
    Desired trade from allocator.

    Represents the target position that the execution engine
    should aim to achieve.
    """
    timestamp: datetime
    symbol: str
    target_position: float  # Target position in [-1, +1] or absolute notional
    allocator_details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Sanitize fields (Sweep H.1 - Enhanced)."""
        # Sanitize timestamp
        self.timestamp = _sanitize_timestamp(self.timestamp)

        # Sanitize symbol
        self.symbol = _sanitize_str(self.symbol, "UNKNOWN")

        # Sanitize target_position
        self.target_position = _safe_float(self.target_position, 0.0)

        # Sanitize allocator_details
        self.allocator_details = _sanitize_dict(self.allocator_details, {})


# ============================================================================
# TRADE FILL
# ============================================================================

@dataclass
class TradeFill:
    """
    Actual executed trade.

    Represents a completed trade with slippage and commission applied.

    Sweep H.1 - Enhanced with additional diagnostics.
    """
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    size: float  # Shares/contracts traded
    price: float  # Base price
    slippage: float  # Slippage cost (positive = cost)
    commission: float  # Commission cost
    final_price: float  # Actual execution price

    # Sweep H.1 - Additional diagnostics
    notional: float = 0.0  # Trade notional value
    post_trade_position: float = 0.0  # Position after trade
    post_trade_equity: float = 0.0  # Equity after trade
    pnl_realized: float = 0.0  # Realized P&L from this trade
    pnl_unrealized: float = 0.0  # Unrealized P&L after trade
    risk_flags: List[str] = field(default_factory=list)  # Risk warnings

    def __post_init__(self):
        """Sanitize fields (Sweep H.1 - Enhanced)."""
        # Sanitize timestamp
        self.timestamp = _sanitize_timestamp(self.timestamp)

        # Sanitize symbol
        self.symbol = _sanitize_str(self.symbol, "UNKNOWN")

        # Ensure side is BUY or SELL
        if not isinstance(self.side, str):
            self.side = 'BUY'
        else:
            self.side = self.side.upper()
            if self.side not in ['BUY', 'SELL']:
                self.side = 'BUY'

        # Sanitize numeric fields
        self.size = _safe_float(self.size, 0.0)
        self.price = _safe_float(self.price, 0.0)
        self.slippage = _safe_float(self.slippage, 0.0)
        self.commission = _safe_float(self.commission, 0.0)
        self.final_price = _safe_float(self.final_price, self.price)

        # Sweep H.1 - Sanitize new fields
        self.notional = _safe_float(self.notional, abs(self.size * self.price))
        self.post_trade_position = _safe_float(self.post_trade_position, 0.0)
        self.post_trade_equity = _safe_float(self.post_trade_equity, 0.0)
        self.pnl_realized = _safe_float(self.pnl_realized, 0.0)
        self.pnl_unrealized = _safe_float(self.pnl_unrealized, 0.0)

        # Ensure risk_flags is list
        if not isinstance(self.risk_flags, list):
            self.risk_flags = []

    def total_cost(self) -> float:
        """Compute total cost of trade including slippage and commission."""
        return abs(self.size * self.final_price) + self.commission

    def pnl_impact(self) -> float:
        """Compute P&L impact (negative = cost)."""
        return -self.total_cost() if self.side == 'BUY' else self.total_cost()


# ============================================================================
# PORTFOLIO STATE
# ============================================================================

@dataclass
class PortfolioState:
    """
    Current portfolio state.

    Tracks position, cash, equity, and trade history.

    Sweep H.1 - Enhanced with P&L tracking and risk logging.
    """
    timestamp: datetime
    symbol: str
    position: float  # Current position (shares/contracts)
    cash: float  # Available cash
    equity: float  # Total equity (cash + position value)
    entry_price: Optional[float] = None  # Average entry price
    trade_history: List[TradeFill] = field(default_factory=list)

    # Sweep H.1 - Additional tracking
    daily_pnl: float = 0.0  # Daily P&L
    total_pnl: float = 0.0  # Total P&L
    risk_log: List[str] = field(default_factory=list)  # Risk event log

    def __post_init__(self):
        """Sanitize fields (Sweep H.1 - Enhanced)."""
        # Sanitize timestamp
        self.timestamp = _sanitize_timestamp(self.timestamp)

        # Sanitize symbol
        self.symbol = _sanitize_str(self.symbol, "UNKNOWN")

        # Sanitize numeric fields
        self.position = _safe_float(self.position, 0.0)
        self.cash = _safe_float(self.cash, 0.0)
        self.equity = _safe_float(self.equity, 0.0)

        # Sweep H.1 - Prevent divide-by-zero: minimum equity
        self.equity = max(self.equity, 1e-6)

        if self.entry_price is not None:
            self.entry_price = _safe_float(self.entry_price, None)

        # Ensure lists
        if not isinstance(self.trade_history, list):
            self.trade_history = []

        # Sweep H.1 - Sanitize new fields
        self.daily_pnl = _safe_float(self.daily_pnl, 0.0)
        self.total_pnl = _safe_float(self.total_pnl, 0.0)

        if not isinstance(self.risk_log, list):
            self.risk_log = []

    def unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized P&L."""
        if self.position == 0.0 or self.entry_price is None:
            return 0.0

        current_price = _safe_float(current_price, 0.0)
        return self.position * (current_price - self.entry_price)

    def total_pnl_current(self, current_price: float) -> float:
        """Compute total P&L (realized + unrealized)."""
        unrealized = self.unrealized_pnl(current_price)

        # Realized P&L from trade history
        realized = 0.0
        for fill in self.trade_history:
            # Simplified: commission is always a cost
            realized -= fill.commission

        return realized + unrealized

    def leverage(self, current_price: float) -> float:
        """Compute current leverage ratio."""
        if self.equity <= 1e-6:
            return 0.0

        current_price = _safe_float(current_price, 0.0)
        notional = abs(self.position * current_price)

        return notional / self.equity if self.equity > 1e-6 else 0.0

    def copy(self) -> 'PortfolioState':
        """Create a copy of the portfolio state."""
        return PortfolioState(
            timestamp=self.timestamp,
            symbol=self.symbol,
            position=self.position,
            cash=self.cash,
            equity=self.equity,
            entry_price=self.entry_price,
            trade_history=self.trade_history.copy(),
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl,
            risk_log=self.risk_log.copy()
        )
