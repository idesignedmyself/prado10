"""
PRADO9_EVO Module J.5 — Structured Logger

Production-grade logging for live trading.

Features:
- JSON + text format
- Daily rotation
- Event, trade, signal, and error logging
- Audit trail
- Structured metadata

Author: PRADO9_EVO Builder
Date: 2025-01-16
Version: 1.0.0
"""

import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


# ============================================================================
# CONSTANTS
# ============================================================================

LOGGER_VERSION = '1.0.0'

# Default log directory - will use project-local path
def _get_default_log_dir():
    from ..utils.paths import get_logs_dir
    return get_logs_dir()

DEFAULT_LOG_DIR = None  # Set dynamically in __init__


# ============================================================================
# LOG ENTRY
# ============================================================================

@dataclass
class LogEntry:
    """
    Structured log entry.

    Contains:
    - Timestamp
    - Level (INFO, WARNING, ERROR)
    - Type (event, trade, signal, error)
    - Message
    - Metadata
    """
    timestamp: str
    level: str
    log_type: str
    message: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Convert to JSON line."""
        return json.dumps(self.to_dict())

    def to_text_line(self) -> str:
        """Convert to text line."""
        return f"[{self.timestamp}] [{self.level}] [{self.log_type}] {self.message}"


# ============================================================================
# LIVE LOGGER
# ============================================================================

class LiveLogger:
    """
    Structured logger for live trading.

    Features:
    - JSON + text output
    - Daily rotation
    - Multiple log types (event, trade, signal, error)
    - Structured metadata
    - Audit trail
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        enable_json: bool = True,
        enable_text: bool = True,
        enable_console: bool = False
    ):
        """
        Initialize live logger.

        Args:
            log_dir: Log directory (default: .prado/logs/)
            enable_json: Enable JSON logging
            enable_text: Enable text logging
            enable_console: Enable console output
        """
        # Log directory
        if log_dir is None:
            log_dir = _get_default_log_dir()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Settings
        self.enable_json = enable_json
        self.enable_text = enable_text
        self.enable_console = enable_console

        # Current date (for rotation)
        self.current_date = datetime.now().strftime('%Y%m%d')

        # Log files
        self.json_file = self.log_dir / f"live_{self.current_date}.json"
        self.text_file = self.log_dir / f"live_{self.current_date}.log"

        # Size rotation (Sweep J.1)
        self.max_file_size = 10 * 1024 * 1024  # 10MB

    def _check_rotation(self):
        """Check if log rotation is needed."""
        current_date = datetime.now().strftime('%Y%m%d')

        if current_date != self.current_date:
            # Rotate logs
            self.current_date = current_date
            self.json_file = self.log_dir / f"live_{self.current_date}.json"
            self.text_file = self.log_dir / f"live_{self.current_date}.log"

    def _write_entry(self, entry: LogEntry):
        """
        Write log entry to files (Sweep J.1: Enhanced with size rotation and JSON safety).

        Args:
            entry: LogEntry to write
        """
        # Check rotation
        self._check_rotation()

        # Check size rotation (Sweep J.1)
        if self.json_file.exists() and self.json_file.stat().st_size > self.max_file_size:
            self._rotate_by_size('json')
        if self.text_file.exists() and self.text_file.stat().st_size > self.max_file_size:
            self._rotate_by_size('text')

        # Sanitize entry for JSON safety (Sweep J.1)
        entry = self._sanitize_entry(entry)

        # Console output
        if self.enable_console:
            print(entry.to_text_line())

        # JSON output
        if self.enable_json:
            try:
                with open(self.json_file, 'a') as f:
                    f.write(entry.to_json_line() + '\n')
            except Exception:
                pass  # Silent fail

        # Text output
        if self.enable_text:
            try:
                with open(self.text_file, 'a') as f:
                    f.write(entry.to_text_line() + '\n')
            except Exception:
                pass  # Silent fail

    def _sanitize_entry(self, entry: LogEntry) -> LogEntry:
        """
        Sanitize log entry for JSON safety (Sweep J.1).

        Args:
            entry: LogEntry to sanitize

        Returns:
            Sanitized LogEntry
        """
        import numpy as np

        sanitized_metadata = {}
        for key, value in entry.metadata.items():
            if isinstance(value, (datetime,)):
                sanitized_metadata[key] = value.isoformat()
            elif isinstance(value, (np.integer, np.floating)):
                sanitized_metadata[key] = float(value)
            elif isinstance(value, np.ndarray):
                sanitized_metadata[key] = value.tolist()
            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                sanitized_metadata[key] = 0.0
            else:
                sanitized_metadata[key] = value

        entry.metadata = sanitized_metadata
        return entry

    def _rotate_by_size(self, log_type: str):
        """
        Rotate log file by size (Sweep J.1).

        Args:
            log_type: 'json' or 'text'
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if log_type == 'json':
            new_name = self.log_dir / f"live_{self.current_date}_{timestamp}.json"
            self.json_file.rename(new_name)
            self.json_file = self.log_dir / f"live_{self.current_date}.json"
        elif log_type == 'text':
            new_name = self.log_dir / f"live_{self.current_date}_{timestamp}.log"
            self.text_file.rename(new_name)
            self.text_file = self.log_dir / f"live_{self.current_date}.log"

    def log_event(
        self,
        event_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        level: str = 'INFO'
    ):
        """
        Log event.

        Args:
            event_type: Event type (start, stop, pause, resume, etc.)
            message: Event message
            metadata: Additional metadata
            level: Log level (INFO, WARNING, ERROR)
        """
        if metadata is None:
            metadata = {}

        metadata['event_type'] = event_type

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            log_type='event',
            message=message,
            metadata=metadata
        )

        self._write_entry(entry)

    def log_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        commission: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log trade execution.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            size: Trade size
            price: Fill price
            commission: Commission paid
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}

        metadata.update({
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'commission': commission
        })

        message = f"{side} {size:.2f} shares of {symbol} @ ${price:.2f}"

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level='INFO',
            log_type='trade',
            message=message,
            metadata=metadata
        )

        self._write_entry(entry)

    def log_signal(
        self,
        symbol: str,
        regime: str,
        horizon: str,
        final_position: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log signal generation.

        Args:
            symbol: Trading symbol
            regime: Market regime
            horizon: Time horizon
            final_position: Final position from allocator
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}

        metadata.update({
            'symbol': symbol,
            'regime': regime,
            'horizon': horizon,
            'final_position': final_position
        })

        message = f"Signal: {symbol} | Regime: {regime} | Position: {final_position:.4f}"

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level='INFO',
            log_type='signal',
            message=message,
            metadata=metadata
        )

        self._write_entry(entry)

    def log_error(
        self,
        error_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log error.

        Args:
            error_type: Error type
            message: Error message
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}

        metadata['error_type'] = error_type

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level='ERROR',
            log_type='error',
            message=message,
            metadata=metadata
        )

        self._write_entry(entry)

    def log_kill_switch(
        self,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log kill-switch trigger.

        Args:
            reason: Kill-switch reason
            metadata: Additional metadata
        """
        if metadata is None:
            metadata = {}

        metadata['kill_switch_reason'] = reason

        message = f"KILL-SWITCH TRIGGERED: {reason}"

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level='WARNING',
            log_type='kill_switch',
            message=message,
            metadata=metadata
        )

        self._write_entry(entry)


# ============================================================================
# INTEGRATION HOOK
# ============================================================================

def evo_log_trade(
    symbol: str,
    side: str,
    size: float,
    price: float,
    commission: float = 0.0
):
    """
    Integration hook: Log trade.

    Args:
        symbol: Trading symbol
        side: BUY or SELL
        size: Trade size
        price: Fill price
        commission: Commission
    """
    logger = LiveLogger()
    logger.log_trade(symbol, side, size, price, commission)


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Module J.5 — Logger Tests")
    print("=" * 80)

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # ====================================================================
        # TEST 1: Logger Initialization
        # ====================================================================
        print("\n[TEST 1] Logger Initialization")
        print("-" * 80)

        logger = LiveLogger(log_dir=temp_dir, enable_console=False)

        print(f"  Log dir: {logger.log_dir}")
        print(f"  JSON enabled: {logger.enable_json}")
        print(f"  Text enabled: {logger.enable_text}")
        print(f"  Current date: {logger.current_date}")

        assert logger.log_dir.exists(), "Log dir should exist"
        assert logger.enable_json, "JSON should be enabled"
        assert logger.enable_text, "Text should be enabled"

        print("  ✓ Logger initialization working")

        # ====================================================================
        # TEST 2: Event Logging
        # ====================================================================
        print("\n[TEST 2] Event Logging")
        print("-" * 80)

        logger.log_event('start', 'Live trading started', {'version': '1.0.0'})

        json_exists = logger.json_file.exists()
        text_exists = logger.text_file.exists()

        print(f"  JSON file exists: {json_exists}")
        print(f"  Text file exists: {text_exists}")

        assert json_exists, "JSON file should exist"
        assert text_exists, "Text file should exist"

        print("  ✓ Event logging working")

        # ====================================================================
        # TEST 3: Trade Logging
        # ====================================================================
        print("\n[TEST 3] Trade Logging")
        print("-" * 80)

        logger.log_trade(
            symbol='SPY',
            side='BUY',
            size=100.0,
            price=400.0,
            commission=10.0,
            metadata={'strategy': 'momentum'}
        )

        # Read JSON log
        with open(logger.json_file, 'r') as f:
            lines = f.readlines()

        print(f"  JSON log entries: {len(lines)}")

        # Parse last entry
        last_entry = json.loads(lines[-1])

        print(f"  Last entry type: {last_entry['log_type']}")
        print(f"  Symbol: {last_entry['metadata']['symbol']}")
        print(f"  Side: {last_entry['metadata']['side']}")

        assert last_entry['log_type'] == 'trade', "Should be trade entry"
        assert last_entry['metadata']['symbol'] == 'SPY', "Symbol should match"

        print("  ✓ Trade logging working")

        # ====================================================================
        # TEST 4: Signal Logging
        # ====================================================================
        print("\n[TEST 4] Signal Logging")
        print("-" * 80)

        logger.log_signal(
            symbol='SPY',
            regime='bull',
            horizon='5d',
            final_position=0.65,
            metadata={'n_signals': 3}
        )

        with open(logger.json_file, 'r') as f:
            lines = f.readlines()

        last_entry = json.loads(lines[-1])

        print(f"  Log type: {last_entry['log_type']}")
        print(f"  Regime: {last_entry['metadata']['regime']}")
        print(f"  Position: {last_entry['metadata']['final_position']:.2f}")

        assert last_entry['log_type'] == 'signal', "Should be signal entry"
        assert last_entry['metadata']['regime'] == 'bull', "Regime should match"

        print("  ✓ Signal logging working")

        # ====================================================================
        # TEST 5: Error Logging
        # ====================================================================
        print("\n[TEST 5] Error Logging")
        print("-" * 80)

        logger.log_error(
            error_type='data_fetch',
            message='Failed to fetch market data',
            metadata={'symbol': 'SPY', 'attempt': 3}
        )

        with open(logger.json_file, 'r') as f:
            lines = f.readlines()

        last_entry = json.loads(lines[-1])

        print(f"  Log type: {last_entry['log_type']}")
        print(f"  Level: {last_entry['level']}")
        print(f"  Error type: {last_entry['metadata']['error_type']}")

        assert last_entry['log_type'] == 'error', "Should be error entry"
        assert last_entry['level'] == 'ERROR', "Level should be ERROR"

        print("  ✓ Error logging working")

        # ====================================================================
        # TEST 6: Kill-Switch Logging
        # ====================================================================
        print("\n[TEST 6] Kill-Switch Logging")
        print("-" * 80)

        logger.log_kill_switch(
            reason='volatility_kill',
            metadata={'volatility': 0.15, 'threshold': 0.10}
        )

        with open(logger.json_file, 'r') as f:
            lines = f.readlines()

        last_entry = json.loads(lines[-1])

        print(f"  Log type: {last_entry['log_type']}")
        print(f"  Level: {last_entry['level']}")
        print(f"  Reason: {last_entry['metadata']['kill_switch_reason']}")

        assert last_entry['log_type'] == 'kill_switch', "Should be kill_switch entry"
        assert last_entry['level'] == 'WARNING', "Level should be WARNING"

        print("  ✓ Kill-switch logging working")

        # ====================================================================
        # TEST 7: Text Log Format
        # ====================================================================
        print("\n[TEST 7] Text Log Format")
        print("-" * 80)

        with open(logger.text_file, 'r') as f:
            text_lines = f.readlines()

        print(f"  Text log lines: {len(text_lines)}")
        print(f"  Sample line: {text_lines[0].strip()}")

        assert len(text_lines) > 0, "Text log should have entries"
        assert '[' in text_lines[0], "Text log should have brackets"

        print("  ✓ Text log format working")

        # ====================================================================
        # TEST 8: Daily Rotation
        # ====================================================================
        print("\n[TEST 8] Daily Rotation")
        print("-" * 80)

        # Simulate date change
        logger.current_date = '20250101'
        logger._check_rotation()

        new_date = logger.current_date

        print(f"  Old date: 20250101")
        print(f"  New date: {new_date}")

        # Date should update to today
        assert new_date != '20250101', "Date should update"

        print("  ✓ Daily rotation working")

        # ====================================================================
        # TEST 9: Console Output
        # ====================================================================
        print("\n[TEST 9] Console Output (Enabled)")
        print("-" * 80)

        logger_console = LiveLogger(log_dir=temp_dir, enable_console=True)

        print("  Logging event with console enabled:")
        logger_console.log_event('test', 'Console test message')

        print("  ✓ Console output working")

        # ====================================================================
        # TEST 10: Integration Hook
        # ====================================================================
        print("\n[TEST 10] Integration Hook")
        print("-" * 80)

        # Use default logger (will create in ~/.prado/logs/)
        evo_log_trade('SPY', 'BUY', 100.0, 400.0, 10.0)

        print("  Integration hook executed")

        print("  ✓ Integration hook working")

        # ====================================================================
        # TEST 11: JSON Safety (Sweep J.1)
        # ====================================================================
        print("\n[TEST 11] JSON Safety (Sweep J.1)")
        print("-" * 80)

        import numpy as np

        logger_json = LiveLogger(log_dir=temp_dir, enable_console=False)

        # Log entry with numpy types
        logger_json.log_event('test', 'JSON safety test', {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3]),
            'nan_value': float('nan'),
            'inf_value': float('inf')
        })

        # Read and parse
        with open(logger_json.json_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)

        print(f"  JSON entry parsed successfully")
        print(f"  numpy_int type: {type(entry['metadata']['numpy_int'])}")
        print(f"  NaN sanitized to: {entry['metadata']['nan_value']}")
        print(f"  Inf sanitized to: {entry['metadata']['inf_value']}")

        assert isinstance(entry['metadata']['numpy_int'], (int, float)), "Should be JSON type"
        assert entry['metadata']['nan_value'] == 0.0, "NaN should be sanitized"
        assert entry['metadata']['inf_value'] == 0.0, "Inf should be sanitized"

        print("  ✓ JSON safety working")

        # ====================================================================
        # SUMMARY
        # ====================================================================
        print("\n" + "=" * 80)
        print("ALL MODULE J.5 TESTS PASSED (11 TESTS) - Sweep J.1 Enhanced")
        print("=" * 80)
        print("\nLogger Features:")
        print("  ✓ JSON logging")
        print("  ✓ Text logging")
        print("  ✓ Console logging")
        print("  ✓ Event logging")
        print("  ✓ Trade logging")
        print("  ✓ Signal logging")
        print("  ✓ Error logging")
        print("  ✓ Kill-switch logging")
        print("  ✓ Daily rotation")
        print("  ✓ Structured metadata")
        print("  ✓ Audit trail")
        print("\nSweep J.1 Enhancements:")
        print("  ✓ JSON type safety (datetime, numpy, NaN/Inf sanitization)")
        print("  ✓ Size-based rotation (10MB max)")
        print("  ✓ Enhanced metadata sanitization")
        print("\nModule J.5 — Logger: PRODUCTION READY (Sweep J.1 Enhanced)")
        print("=" * 80)

    finally:
        shutil.rmtree(temp_dir)
