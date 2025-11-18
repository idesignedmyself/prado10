# SWEEP J.1 — Final Hardening and Determinism Sweep

## Overview
This sweep hardens Module J (Live Trading Engine) with determinism, stability, and safety enhancements. **Simulate mode only.** No architecture changes.

---

## SCOPE: 6 Files to Modify

1. `src/afml_system/live/live_engine.py`
2. `src/afml_system/live/data_feed.py`
3. `src/afml_system/live/signal_engine.py`
4. `src/afml_system/live/broker_router.py`
5. `src/afml_system/live/live_portfolio.py`
6. `src/afml_system/live/logger.py`

**Do NOT:**
- Create new files
- Change architecture
- Add streaming/live modes
- Add new broker integrations
- Add complexity

**DO:**
- Enhance existing code only
- Add 11+ new inline tests
- Ensure full determinism
- Maintain simulate mode focus
- Keep yfinance polling only

---

## ENHANCEMENT AREAS (6 Total)

### 1. Deterministic Live Simulation (CRITICAL)
**File:** `live_engine.py`

**Add:**
- `_seed_all(seed)` method to seed numpy/random
- Deterministic timestamp generation for tests
- Seed parameter in EngineConfig
- Ensure consistent execution order
- Deterministic sleep intervals (use tick counts, not real time)

**Tests to Add:**
- TEST: Deterministic polling (same seed → same behavior)
- TEST: Deterministic signals (same seed → same signals)
- TEST: Deterministic full loop (100 ticks identical across runs)

**Implementation:**
```python
class EngineConfig:
    # Add field
    random_seed: Optional[int] = None

class LiveTradingEngine:
    def _seed_all(self, seed: int):
        """Seed all randomness for determinism."""
        np.random.seed(seed)
        import random
        random.seed(seed)

    def __init__(...):
        # Call seeding if configured
        if config.random_seed is not None:
            self._seed_all(config.random_seed)
```

---

### 2. Polling Stability & Fallback Logic
**File:** `data_feed.py`

**Add:**
- Retry layer (max 3 retries with exponential backoff)
- Fallback to last-good candle on API failure
- Stale timestamp detection (reject data older than threshold)
- Minimum lookback enforcement (ensure at least N bars)
- Enhanced DataFrame sanitization

**Tests to Add:**
- TEST: Feed failure fallback (API fails → use cached data)
- TEST: Stale candle rejection (old data → rejected)

**Implementation:**
```python
class LiveDataFeed:
    def __init__(...):
        self.max_retries = 3
        self.min_bars_required = 50
        self.max_data_age_seconds = 3600  # 1 hour

    def get_recent_bars(self, symbol, lookback):
        """Enhanced with retry and fallback."""
        for attempt in range(self.max_retries):
            try:
                result = self._fetch_data(...)
                if self._validate_data(result):
                    return result.df
                # Backoff
                time.sleep(2 ** attempt)
            except Exception:
                if attempt == self.max_retries - 1:
                    # Final fallback to cache
                    return self._load_from_cache(symbol)
        return self._load_from_cache(symbol)

    def _validate_data(self, result):
        """Check data freshness and completeness."""
        if result.df.empty:
            return False
        if len(result.df) < self.min_bars_required:
            return False
        # Check timestamp freshness
        latest_ts = result.df.index[-1]
        age = (datetime.now() - latest_ts).total_seconds()
        if age > self.max_data_age_seconds:
            return False
        return True
```

---

### 3. Simulated Execution Safety
**File:** `broker_router.py`

**Simplify to simulate-only:**
- Remove live/paper mode complexity
- Keep ONLY simulate mode
- Enforce position limits strictly
- Prevent overfills
- Prevent negative balances
- Ensure all fills are finite

**Tests to Add:**
- TEST: Simulate fill correctness (position updates match expected)
- TEST: Simulate fill determinism (same inputs → same fills)

**Implementation:**
```python
class BrokerRouter:
    """Simulate-only broker router (Sweep J.1)."""

    def __init__(self):
        """Simulate mode only - no API keys needed."""
        self.mode = 'simulate'  # Hard-coded
        self.positions: Dict[str, float] = {}
        self.cash: float = 100000.0

    def submit_order(self, symbol, side, size, price=None):
        """Execute simulated order with safety checks."""
        # Enforce position limits
        new_pos = self._compute_new_position(symbol, side, size)
        if abs(new_pos) > 1.0:
            raise ValueError("Position limit exceeded")

        # Prevent negative cash
        cost = self._compute_cost(size, price or 100.0)
        if self.cash - cost < 0:
            raise ValueError("Insufficient cash")

        # Execute
        fill = self._execute_simulated(...)

        # Validate fill is finite
        assert np.isfinite(fill.price)
        assert np.isfinite(fill.size)

        return fill
```

---

### 4. Portfolio Persistence & Crash Recovery
**File:** `live_portfolio.py`

**Add:**
- Atomic JSON writes (write to temp file → rename)
- Recovery validation on load
- Repair missing/corrupted fields
- Ensure all numeric fields are finite
- Sanitize trade history on load

**Tests to Add:**
- TEST: Save → reload → identical state (full round-trip)

**Implementation:**
```python
class LivePortfolio:
    def save(self):
        """Atomic save with temp file."""
        state = self.get_state()

        # Ensure all values are JSON-safe
        state = self._sanitize_for_json(state)

        # Atomic write
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Atomic rename
        temp_file.replace(self.state_file)

    def load(self):
        """Load with validation and repair."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Validate and repair
            state = self._validate_and_repair(state)

            # Restore
            self._restore_from_state(state)

        except Exception:
            # Corruption - use defaults
            pass

    def _validate_and_repair(self, state):
        """Ensure all fields present and finite."""
        # Repair missing fields
        defaults = {
            'cash': self.initial_cash,
            'equity': self.initial_cash,
            'positions': {},
            'unrealized_pnl': 0.0,
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
        }
        for key, default in defaults.items():
            if key not in state:
                state[key] = default

        # Ensure finite
        for key in ['cash', 'equity', 'unrealized_pnl', 'daily_pnl', 'total_pnl']:
            state[key] = _safe_float(state[key], defaults[key])

        return state
```

---

### 5. Logging Hardening
**File:** `logger.py`

**Add:**
- JSON-safe type enforcement (convert datetime, numpy types)
- Rotation by date (already exists)
- Rotation by size (optional, 10MB max)
- Heartbeat logging (every tick)
- Enhanced event logging (signal + trade + allocation summary)
- Fallback event logging (when data feed fails)

**Tests to Add:**
- TEST: JSON logs valid (all entries parseable)
- TEST: Rotation logic deterministic (date changes → new file)

**Implementation:**
```python
class LiveLogger:
    def __init__(...):
        self.max_file_size = 10 * 1024 * 1024  # 10MB

    def _write_entry(self, entry: LogEntry):
        """Write with size-based rotation."""
        self._check_rotation()

        # Check file size
        if self.json_file.exists():
            if self.json_file.stat().st_size > self.max_file_size:
                self._rotate_by_size()

        # Ensure JSON-safe
        entry = self._sanitize_entry(entry)

        # Write
        if self.enable_json:
            with open(self.json_file, 'a') as f:
                f.write(entry.to_json_line() + '\n')

    def _sanitize_entry(self, entry):
        """Ensure all metadata is JSON-safe."""
        metadata = {}
        for key, value in entry.metadata.items():
            if isinstance(value, (datetime, np.datetime64)):
                metadata[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
            elif isinstance(value, (np.integer, np.floating)):
                metadata[key] = float(value)
            elif isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            else:
                metadata[key] = value
        entry.metadata = metadata
        return entry

    def log_heartbeat(self, tick: int, status: Dict):
        """Log heartbeat every tick."""
        self.log_event('heartbeat', f'Tick {tick}', {
            'tick': tick,
            'equity': status.get('equity', 0.0),
            'position': status.get('position', 0.0)
        })

    def log_fallback(self, symbol: str, reason: str):
        """Log data feed fallback."""
        self.log_event('data_fallback', f'{symbol}: {reason}', {
            'symbol': symbol,
            'reason': reason
        }, level='WARNING')
```

---

### 6. Final End-to-End Simulate Session Test
**File:** `live_engine.py` (in `if __name__ == "__main__"` section)

**Add:**
- TEST: 100-tick deterministic simulation
  - Use synthetic data (no real API calls)
  - Validate deterministic behavior
  - Ensure no crashes
  - Verify kill-switches trigger correctly
  - Check portfolio remains finite
  - Verify logs produced
  - Verify portfolio saved
  - Verify recovery matches final state

**Implementation:**
```python
if __name__ == "__main__":
    # ... existing tests ...

    # ====================================================================
    # TEST 11: 100-Tick Deterministic End-to-End Simulation
    # ====================================================================
    print("\n[TEST 11] 100-Tick Deterministic End-to-End Simulation")
    print("-" * 80)

    import pandas as pd
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create synthetic data feed
        def create_synthetic_data(symbol, lookback):
            """Generate deterministic synthetic OHLCV data."""
            np.random.seed(42)  # Deterministic
            dates = pd.date_range('2024-01-01', periods=lookback, freq='D')
            df = pd.DataFrame({
                'Open': 100 + np.cumsum(np.random.randn(lookback) * 2),
                'High': 105 + np.cumsum(np.random.randn(lookback) * 2),
                'Low': 95 + np.cumsum(np.random.randn(lookback) * 2),
                'Close': 100 + np.cumsum(np.random.randn(lookback) * 2),
                'Volume': np.random.randint(1000000, 10000000, lookback)
            }, index=dates)
            return df

        # Configure engine
        config = EngineConfig(
            symbols=['SPY'],
            mode='simulate',
            poll_interval=0.01,  # Fast for testing
            check_market_hours=False,
            random_seed=42,  # Deterministic
            enable_logging=True,
            enable_console=False
        )

        # Create engine
        engine = LiveTradingEngine(config, strategies={
            'momentum': momentum_strategy,
            'mean_reversion': mean_reversion_strategy
        })

        # Mock data feed
        engine.data_feed.get_recent_bars = lambda symbol, lookback: create_synthetic_data(symbol, lookback)

        # Run 100 ticks
        for i in range(100):
            engine._process_symbol('SPY')
            engine.tick_count += 1

        # Validate results
        status = engine.get_status()
        portfolio = status['portfolios']['SPY']

        print(f"  Ticks processed: {engine.tick_count}")
        print(f"  Final equity: ${portfolio['equity']:,.2f}")
        print(f"  Final position: {portfolio['position']:.2f}")
        print(f"  Total P&L: ${portfolio['total_pnl']:,.2f}")

        # Check finite
        assert np.isfinite(portfolio['equity']), "Equity should be finite"
        assert np.isfinite(portfolio['position']), "Position should be finite"
        assert np.isfinite(portfolio['total_pnl']), "P&L should be finite"

        # Check logs exist
        log_files = list(engine.logger.log_dir.glob('*.log'))
        assert len(log_files) > 0, "Should have log files"

        # Test crash recovery
        portfolio_obj = engine.portfolios['SPY']
        state_before = portfolio_obj.get_state()

        # Save
        portfolio_obj.save()

        # Create new portfolio (load from disk)
        portfolio_new = LivePortfolio(symbol='SPY', initial_cash=100000.0, state_dir=temp_dir)
        state_after = portfolio_new.get_state()

        # Verify identical
        assert abs(state_before['equity'] - state_after['equity']) < 1e-6, "Equity should match"
        assert abs(state_before['cash'] - state_after['cash']) < 1e-6, "Cash should match"

        print("  ✓ 100-tick simulation complete")
        print("  ✓ All values finite")
        print("  ✓ Logs produced")
        print("  ✓ Crash recovery verified")

        # Run AGAIN with same seed - should be identical
        engine2 = LiveTradingEngine(config, strategies={
            'momentum': momentum_strategy,
            'mean_reversion': mean_reversion_strategy
        })
        engine2.data_feed.get_recent_bars = lambda symbol, lookback: create_synthetic_data(symbol, lookback)

        for i in range(100):
            engine2._process_symbol('SPY')
            engine2.tick_count += 1

        status2 = engine2.get_status()
        portfolio2 = status2['portfolios']['SPY']

        # Check determinism
        assert abs(portfolio['equity'] - portfolio2['equity']) < 1e-10, "Should be deterministic"
        assert abs(portfolio['position'] - portfolio2['position']) < 1e-10, "Should be deterministic"

        print("  ✓ Determinism verified (identical across runs)")

    finally:
        shutil.rmtree(temp_dir)

    print("  ✓ End-to-end 100-tick simulation working")
```

---

## TEST REQUIREMENTS

**Minimum 11 new tests across all files:**

1. ✓ TEST: Deterministic polling (live_engine.py)
2. ✓ TEST: Deterministic signals (live_engine.py)
3. ✓ TEST: Deterministic full loop (live_engine.py)
4. ✓ TEST: Feed failure fallback (data_feed.py)
5. ✓ TEST: Stale candle rejection (data_feed.py)
6. ✓ TEST: Simulate fill correctness (broker_router.py)
7. ✓ TEST: Simulate fill determinism (broker_router.py)
8. ✓ TEST: Save → reload → identical state (live_portfolio.py)
9. ✓ TEST: JSON logs valid (logger.py)
10. ✓ TEST: Rotation logic deterministic (logger.py)
11. ✓ TEST: 100-tick end-to-end simulation (live_engine.py)

**All tests must:**
- Be inline (in `if __name__ == "__main__"`)
- Pass with assertions
- Test determinism where applicable
- Use simulate mode only
- Require no external APIs

---

## VALIDATION CHECKLIST

Before completion, verify:

- [ ] All 6 files updated
- [ ] All 11+ tests added
- [ ] All tests passing
- [ ] Determinism verified (same seed → same output)
- [ ] No architecture changes
- [ ] No new files created
- [ ] Simulate mode only (no live/paper complexity)
- [ ] yfinance polling only (no new data sources)
- [ ] All numeric values finite (no NaN/Inf)
- [ ] All JSON logs parseable
- [ ] Crash recovery working
- [ ] No placeholders/TODOs

---

## IMPLEMENTATION ORDER

1. **live_engine.py** - Add seeding + deterministic tests
2. **data_feed.py** - Add retry/fallback + validation tests
3. **broker_router.py** - Simplify to simulate-only + safety tests
4. **live_portfolio.py** - Add atomic persistence + recovery test
5. **logger.py** - Add JSON safety + rotation tests
6. **signal_engine.py** - Minor enhancements (if needed for determinism)

---

## SUCCESS CRITERIA

**Sweep J.1 is complete when:**

✓ All 6 files enhanced with documented changes
✓ All 11+ inline tests passing
✓ Full determinism verified (100-tick test passes twice identically)
✓ No crashes in 100-tick simulation
✓ All portfolios remain finite
✓ Logs valid and parseable
✓ Crash recovery verified
✓ No architecture changes
✓ Code clean (no placeholders)

**Expected output:**
```
SWEEP J.1 COMPLETE - ALL TESTS PASSING (11+ tests)
Module J hardened with determinism and stability enhancements
Ready for production deployment in simulate mode
```

---

## NOTES

- Keep changes **minimal and focused**
- Maintain **existing architecture**
- Prioritize **determinism** above all else
- Every change must have a **test**
- No new features - **hardening only**

---

**Ready to implement Sweep J.1**

Next step: Open new window, read this file, implement all enhancements.
