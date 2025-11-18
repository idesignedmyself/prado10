"""
PRADO9_EVO Live Trading Engine - Demonstration

This script demonstrates the complete live trading system in simulate mode.

WARNING: This is for demonstration only. For live trading:
1. Set mode='paper' for paper trading
2. Set mode='live' for real trading (requires API keys and broker integration)
3. Always test thoroughly in simulate/paper before going live
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from afml_system.live import (
    LiveTradingEngine,
    EngineConfig,
    momentum_strategy,
    mean_reversion_strategy
)

print("=" * 80)
print("PRADO9_EVO Live Trading Engine - Demonstration")
print("=" * 80)
print("\nThis demo shows the live trading engine in SIMULATE mode.")
print("The engine will:")
print("  1. Fetch market data (cached/simulated)")
print("  2. Generate signals from strategies")
print("  3. Execute simulated trades")
print("  4. Track portfolio state")
print("  5. Log all activity")
print("\nPress Ctrl+C to stop gracefully.")
print("=" * 80)

# Configure engine
config = EngineConfig(
    symbols=['SPY'],              # Trade SPY
    mode='simulate',              # Simulate mode (no real money)
    poll_interval=5.0,            # Check every 5 seconds (fast for demo)
    check_market_hours=False,     # Ignore market hours for demo
    initial_cash=100000.0,        # Start with $100k
    max_position=1.0,             # Max 100% long or short
    max_daily_loss=0.05,          # Stop if lose 5% in a day
    enable_logging=True,          # Log everything
    enable_console=True           # Show logs in console
)

# Define strategies
strategies = {
    'momentum': momentum_strategy,
    'mean_reversion': mean_reversion_strategy
}

# Create engine
engine = LiveTradingEngine(config, strategies=strategies)

print("\n[INIT] Engine initialized")
print(f"  Symbols: {config.symbols}")
print(f"  Mode: {config.mode}")
print(f"  Poll interval: {config.poll_interval}s")
print(f"  Initial cash: ${config.initial_cash:,.0f}")

# Show initial status
status = engine.get_status()
print(f"\n[STATUS] Initial State")
for symbol, portfolio in status['portfolios'].items():
    print(f"  {symbol}:")
    print(f"    Equity: ${portfolio['equity']:,.2f}")
    print(f"    Cash: ${portfolio['cash']:,.2f}")
    print(f"    Position: {portfolio['position']:.2f} shares")

print("\n[START] Starting live trading engine...")
print("  Processing will run for 30 seconds (6 ticks)")
print("  Each tick will:")
print("    - Fetch market data")
print("    - Generate signals")
print("    - Execute trades (if signal changes)")
print("    - Update portfolio")
print("    - Log activity")
print()

# Start engine in a thread and let it run for 30 seconds
try:
    # Start the engine
    import threading

    def run_engine():
        engine.start()

    engine_thread = threading.Thread(target=run_engine, daemon=True)
    engine_thread.start()

    # Let it run for 30 seconds
    for i in range(30):
        time.sleep(1)
        if i % 5 == 0 and i > 0:
            # Show status every 5 seconds
            status = engine.get_status()
            print(f"\n[STATUS] After {engine.tick_count} ticks")
            for symbol, portfolio in status['portfolios'].items():
                print(f"  {symbol}:")
                print(f"    Equity: ${portfolio['equity']:,.2f}")
                print(f"    Daily P&L: ${portfolio['daily_pnl']:,.2f}")
                print(f"    Position: {portfolio['position']:.2f} shares")
                if portfolio['kill_flags']:
                    print(f"    Kill flags: {portfolio['kill_flags']}")

    # Stop engine
    print("\n[STOP] Stopping engine...")
    engine.stop()

    # Final status
    status = engine.get_status()
    print(f"\n[FINAL] Engine Statistics")
    print(f"  Total ticks processed: {status['tick_count']}")
    print(f"  Mode: {status['mode']}")

    for symbol, portfolio in status['portfolios'].items():
        print(f"\n  {symbol} Portfolio:")
        print(f"    Final equity: ${portfolio['equity']:,.2f}")
        print(f"    Cash: ${portfolio['cash']:,.2f}")
        print(f"    Position: {portfolio['position']:.2f} shares")
        print(f"    Daily P&L: ${portfolio['daily_pnl']:,.2f}")
        print(f"    Total P&L: ${portfolio['total_pnl']:,.2f}")

    print("\n[LOGS] Activity logged to:")
    print(f"  JSON: ~/.prado/logs/live_*.json")
    print(f"  Text: ~/.prado/logs/live_*.log")

    print("\n[PORTFOLIO] State saved to:")
    print(f"  ~/.prado/live/portfolio/SPY.json")

except KeyboardInterrupt:
    print("\n\n[INTERRUPT] User stopped engine")
    engine.stop()

print("\n" + "=" * 80)
print("Demo complete!")
print("\nTo view logs:")
print("  cat ~/.prado/logs/live_*.log")
print("\nTo view portfolio state:")
print("  cat ~/.prado/live/portfolio/SPY.json")
print("=" * 80)
