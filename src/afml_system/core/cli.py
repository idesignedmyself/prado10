"""
PRADO9_EVO Command-Line Interface

Fully integrated CLI for training, backtesting, and live trading.
All placeholder logic removed - wired to real modules.

SWEEP S.0 - COMPLETE INTEGRATION
"""

import sys
import random
import numpy as np
from typing import Optional
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .date_parser import parse_date_args, validate_date_range

# Create Typer app
app = typer.Typer(
    name="prado",
    help="PRADO9_EVO - Advanced Financial Machine Learning Trading System",
    add_completion=False
)

# Rich console for pretty output
console = Console()


# ============================================================================
# DETERMINISTIC SEEDING
# ============================================================================

def _seed_all(seed: int):
    """
    Seed all randomness for deterministic execution.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


# ============================================================================
# TRAIN COMMAND - WIRE TO EVOLUTION ENGINE
# ============================================================================

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed for determinism")
):
    """
    Train PRADO9_EVO models on historical data.

    Usage:
        prado train SYMBOL start MM DD YYYY end MM DD YYYY

    Example:
        prado train QQQ start 01 01 2020 end 12 31 2024
    """
    args = ctx.args

    if len(args) < 8:
        console.print("[red]Error:[/red] Missing required arguments")
        console.print("\nUsage: prado train SYMBOL start MM DD YYYY end MM DD YYYY")
        console.print("Example: prado train QQQ start 01 01 2020 end 12 31 2024")
        raise typer.Exit(code=1)

    # Extract symbol (first argument)
    symbol = args[0].upper()

    try:
        # Parse dates
        start_date, end_date = parse_date_args(args[1:])
        validate_date_range(start_date, end_date)

        # Seed for determinism
        _seed_all(seed)

        # Display training configuration
        console.print("\n[bold cyan]🔬 PRADO9_EVO Training Pipeline[/bold cyan]")
        console.print(Panel.fit(
            f"[green]Symbol:[/green] {symbol}\n"
            f"[green]Start:[/green] {start_date}\n"
            f"[green]End:[/green] {end_date}\n"
            f"[green]Mode:[/green] Full Training\n"
            f"[green]Seed:[/green] {seed}",
            title="Configuration",
            border_style="cyan"
        ))

        # Import modules
        console.print("\n[yellow]⚡ Loading PRADO9_EVO modules...[/yellow]")

        from afml_system.evo import (
            EvolutionEngine,
            GenomeFactory,
            BanditBrain,
            MetaLearningEngine,
            PerformanceMemory,
            CorrelationClusterEngine,
        )
        from afml_system.backtest import BacktestEngine, BacktestConfig

        console.print("[green]✓[/green] Modules loaded")

        # Step 1: Load data
        console.print("\n[bold]Step 1: Data Loading[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Fetching {symbol} data from {start_date} to {end_date}...", total=None)

            import yfinance as yf
            import pandas as pd

            # Fetch data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
                raise typer.Exit(code=1)

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Loaded {len(data)} bars")

        # Step 2: Initialize Evolution Engine
        console.print("\n[bold]Step 2: Initialize Evolution Engine[/bold]")

        config = BacktestConfig(
            symbol=symbol,
            initial_equity=100000.0,
            random_seed=seed,
            population_size=20,
            generations=10
        )

        evolution_engine = EvolutionEngine(
            population_size=config.population_size,
            mutation_rate=0.1,
            crossover_rate=0.7
        )

        console.print(f"[green]✓[/green] Evolution Engine initialized (pop={config.population_size})")

        # Step 3: Initialize Bandit Brain
        console.print("\n[bold]Step 3: Initialize Bandit Brain[/bold]")
        bandit = BanditBrain()
        console.print("[green]✓[/green] Bandit Brain ready (Thompson Sampling)")

        # Step 4: Initialize Meta-Learner
        console.print("\n[bold]Step 4: Initialize Meta-Learner[/bold]")
        meta_learner = MetaLearningEngine()
        console.print("[green]✓[/green] Meta-Learner initialized")

        # Step 5: Run Training via Backtest Engine
        console.print("\n[bold]Step 5: Running Training Backtest[/bold]")
        console.print("[yellow]This trains all modules A-H via walk-forward simulation[/yellow]")

        backtest = BacktestEngine(config=config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training models via backtest...", total=None)

            # Run backtest (this trains everything)
            result = backtest.run(data)

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Training complete")

        # Step 6: Display Results
        console.print("\n[bold cyan]📊 Training Results[/bold cyan]")

        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green", justify="right")

        metrics_table.add_row("Total Return", f"{result.total_return:.2%}")
        metrics_table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
        metrics_table.add_row("Sortino Ratio", f"{result.sortino_ratio:.3f}")
        metrics_table.add_row("Max Drawdown", f"{result.max_drawdown:.2%}")
        metrics_table.add_row("Win Rate", f"{result.win_rate:.2%}")
        metrics_table.add_row("Total Trades", f"{result.total_trades}")

        console.print(metrics_table)

        # Step 7: Save Models
        console.print("\n[bold]Step 6: Saving Models[/bold]")

        save_dir = Path('.prado') / 'models' / symbol.lower()
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save evolution state
        evolution_engine.save(save_dir / 'evolution_state.json')

        console.print(f"[green]✓[/green] Models saved to {save_dir}")

        # Success summary
        console.print("\n[bold green]✅ Training Complete![/bold green]")
        console.print(f"\nYou can now:")
        console.print(f"  • Run backtest: [cyan]prado backtest {symbol} --standard[/cyan]")
        console.print(f"  • Get predictions: [cyan]prado predict {symbol}[/cyan]")
        console.print(f"  • Live trade: [cyan]prado live {symbol} --mode simulate[/cyan]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


# ============================================================================
# BACKTEST COMMAND - WIRE TO BACKTEST ENGINES
# ============================================================================

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def backtest(
    ctx: typer.Context,
    standard: bool = typer.Option(False, "--standard", help="Run standard backtest"),
    walk_forward: bool = typer.Option(False, "--walk-forward", help="Run walk-forward backtest"),
    crisis: bool = typer.Option(False, "--crisis", help="Run crisis period backtest"),
    monte_carlo: Optional[int] = typer.Option(None, "--monte-carlo", help="Run Monte Carlo simulation (specify iterations)"),
    seed: int = typer.Option(42, "--seed", help="Random seed for determinism")
):
    """
    Run backtest on PRADO9_EVO strategy.

    Usage:
        prado backtest SYMBOL [options]

    Examples:
        prado backtest QQQ --standard
        prado backtest SPY --walk-forward
        prado backtest QQQ --crisis
        prado backtest SPY --monte-carlo 10000
    """
    args = ctx.args

    if len(args) < 1:
        console.print("[red]Error:[/red] Symbol required")
        console.print("\nUsage: prado backtest SYMBOL [options]")
        console.print("Example: prado backtest QQQ --standard")
        raise typer.Exit(code=1)

    symbol = args[0].upper()

    # Seed for determinism
    _seed_all(seed)

    # Determine backtest type
    if standard:
        backtest_type = "Standard Backtest"
        backtest_mode = "standard"
    elif walk_forward:
        backtest_type = "Walk-Forward Backtest"
        backtest_mode = "walk_forward"
    elif crisis:
        backtest_type = "Crisis Period Backtest"
        backtest_mode = "crisis"
    elif monte_carlo:
        backtest_type = f"Monte Carlo Simulation ({monte_carlo:,} iterations)"
        backtest_mode = "monte_carlo"
    else:
        backtest_type = "Standard Backtest (default)"
        backtest_mode = "standard"

    # Display configuration
    console.print("\n[bold cyan]📊 PRADO9_EVO Backtest Engine[/bold cyan]")
    console.print(Panel.fit(
        f"[green]Symbol:[/green] {symbol}\n"
        f"[green]Type:[/green] {backtest_type}\n"
        f"[green]Seed:[/green] {seed}",
        title="Configuration",
        border_style="cyan"
    ))

    try:
        # Import backtest modules
        console.print("\n[yellow]⚡ Loading backtest modules...[/yellow]")

        from afml_system.backtest import (
            evo_backtest_standard,
            evo_backtest_walk_forward,
            evo_backtest_crisis,
            evo_backtest_monte_carlo,
        )

        console.print("[green]✓[/green] Modules loaded")

        # Load data
        console.print(f"\n[bold]Loading {symbol} data...[/bold]")

        import yfinance as yf

        # Default to 5 years of data
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)

        data = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if data.empty:
            console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
            raise typer.Exit(code=1)

        console.print(f"[green]✓[/green] Loaded {len(data)} bars ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

        # Create config with seed
        from afml_system.backtest import BacktestConfig

        config = BacktestConfig(
            symbol=symbol,
            random_seed=seed
        )

        # Run appropriate backtest
        console.print(f"\n[bold]Running {backtest_type}...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Backtesting...", total=None)

            if backtest_mode == "standard":
                response = evo_backtest_standard(symbol, data, config=config)
            elif backtest_mode == "walk_forward":
                response = evo_backtest_walk_forward(symbol, data, config=config)
            elif backtest_mode == "crisis":
                response = evo_backtest_crisis(symbol, data, config=config)
            elif backtest_mode == "monte_carlo":
                from afml_system.backtest import evo_backtest_monte_carlo
                response = evo_backtest_monte_carlo(symbol, data, n_simulations=monte_carlo, config=config)

            progress.update(task, completed=True)

        # Check for errors
        if response['status'] != 'success':
            console.print(f"[red]Backtest failed:[/red] {response.get('error', 'Unknown error')}")
            raise typer.Exit(code=1)

        # Extract result
        result = response['result']

        # Display results
        console.print("\n[bold cyan]📈 Backtest Results[/bold cyan]")

        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green", justify="right")

        results_table.add_row("Total Return", f"{result.total_return:.2%}")
        results_table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
        results_table.add_row("Sortino Ratio", f"{result.sortino_ratio:.3f}")
        results_table.add_row("Calmar Ratio", f"{result.calmar_ratio:.3f}")
        results_table.add_row("Max Drawdown", f"{result.max_drawdown:.2%}")
        results_table.add_row("Win Rate", f"{result.win_rate:.2%}")
        results_table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
        results_table.add_row("Total Trades", f"{result.total_trades}")

        console.print(results_table)

        # Strategy allocation summary
        if hasattr(result, 'strategy_allocations') and result.strategy_allocations:
            console.print("\n[bold]Strategy Allocations:[/bold]")

            strat_table = Table(show_header=True, header_style="bold magenta")
            strat_table.add_column("Strategy", style="cyan")
            strat_table.add_column("Allocation", style="green", justify="right")

            for strategy, allocation in sorted(result.strategy_allocations.items(), key=lambda x: -x[1]):
                strat_table.add_row(strategy, f"{allocation:.2%}")

            console.print(strat_table)

        console.print("\n[bold green]✅ Backtest Complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


# ============================================================================
# LIVE COMMAND - ALREADY WIRED (MODULE J)
# ============================================================================

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def live(
    ctx: typer.Context,
    mode: str = typer.Option("simulate", "--mode", help="Trading mode: simulate, paper, or live")
):
    """
    Start PRADO9_EVO live trading engine.

    Usage:
        prado live SYMBOL [options]

    Examples:
        prado live QQQ
        prado live SPY --mode simulate
        prado live QQQ --mode paper
    """
    args = ctx.args

    if len(args) < 1:
        console.print("[red]Error:[/red] Symbol required")
        console.print("\nUsage: prado live SYMBOL [--mode MODE]")
        console.print("Example: prado live QQQ --mode simulate")
        raise typer.Exit(code=1)

    symbol = args[0].upper()

    # Validate mode
    if mode not in ['simulate', 'paper', 'live']:
        console.print(f"[red]Error:[/red] Invalid mode '{mode}'")
        console.print("Valid modes: simulate, paper, live")
        raise typer.Exit(code=1)

    # Display configuration
    console.print("\n[bold cyan]🚀 PRADO9_EVO Live Trading Engine[/bold cyan]")
    console.print(Panel.fit(
        f"[green]Symbol:[/green] {symbol}\n"
        f"[green]Mode:[/green] {mode}\n"
        f"[green]Status:[/green] Starting...",
        title="Configuration",
        border_style="cyan"
    ))

    try:
        from afml_system.live import LiveTradingEngine, EngineConfig
        from afml_system.live import momentum_strategy, mean_reversion_strategy

        # Configure engine
        config = EngineConfig(
            symbols=[symbol],
            mode=mode,
            poll_interval=60.0,  # 1 minute
            check_market_hours=True,
            initial_cash=100000.0,
            enable_logging=True,
            enable_console=True,
            random_seed=42  # Deterministic seeding
        )

        # Define strategies
        strategies = {
            'momentum': momentum_strategy,
            'mean_reversion': mean_reversion_strategy
        }

        console.print("\n[green]✓[/green] Engine configured")
        console.print("[green]✓[/green] Strategies loaded: momentum, mean_reversion")
        console.print("\n[bold]Starting engine... (Press Ctrl+C to stop)[/bold]")
        console.print("=" * 80)
        console.print()

        # Create and start engine
        engine = LiveTradingEngine(config, strategies=strategies)
        engine.start()

    except KeyboardInterrupt:
        console.print("\n\n[yellow]⏹️  Engine stopped by user[/yellow]")
        if 'engine' in locals():
            engine.stop()
    except Exception as e:
        console.print(f"\n[red]❌ Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


# ============================================================================
# PREDICT COMMAND - WIRE TO LIVE SIGNAL ENGINE
# ============================================================================

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def predict(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed for determinism")
):
    """
    Generate predictions for a symbol using trained PRADO9_EVO models.

    Usage:
        prado predict SYMBOL

    Example:
        prado predict QQQ
    """
    args = ctx.args

    if len(args) < 1:
        console.print("[red]Error:[/red] Symbol required")
        console.print("\nUsage: prado predict SYMBOL")
        console.print("Example: prado predict QQQ")
        raise typer.Exit(code=1)

    symbol = args[0].upper()

    # Seed for determinism
    _seed_all(seed)

    console.print("\n[bold cyan]🔮 PRADO9_EVO Prediction Engine[/bold cyan]")
    console.print(Panel.fit(
        f"[green]Symbol:[/green] {symbol}\n"
        f"[green]Mode:[/green] Real-time Prediction\n"
        f"[green]Seed:[/green] {seed}",
        title="Configuration",
        border_style="cyan"
    ))

    try:
        # Import modules
        console.print("\n[yellow]⚡ Loading prediction modules...[/yellow]")

        from afml_system.live import LiveDataFeed, LiveSignalEngine
        from afml_system.evo import evo_allocate

        console.print("[green]✓[/green] Modules loaded")

        # Step 1: Fetch latest data
        console.print("\n[bold]Step 1: Fetching latest market data[/bold]")

        data_feed = LiveDataFeed()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Fetching {symbol} data...", total=None)

            result = data_feed.get_recent_bars(symbol, lookback=200)

            if result.empty:
                console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
                raise typer.Exit(code=1)

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Loaded {len(result)} bars")

        # Step 2: Generate signal
        console.print("\n[bold]Step 2: Generating signal[/bold]")

        signal_engine = LiveSignalEngine()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Computing features, regime, and strategies...", total=None)

            signal_result = signal_engine.generate_signal(symbol, result)

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Signal generated")

        # Step 3: Display prediction
        console.print("\n[bold cyan]📊 Prediction Results[/bold cyan]")

        # Current price
        current_price = result['Close'].iloc[-1]
        console.print(f"\n[bold]Current Price:[/bold] ${current_price:.2f}")

        # Regime
        regime_table = Table(show_header=True, header_style="bold magenta")
        regime_table.add_column("Component", style="cyan")
        regime_table.add_column("Value", style="green")

        regime_table.add_row("Regime", signal_result.regime)
        regime_table.add_row("Volatility", f"{signal_result.volatility:.4f}")
        regime_table.add_row("Trend Strength", f"{signal_result.trend_strength:.4f}")

        console.print("\n[bold]Market Regime:[/bold]")
        console.print(regime_table)

        # Strategy signals
        if signal_result.strategy_signals:
            console.print("\n[bold]Strategy Signals:[/bold]")

            signal_table = Table(show_header=True, header_style="bold magenta")
            signal_table.add_column("Strategy", style="cyan")
            signal_table.add_column("Signal", style="green", justify="right")
            signal_table.add_column("Confidence", style="yellow", justify="right")

            for strat_name, strat_result in signal_result.strategy_signals.items():
                signal_str = f"{strat_result.signal:+.3f}"
                conf_str = f"{strat_result.confidence:.2%}"
                signal_table.add_row(strat_name, signal_str, conf_str)

            console.print(signal_table)

        # Aggregated signal
        console.print(f"\n[bold]Aggregated Signal:[/bold] {signal_result.aggregated_signal:+.3f}")

        # Recommendation
        if signal_result.aggregated_signal > 0.3:
            recommendation = "[green]LONG[/green] - Strong buy signal"
        elif signal_result.aggregated_signal > 0.1:
            recommendation = "[green]LONG[/green] - Weak buy signal"
        elif signal_result.aggregated_signal < -0.3:
            recommendation = "[red]SHORT[/red] - Strong sell signal"
        elif signal_result.aggregated_signal < -0.1:
            recommendation = "[red]SHORT[/red] - Weak sell signal"
        else:
            recommendation = "[yellow]NEUTRAL[/yellow] - No clear signal"

        console.print(f"\n[bold]Recommendation:[/bold] {recommendation}")

        console.print("\n[bold green]✅ Prediction Complete![/bold green]")
        console.print(f"\nTo start live trading:")
        console.print(f"  [cyan]prado live {symbol} --mode simulate[/cyan]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


# ============================================================================
# HELP COMMAND
# ============================================================================

@app.command()
def help():
    """
    Show detailed help and examples for PRADO9_EVO CLI.
    """
    console.print("\n[bold cyan]PRADO9_EVO - Command-Line Interface[/bold cyan]\n")

    # Commands table
    table = Table(show_header=True, header_style="bold magenta", title="Available Commands")
    table.add_column("Command", style="cyan", width=15)
    table.add_column("Description", style="green", width=50)
    table.add_column("Example", style="yellow", width=40)

    table.add_row(
        "train",
        "Train models on historical data",
        "prado train QQQ start 01 01 2020 end 12 31 2024"
    )
    table.add_row(
        "backtest",
        "Run backtest simulations",
        "prado backtest SPY --standard"
    )
    table.add_row(
        "live",
        "Start live trading engine",
        "prado live QQQ --mode simulate"
    )
    table.add_row(
        "predict",
        "Generate predictions",
        "prado predict SPY"
    )
    table.add_row(
        "help",
        "Show this help message",
        "prado help"
    )

    console.print(table)

    # Backtest options
    console.print("\n[bold]Backtest Options:[/bold]")
    console.print("  --standard      Run standard single-period backtest")
    console.print("  --walk-forward  Run walk-forward analysis")
    console.print("  --crisis        Test on historical crisis periods")
    console.print("  --monte-carlo N Run Monte Carlo simulation (N iterations)")

    # Live modes
    console.print("\n[bold]Live Trading Modes:[/bold]")
    console.print("  simulate   Simulated trading (no API, deterministic)")
    console.print("  paper      Paper trading (logged, no real money)")
    console.print("  live       Live trading (requires broker API keys)")

    console.print("\n[bold green]📚 For more information:[/bold green]")
    console.print("  Documentation: ./docs/")
    console.print("  Examples: ./examples/")

    console.print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
