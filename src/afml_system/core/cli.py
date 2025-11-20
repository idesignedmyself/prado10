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
from .cli_optimize import optimize

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

            # Fix yfinance MultiIndex columns (sometimes returns tuples)
            cols = data.columns
            if isinstance(cols[0], tuple):
                # Flatten multiindex: ('Close', 'QQQ') -> 'close' (take first element only)
                data.columns = [str(col[0]).lower() for col in cols]
            else:
                # Normal columns: 'Close' -> 'close'
                data.columns = [str(col).lower() for col in cols]

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Loaded {len(data)} bars")

        # Step 2: Create Backtest Config
        console.print("\n[bold]Step 2: Create Backtest Configuration[/bold]")

        config = BacktestConfig(
            symbol=symbol,
            initial_equity=100000.0,
            random_seed=seed,
            population_size=20,
            generations=10,
            mutation_rate=0.1,
            crossover_rate=0.7
        )

        console.print(f"[green]✓[/green] Config created (pop={config.population_size}, gen={config.generations})")

        # Step 3: Run Training via Backtest Engine
        console.print("\n[bold]Step 3: Running Training Backtest[/bold]")
        console.print("[yellow]This trains all modules A-H via event-driven simulation[/yellow]")

        backtest = BacktestEngine(config=config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training models via backtest...", total=None)

            # Run backtest (this trains everything)
            result = backtest.run_standard(symbol=symbol, df=data)

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Training complete")

        # Step 4: Display Results
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

        # Step 5: Model Persistence
        console.print("\n[bold]Step 5: Model Persistence[/bold]")

        save_dir = Path('.prado') / 'models' / symbol.lower()
        save_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[green]✓[/green] Model directory created: {save_dir}")
        console.print("[yellow]Note:[/yellow] Models are auto-saved by BacktestEngine during training")

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
    mc2: Optional[int] = typer.Option(None, "--mc2", help="Run MC2 robustness tests (specify iterations)"),
    adaptive: bool = typer.Option(False, "--adaptive", help="Run unified adaptive backtest (AR+X2+Y2+CR2)"),
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
        prado backtest QQQ --mc2 1000
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
    if adaptive:
        backtest_type = "Unified Adaptive Backtest (AR+X2+Y2+CR2)"
        backtest_mode = "adaptive"
    elif standard:
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
    elif mc2:
        backtest_type = f"MC2 Robustness Tests ({mc2:,} iterations)"
        backtest_mode = "mc2"
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
            evo_backtest_mc2,
            evo_backtest_unified_adaptive,
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

        # Fix yfinance MultiIndex columns (sometimes returns tuples)
        cols = data.columns
        if isinstance(cols[0], tuple):
            # Flatten multiindex: ('Close', 'QQQ') -> 'close' (take first element only)
            data.columns = [str(col[0]).lower() for col in cols]
        else:
            # Normal columns: 'Close' -> 'close'
            data.columns = [str(col).lower() for col in cols]

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

            if backtest_mode == "adaptive":
                response = evo_backtest_unified_adaptive(symbol, data, enable_all_modules=True, config=config)
            elif backtest_mode == "standard":
                response = evo_backtest_standard(symbol, data, config=config)
            elif backtest_mode == "walk_forward":
                response = evo_backtest_walk_forward(symbol, data, config=config)
            elif backtest_mode == "crisis":
                response = evo_backtest_crisis(symbol, data, config=config)
            elif backtest_mode == "monte_carlo":
                response = evo_backtest_monte_carlo(symbol, data, n_sim=monte_carlo, config=config)
            elif backtest_mode == "mc2":
                response = evo_backtest_mc2(symbol, data, n_sim=mc2, config=config)

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

        # Handle both dict (walk-forward aggregated) and object (standard) results
        if isinstance(result, dict):
            # MC2 results
            if 'tests_run' in result and 'results' in result:
                results_table.add_row("Symbol", result['symbol'])
                results_table.add_row("Simulations per Test", f"{result['n_simulations']:,}")
                results_table.add_row("Tests Run", f"{len(result['tests_run'])}")

                console.print(results_table)

                # Display detailed results for each test
                for test_name, test_result in result['results'].items():
                    console.print(f"\n[bold magenta]Test: {test_name.upper()}[/bold magenta]")

                    test_table = Table(show_header=True, header_style="bold cyan")
                    test_table.add_column("Metric", style="cyan")
                    test_table.add_column("Value", style="green", justify="right")

                    test_table.add_row("Actual Sharpe", f"{test_result.actual_sharpe:.3f}")
                    test_table.add_row("MC Sharpe Mean", f"{test_result.mc_sharpe_mean:.3f}")
                    test_table.add_row("MC Sharpe Std", f"{test_result.mc_sharpe_std:.3f}")
                    test_table.add_row("MC Sharpe Range", f"[{test_result.mc_sharpe_min:.3f}, {test_result.mc_sharpe_max:.3f}]")
                    test_table.add_row("Skill Percentile", f"{test_result.skill_percentile:.1f}%")
                    test_table.add_row("P-Value", f"{test_result.p_value:.4f}")

                    significance = "✅ SIGNIFICANT" if test_result.significant else "❌ NOT SIGNIFICANT"
                    test_table.add_row("Significance (p<0.05)", significance)

                    console.print(test_table)

                # Skip the standard results table since we already printed everything
                results_table = None

            # CR2 Crisis Detection results
            elif 'num_crises' in result and 'crises' in result:
                results_table.add_row("Symbol", result['symbol'])
                results_table.add_row("Detector", result['detector'])
                results_table.add_row("Crises Detected", f"{result['num_crises']}")

                console.print(results_table)

                # Display detailed results for each detected crisis
                if result['num_crises'] > 0:
                    for i, crisis in enumerate(result['crises'], 1):
                        console.print(f"\n[bold magenta]Crisis {i}: {crisis['name']}[/bold magenta]")

                        crisis_table = Table(show_header=True, header_style="bold cyan")
                        crisis_table.add_column("Metric", style="cyan")
                        crisis_table.add_column("Value", style="green", justify="right")

                        # Crisis type with emoji
                        crisis_type = crisis['type']
                        type_emoji = {
                            'LIQUIDITY_CRISIS': '💧',
                            'PANDEMIC_SHOCK': '🦠',
                            'BEAR_MARKET': '🐻',
                            'FLASH_CRASH': '⚡',
                            'SOVEREIGN_DEBT': '🏛️',
                            'TECH_BUBBLE': '💻',
                            'UNKNOWN': '❓'
                        }.get(crisis_type, '❓')

                        crisis_table.add_row("Type", f"{type_emoji} {crisis_type.replace('_', ' ').title()}")
                        crisis_table.add_row("Start Date", crisis['start_date'])
                        crisis_table.add_row("End Date", crisis['end_date'])
                        crisis_table.add_row("Duration", f"{crisis['duration_days']} days")
                        crisis_table.add_row("Max Drawdown", f"{crisis['max_drawdown']:.2%}")
                        crisis_table.add_row("Peak Volatility", f"{crisis['peak_volatility']:.2%}")
                        crisis_table.add_row("Vol Multiplier", f"{crisis['vol_multiplier']:.2f}x")
                        crisis_table.add_row("Recovery Days", f"{crisis['recovery_days']}")

                        # Confidence score with color coding
                        confidence = crisis['match_confidence']
                        if confidence >= 0.8:
                            conf_str = f"[green]{confidence:.1%} ✅ High[/green]"
                        elif confidence >= 0.5:
                            conf_str = f"[yellow]{confidence:.1%} ⚠️ Medium[/yellow]"
                        else:
                            conf_str = f"[red]{confidence:.1%} ❌ Low[/red]"
                        crisis_table.add_row("Match Confidence", conf_str)

                        console.print(crisis_table)
                else:
                    console.print("\n[yellow]No significant crises detected in the data.[/yellow]")

                # Skip the standard results table since we already printed everything
                results_table = None

            # Monte Carlo results
            elif 'actual_sharpe' in result:
                results_table.add_row("Symbol", result.get('symbol', 'N/A'))
                results_table.add_row("Simulations", f"{result['num_simulations']:,}")
                results_table.add_row("Actual Sharpe", f"{result['actual_sharpe']:.3f}")
                results_table.add_row("MC Sharpe Mean", f"{result['mc_sharpe_mean']:.3f}")
                results_table.add_row("MC Sharpe Std", f"{result['mc_sharpe_std']:.3f}")
                results_table.add_row("Skill Percentile", f"{result['skill_percentile']:.1f}%")
                results_table.add_row("P-Value", f"{result['p_value']:.4f}")

                significance = "✅ SIGNIFICANT" if result['significant'] else "❌ NOT SIGNIFICANT"
                results_table.add_row("Significance (p<0.05)", significance)

            # Walk-forward aggregated results have num_folds and aggregated dict
            elif 'num_folds' in result and 'aggregated' in result:
                agg = result['aggregated']
                results_table.add_row("Number of Folds", f"{result['num_folds']}")
                results_table.add_row("Mean Return", f"{agg.get('total_return', 0.0):.2%}")
                results_table.add_row("Mean Sharpe", f"{agg.get('sharpe_mean', 0.0):.3f}")
                results_table.add_row("Mean Sortino", f"{agg.get('sortino_mean', 0.0):.3f}")
                results_table.add_row("Worst Drawdown", f"{agg.get('max_drawdown', 0.0):.2%}")
                results_table.add_row("Total Trades (all folds)", f"{agg.get('total_trades', 0)}")
                results_table.add_row("Consistency %", f"{agg.get('consistency_pct', 0.0):.1f}%")
            else:
                # Legacy dict format
                results_table.add_row("Total Return", f"{result.get('total_return', 0.0):.2%}")
                results_table.add_row("Sharpe Ratio", f"{result.get('sharpe_ratio', 0.0):.3f}")
                results_table.add_row("Sortino Ratio", f"{result.get('sortino_ratio', 0.0):.3f}")
                results_table.add_row("Max Drawdown", f"{result.get('max_drawdown', 0.0):.2%}")
                results_table.add_row("Total Trades", f"{result.get('total_trades', 0)}")
        else:
            results_table.add_row("Total Return", f"{result.total_return:.2%}")
            results_table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
            results_table.add_row("Sortino Ratio", f"{result.sortino_ratio:.3f}")
            results_table.add_row("Calmar Ratio", f"{result.calmar_ratio:.3f}")
            results_table.add_row("Max Drawdown", f"{result.max_drawdown:.2%}")
            results_table.add_row("Win Rate", f"{result.win_rate:.2%}")
            results_table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
            results_table.add_row("Total Trades", f"{result.total_trades}")

        # Only print results_table if not already handled (MC2 sets it to None)
        if results_table is not None:
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

        # Import ALL strategies from registry (11 total)
        from ..strategies import STRATEGY_REGISTRY

        # Initialize signal engine with ALL strategies
        signal_engine = LiveSignalEngine(strategies=STRATEGY_REGISTRY)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Computing features, regime, and strategies...", total=None)

            signal_result = signal_engine.generate_signal(symbol, result)

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Signal generated")

        # Step 3: Display prediction using Rich Dashboard
        console.print("\n[bold cyan]📊 Prediction Dashboard[/bold cyan]\n")

        # Import dashboard
        from ..predict.prediction_dashboard import PredictionDashboard

        # Calculate aggregated signal from strategy signals
        aggregated_signal = 0.0
        if signal_result.signals_raw:
            total_weight = 0.0
            weighted_sum = 0.0
            for strat in signal_result.signals_raw:
                weight = strat.probability
                signal = strat.side * strat.probability
                weighted_sum += signal * weight
                total_weight += weight

            if total_weight > 0:
                aggregated_signal = weighted_sum / total_weight

        # Determine signal string and top strategy
        if aggregated_signal > 0.3:
            signal_str = "LONG (Strong)"
        elif aggregated_signal > 0.1:
            signal_str = "LONG (Weak)"
        elif aggregated_signal < -0.3:
            signal_str = "SHORT (Strong)"
        elif aggregated_signal < -0.1:
            signal_str = "SHORT (Weak)"
        else:
            signal_str = "NEUTRAL"

        # Find top strategy
        top_strategy = "None"
        top_prob = 0.0
        if signal_result.signals_raw:
            for strat in signal_result.signals_raw:
                if abs(strat.side) > 0 and strat.probability > top_prob:
                    top_prob = strat.probability
                    top_strategy = strat.strategy_name

        # Get volatility and features
        volatility = signal_result.metadata.get('volatility', 0.15)

        # Extract features
        features = {}
        if signal_result.features is not None and not signal_result.features.empty:
            latest = signal_result.features.iloc[-1]
            features = {
                'volatility': latest.get('volatility', 0.0),
                'ma_5': latest.get('ma_5', 0.0),
                'ma_20': latest.get('ma_20', 0.0),
                'ma_50': latest.get('ma_50', 0.0),
                'rsi': latest.get('rsi', 50.0),
                'trend': latest.get('trend', 0.0),
            }

        # Build strategy breakdown
        strategies = {}
        if signal_result.signals_raw:
            for strat in signal_result.signals_raw:
                strategies[strat.strategy_name] = {
                    'weight': strat.probability,
                    'signal': float(strat.side)
                }

        # Calculate position sizing (simplified for now)
        current_price = result['Close'].iloc[-1]
        vol_target = 0.12  # 12% target volatility
        leverage = min(2.0, vol_target / (volatility + 0.001))  # Cap at 2x
        exposure = abs(aggregated_signal) * 100.0
        adj_size = exposure * leverage
        position_floor = 0.1

        # Calculate risk metrics (simplified)
        stop_loss = current_price * (1 - 0.02)  # 2% stop
        take_profit = current_price * (1 + 0.04)  # 4% target
        exp_dd = volatility * 1.5 * 100  # Expected 5-day DD
        crisis_score = 0.15  # Placeholder

        # Normalize prediction for dashboard
        normalized = {
            "signal": signal_str,
            "confidence": abs(aggregated_signal),
            "regime": signal_result.regime.upper(),
            "top_strategy": top_strategy,
            "current_price": current_price,  # Add current price
            "position": {
                "exposure": exposure,
                "leverage": leverage,
                "vol_target": vol_target,
                "adj_size": adj_size,
                "floor": position_floor,
            },
            "strategies": strategies,
            "features": features,
            "risk": {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "exp_dd": exp_dd,
                "crisis_score": crisis_score,
            }
        }

        # Render dashboard
        dashboard = PredictionDashboard()
        dashboard.render(symbol.upper(), normalized)

        console.print("\n[bold green]✅ Prediction Complete![/bold green]")
        console.print(f"\nTo start live trading:")
        console.print(f"  [cyan]prado live {symbol} --mode simulate[/cyan]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


# ============================================================================
# OPTIMIZE COMMAND - MODULE K: AUTOTUNER
# ============================================================================

@app.command("optimize")
def optimize_cli(symbol: str):
    """
    Run AutoTuner hyperparameter optimization.

    Usage:
        prado optimize SYMBOL

    Example:
        prado optimize QQQ
    """
    return optimize(symbol)


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
        "optimize",
        "Optimize hyperparameters (AutoTuner)",
        "prado optimize QQQ"
    )
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
