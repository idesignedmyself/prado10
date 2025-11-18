"""
PRADO9_EVO Command-Line Interface

Provides commands for training, backtesting, and live trading with the PRADO9_EVO system.
"""

import sys
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
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


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(ctx: typer.Context):
    """
    Train PRADO9_EVO models on historical data.

    Usage:
        prado train SYMBOL start MM DD YYYY end MM DD YYYY

    Example:
        prado train QQQ start 01 01 2020 end 12 31 2023
    """
    args = ctx.args

    if len(args) < 8:
        console.print("[red]Error:[/red] Missing required arguments")
        console.print("\nUsage: prado train SYMBOL start MM DD YYYY end MM DD YYYY")
        console.print("Example: prado train QQQ start 01 01 2020 end 12 31 2023")
        raise typer.Exit(code=1)

    # Extract symbol (first argument)
    symbol = args[0].upper()

    try:
        # Parse dates
        start_date, end_date = parse_date_args(args[1:])
        validate_date_range(start_date, end_date)

        # Display training configuration
        console.print("\n[bold cyan]🔬 PRADO9_EVO Training Pipeline[/bold cyan]")
        console.print(Panel.fit(
            f"[green]Symbol:[/green] {symbol}\n"
            f"[green]Start:[/green] {start_date}\n"
            f"[green]End:[/green] {end_date}\n"
            f"[green]Mode:[/green] Full Training",
            title="Configuration",
            border_style="cyan"
        ))

        # Training pipeline
        console.print("\n[yellow]⚠️  Training pipeline implementation in progress[/yellow]")
        console.print("\nThis will integrate:")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Module", style="cyan")
        table.add_column("Component", style="green")
        table.add_column("Status", style="yellow")

        table.add_row("A", "Bandit Brain (Strategy Selection)", "⏳ Pending")
        table.add_row("B", "Feature Engineering", "⏳ Pending")
        table.add_row("C", "Labeling System", "⏳ Pending")
        table.add_row("D", "Meta-Learner", "⏳ Pending")
        table.add_row("E", "Sample Weights", "⏳ Pending")
        table.add_row("F", "Correlation Engine", "⏳ Pending")

        console.print(table)

        console.print("\n[bold green]💡 Workaround:[/bold green]")
        console.print(f"For now, use live trading in simulate mode:")
        console.print(f"  [cyan]prado live {symbol} --mode simulate[/cyan]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def backtest(
    ctx: typer.Context,
    standard: bool = typer.Option(False, "--standard", help="Run standard backtest"),
    walk_forward: bool = typer.Option(False, "--walk-forward", help="Run walk-forward backtest"),
    crisis: bool = typer.Option(False, "--crisis", help="Run crisis period backtest"),
    monte_carlo: Optional[int] = typer.Option(None, "--monte-carlo", help="Run Monte Carlo simulation (specify iterations)")
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

    # Determine backtest type
    if standard:
        backtest_type = "Standard Backtest"
    elif walk_forward:
        backtest_type = "Walk-Forward Backtest"
    elif crisis:
        backtest_type = "Crisis Period Backtest"
    elif monte_carlo:
        backtest_type = f"Monte Carlo Simulation ({monte_carlo:,} iterations)"
    else:
        backtest_type = "Standard Backtest (default)"

    # Display configuration
    console.print("\n[bold cyan]📊 PRADO9_EVO Backtest Engine[/bold cyan]")
    console.print(Panel.fit(
        f"[green]Symbol:[/green] {symbol}\n"
        f"[green]Type:[/green] {backtest_type}",
        title="Configuration",
        border_style="cyan"
    ))

    # Backtest pipeline
    console.print("\n[yellow]⚠️  Backtest pipeline implementation in progress[/yellow]")
    console.print("\nThis will integrate:")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Module", style="cyan")
    table.add_column("Component", style="green")
    table.add_column("Status", style="yellow")

    table.add_row("G", "Evolutionary Allocator", "⏳ Pending")
    table.add_row("H", "Execution Engine", "✓ Ready")
    table.add_row("I", "Performance Analytics", "⏳ Pending")

    console.print(table)

    console.print("\n[bold green]💡 Workaround:[/bold green]")
    console.print(f"For now, use live trading in simulate mode:")
    console.print(f"  [cyan]prado live {symbol} --mode simulate[/cyan]")


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
            enable_console=True
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
        console.print("\n[yellow]💡 Make sure you're running from the prado_evo directory[/yellow]")
        raise typer.Exit(code=1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def predict(ctx: typer.Context):
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

    console.print("\n[bold cyan]🔮 PRADO9_EVO Prediction Engine[/bold cyan]")
    console.print(Panel.fit(
        f"[green]Symbol:[/green] {symbol}\n"
        f"[green]Mode:[/green] Real-time Prediction",
        title="Configuration",
        border_style="cyan"
    ))

    console.print("\n[yellow]⚠️  Prediction engine implementation in progress[/yellow]")
    console.print("\nThis will:")
    console.print("  • Load trained models")
    console.print("  • Fetch latest market data")
    console.print("  • Generate regime classification")
    console.print("  • Produce strategy signals")
    console.print("  • Calculate position recommendations")

    console.print("\n[bold green]💡 Workaround:[/bold green]")
    console.print(f"For now, use live trading in simulate mode to see predictions:")
    console.print(f"  [cyan]prado live {symbol} --mode simulate[/cyan]")


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
        "prado train QQQ start 01 01 2020 end 12 31 2023"
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


# Main entry point
def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
