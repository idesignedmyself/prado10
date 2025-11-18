"""
PRADO9_EVO Optimize Command

Runs AutoTuner for hyperparameter optimization.
"""

import typer
import yfinance as yf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


def optimize(symbol: str):
    """
    Run AutoTuner hyperparameter optimization.

    Args:
        symbol: Trading symbol to optimize
    """
    from ..autotune.auto_tuner import AutoTuner

    symbol = symbol.upper()

    console.print(f"\n[bold cyan]🔧 PRADO9_EVO AutoTuner[/bold cyan]")
    console.print(Panel.fit(
        f"[green]Symbol:[/green] {symbol}\n"
        f"[green]Mode:[/green] Hyperparameter Optimization",
        title="Configuration",
        border_style="cyan"
    ))

    # Load data
    console.print("\n[bold]Step 1: Loading Data[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Fetching {symbol} data...", total=None)

        # Default to 5 years of data for optimization
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)

        df = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )

        if df.empty:
            console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
            raise typer.Exit(code=1)

        # Fix yfinance MultiIndex columns
        cols = df.columns
        if isinstance(cols[0], tuple):
            # Flatten multiindex: ('Close', 'QQQ') -> 'close' (take first element only)
            df.columns = [str(col[0]).lower() for col in cols]
        else:
            df.columns = [str(col).lower() for col in cols]

        progress.update(task, completed=True)

    console.print(f"[green]✓[/green] Loaded {len(df)} bars")

    # Run AutoTuner
    console.print("\n[bold]Step 2: Running Hyperparameter Sweep[/bold]")
    console.print("[yellow]This will test multiple parameter combinations via CPCV...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Optimizing parameters...", total=None)

        tuner = AutoTuner(symbol)
        result = tuner.run(df)

        progress.update(task, completed=True)

    # Check for errors
    if result.get("status") == "error":
        console.print(f"[red]✗ Optimization failed:[/red] {result.get('message', 'Unknown error')}")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Optimization complete")

    # Display results
    console.print("\n[bold cyan]📊 Optimization Results[/bold cyan]")

    console.print(f"\n[bold]Best Parameters:[/bold]")
    for key, value in result["best_params"].items():
        console.print(f"  • {key}: [cyan]{value}[/cyan]")

    console.print(f"\n[bold]Best Score:[/bold] [green]{result['best_score']:.4f}[/green]")

    config_path = f"~/.prado/configs/{symbol}.yaml"
    console.print(f"\n[green]✓[/green] Config saved to: [cyan]{config_path}[/cyan]")

    console.print("\n[bold green]✅ Optimization Complete![/bold green]")
    console.print(f"\nYou can now train with optimized parameters:")
    console.print(f"  [cyan]prado train {symbol} start 01 01 2020 end 12 31 2024[/cyan]")
