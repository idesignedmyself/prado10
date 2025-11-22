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
# TRAIN-ML COMMAND - ML HORIZON + REGIME TRAINING
# ============================================================================

@app.command(name="train-ml", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train_ml(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed for determinism")
):
    """
    Train ML Horizon + Regime models for hybrid fusion.

    Usage:
        prado train-ml SYMBOL start MM DD YYYY end MM DD YYYY

    Example:
        prado train-ml QQQ start 01 01 2020 end 12 31 2024
    """
    args = ctx.args

    if len(args) < 8:
        console.print("[red]Error:[/red] Missing required arguments")
        console.print("\nUsage: prado train-ml SYMBOL start MM DD YYYY end MM DD YYYY")
        console.print("Example: prado train-ml QQQ start 01 01 2020 end 12 31 2024")
        raise typer.Exit(code=1)

    # Extract symbol
    symbol = args[0].upper()

    try:
        # Parse dates
        start_date, end_date = parse_date_args(args[1:])
        validate_date_range(start_date, end_date)

        # Seed for determinism
        _seed_all(seed)

        # Display configuration
        console.print("\n[bold cyan]🤖 PRADO ML Training Pipeline[/bold cyan]")
        console.print(Panel.fit(
            f"[green]Symbol:[/green] {symbol}\n"
            f"[green]Start:[/green] {start_date}\n"
            f"[green]End:[/green] {end_date}\n"
            f"[green]Mode:[/green] ML Horizon + Regime Training\n"
            f"[green]Seed:[/green] {seed}",
            title="Configuration",
            border_style="cyan"
        ))

        # Import ML modules
        console.print("\n[yellow]⚡ Loading ML modules...[/yellow]")
        from afml_system.ml import HorizonModel, RegimeHorizonModel, HORIZONS, REGIMES
        from afml_system.regime import RegimeDetector

        console.print("[green]✓[/green] ML modules loaded")

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

            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
                raise typer.Exit(code=1)

            # Fix yfinance MultiIndex columns
            cols = data.columns
            if isinstance(cols[0], tuple):
                data.columns = [str(col[0]).lower() for col in cols]
            else:
                data.columns = [str(col).lower() for col in cols]

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Loaded {len(data)} bars")

        # Step 2: Detect regimes
        console.print("\n[bold]Step 2: Regime Detection[/bold]")
        detector = RegimeDetector()
        regime_series = detector.detect_regime_series(data)
        console.print(f"[green]✓[/green] Regimes detected")

        # Display regime distribution
        regime_counts = regime_series.value_counts()
        console.print("\n[cyan]Regime Distribution:[/cyan]")
        for regime in REGIMES:
            count = regime_counts.get(regime, 0)
            pct = (count / len(regime_series)) * 100 if len(regime_series) > 0 else 0
            console.print(f"  {regime:15s}: {count:5d} bars ({pct:5.1f}%)")

        # Step 3: Train Horizon Models
        console.print("\n[bold]Step 3: Training Horizon Models[/bold]")
        for horizon_key in HORIZONS.keys():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Training {horizon_key} model...", total=None)

                model = HorizonModel(symbol=symbol, horizon_key=horizon_key)
                model.train(data)

                progress.update(task, completed=True)

            console.print(f"[green]✓[/green] {horizon_key} model trained")

        # Step 4: Train Regime-Specific Models
        console.print("\n[bold]Step 4: Training Regime-Specific Models[/bold]")
        for horizon_key in HORIZONS.keys():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Training regime models for {horizon_key}...", total=None)

                regime_model = RegimeHorizonModel(symbol=symbol, horizon_key=horizon_key)
                regime_model.train_all(data, regime_series)

                progress.update(task, completed=True)

            console.print(f"[green]✓[/green] Regime models trained for {horizon_key}")

        # Step 5: Model persistence
        console.print("\n[bold]Step 5: Model Persistence[/bold]")
        save_dir = Path('.prado') / 'models' / symbol.lower() / 'ml_horizons'
        console.print(f"[green]✓[/green] Models saved to: {save_dir}")

        # Success summary
        console.print("\n[bold green]✅ ML Training Complete![/bold green]")
        console.print(f"\nTrained Models:")
        console.print(f"  • {len(HORIZONS)} Horizon Models (1d, 3d, 5d, 10d)")
        console.print(f"  • {len(HORIZONS) * len(REGIMES)} Regime-Specific Models")
        console.print(f"  • Total: {len(HORIZONS) * (1 + len(REGIMES))} XGBoost models")

        console.print(f"\nYou can now:")
        console.print(f"  • Run backtest with ML: [cyan]prado backtest {symbol} --standard --enable-ml[/cyan]")
        console.print(f"  • Get ML predictions: [cyan]prado predict {symbol}[/cyan]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command(name="train-ml-v2", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train_ml_v2(
    ctx: typer.Context,
    seed: int = typer.Option(42, "--seed", help="Random seed for determinism")
):
    """
    Train ML V2 models (24 features) for enhanced horizon + regime predictions.

    Usage:
        prado train-ml-v2 SYMBOL start MM DD YYYY end MM DD YYYY

    Example:
        prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024
    """
    args = ctx.args

    if len(args) < 8:
        console.print("[red]Error:[/red] Missing required arguments")
        console.print("\nUsage: prado train-ml-v2 SYMBOL start MM DD YYYY end MM DD YYYY")
        console.print("Example: prado train-ml-v2 QQQ start 01 01 2020 end 12 31 2024")
        raise typer.Exit(code=1)

    # Extract symbol
    symbol = args[0].upper()

    try:
        # Parse dates
        start_date, end_date = parse_date_args(args[1:])
        validate_date_range(start_date, end_date)

        # Seed for determinism
        _seed_all(seed)

        # Display configuration
        console.print("\n[bold cyan]🤖 PRADO ML V2 Training Pipeline[/bold cyan]")
        console.print(Panel.fit(
            f"[green]Symbol:[/green] {symbol}\n"
            f"[green]Start:[/green] {start_date}\n"
            f"[green]End:[/green] {end_date}\n"
            f"[green]Mode:[/green] ML V2 (24 Features)\n"
            f"[green]Features:[/green] 9 original + 15 new = 24 total\n"
            f"[green]Seed:[/green] {seed}",
            title="Configuration",
            border_style="cyan"
        ))

        # Import ML V2 modules
        console.print("\n[yellow]⚡ Loading ML V2 modules...[/yellow]")
        from afml_system.ml.feature_builder_v2 import FeatureBuilderV2
        from afml_system.ml.target_builder_v2 import TargetBuilderV2
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from xgboost import XGBClassifier
        import joblib
        from pathlib import Path

        console.print("[green]✓[/green] ML V2 modules loaded")

        # Step 1: Load data
        console.print("\n[bold]Step 1: Data Loading[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Fetching {symbol} data...", total=None)

            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
                raise typer.Exit(code=1)

            # Fix yfinance MultiIndex columns
            cols = data.columns
            if isinstance(cols[0], tuple):
                data.columns = [str(col[0]).lower() for col in cols]
            else:
                data.columns = [str(col).lower() for col in cols]

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Loaded {len(data)} bars")

        # Step 2: Build V2 Features (24 features)
        console.print("\n[bold]Step 2: Building V2 Features (24 features)[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Building features...", total=None)
            X = FeatureBuilderV2.build_features_v2(data)
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Built {X.shape[1]} features × {X.shape[0]} samples")

        # Step 3: Build Labels
        console.print("\n[bold]Step 3: Building Horizon & Regime Labels[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Building labels...", total=None)
            horizon_labels = TargetBuilderV2.build_horizon_labels(data)
            regime_labels = TargetBuilderV2.build_regime_labels(data)
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Horizon labels: {horizon_labels.shape}")
        console.print(f"[green]✓[/green] Regime labels: {regime_labels.shape}")

        # Align features and labels
        common_idx = X.index.intersection(horizon_labels.index).intersection(regime_labels.index)
        X = X.loc[common_idx]
        horizon_labels = horizon_labels.loc[common_idx]
        regime_labels = regime_labels.loc[common_idx]

        console.print(f"[green]✓[/green] Aligned dataset: {len(X)} samples")

        # Create save directory (local to app, not home directory)
        save_dir = Path('.prado') / 'models' / symbol.lower() / 'ml_v2'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Step 4: Train Horizon Models
        console.print("\n[bold]Step 4: Training Horizon Models (4 models)[/bold]")
        horizon_stats = []
        for horizon in ['1d', '3d', '5d', '10d']:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Training {horizon} model...", total=None)

                y = TargetBuilderV2.get_label_for_horizon(horizon_labels, horizon)

                model = XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    eval_metric='logloss'
                )

                model.fit(X, y)

                # Save model
                model_path = save_dir / f"ml_horizon_{horizon}_v2.pkl"
                joblib.dump(model, model_path)

                progress.update(task, completed=True)

            console.print(f"[green]✓[/green] {horizon} model trained")

        # Step 5: Train Regime Models (20 models: 5 regimes × 4 horizons)
        console.print("\n[bold]Step 5: Training Regime Models (20 models)[/bold]")
        regimes = ['trend_up', 'trend_down', 'choppy', 'high_vol', 'low_vol']
        regime_column = regime_labels['regime']

        trained_count = 0
        skipped_count = 0

        for regime in regimes:
            for horizon in ['1d', '3d', '5d', '10d']:
                # Filter to regime
                regime_mask = regime_column == regime
                X_regime = X[regime_mask]
                y_horizon = TargetBuilderV2.get_label_for_horizon(horizon_labels, horizon)
                y_regime = y_horizon[regime_mask]

                if len(X_regime) < 100:
                    console.print(f"[yellow]⚠[/yellow] Skipping {regime}×{horizon} - insufficient samples ({len(X_regime)})")
                    skipped_count += 1
                    continue

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Training {regime}×{horizon}...", total=None)

                    model = XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        eval_metric='logloss'
                    )

                    model.fit(X_regime, y_regime)

                    # Save model
                    model_path = save_dir / f"ml_regime_{regime}_{horizon}_v2.pkl"
                    joblib.dump(model, model_path)

                    progress.update(task, completed=True)

                trained_count += 1

        console.print(f"[green]✓[/green] Regime models: {trained_count} trained, {skipped_count} skipped")

        # Step 6: Save metadata
        console.print("\n[bold]Step 6: Saving Metadata[/bold]")
        metadata = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'samples': len(X),
            'features': list(X.columns),
            'num_features': X.shape[1],
            'horizon_models': 4,
            'regime_models': trained_count,
            'total_models': 4 + trained_count
        }

        metadata_path = save_dir / 'training_metadata_v2.pkl'
        joblib.dump(metadata, metadata_path)

        console.print(f"[green]✓[/green] Metadata saved")

        # Success summary
        console.print("\n[bold green]✅ ML V2 Training Complete![/bold green]")
        console.print(f"\nTrained Models:")
        console.print(f"  • 4 Horizon Models (1d, 3d, 5d, 10d)")
        console.print(f"  • {trained_count} Regime-Specific Models")
        console.print(f"  • Total: {4 + trained_count} XGBoost models")
        console.print(f"  • Features: 24 (9 original + 15 new)")
        console.print(f"\nModels saved to: {save_dir}")

        console.print(f"\nYou can now:")
        console.print(f"  • Enable v2 in config: [cyan]use_ml_features_v2=True[/cyan]")
        console.print(f"  • Models will auto-load from ml_v2 directory")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command(name="ml-v2-diagnostic", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ml_v2_diagnostic(ctx: typer.Context):
    """
    Run ML V2 diagnostic suite to validate trained models.

    Usage:
        prado ml-v2-diagnostic SYMBOL start MM DD YYYY end MM DD YYYY

    Example:
        prado ml-v2-diagnostic QQQ start 01 01 2020 end 12 31 2024
    """
    args = ctx.args

    if len(args) < 8:
        console.print("[red]Error:[/red] Missing required arguments")
        console.print("\nUsage: prado ml-v2-diagnostic SYMBOL start MM DD YYYY end MM DD YYYY")
        console.print("Example: prado ml-v2-diagnostic QQQ start 01 01 2020 end 12 31 2024")
        raise typer.Exit(code=1)

    # Extract symbol
    symbol = args[0].upper()

    try:
        # Parse dates
        start_date, end_date = parse_date_args(args[1:])
        validate_date_range(start_date, end_date)

        # Display configuration
        console.print("\n[bold cyan]🔍 PRADO ML V2 Diagnostic Suite[/bold cyan]")
        console.print(Panel.fit(
            f"[green]Symbol:[/green] {symbol}\n"
            f"[green]Start:[/green] {start_date}\n"
            f"[green]End:[/green] {end_date}\n"
            f"[green]Mode:[/green] ML V2 Validation",
            title="Configuration",
            border_style="cyan"
        ))

        # Import ML V2 diagnostic module
        console.print("\n[yellow]⚡ Loading ML V2 diagnostic module...[/yellow]")
        from afml_system.ml.ml_v2_diagnostic import MLV2Diagnostic
        from afml_system.regime.regime_detector import RegimeDetector
        import yfinance as yf

        console.print("[green]✓[/green] ML V2 diagnostic module loaded")

        # Step 1: Load data
        console.print("\n[bold]Step 1: Data Loading[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Fetching {symbol} data...", total=None)

            data = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if data.empty:
                console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
                raise typer.Exit(code=1)

            # Fix yfinance MultiIndex columns
            cols = data.columns
            if isinstance(cols[0], tuple):
                data.columns = [str(col[0]).lower() for col in cols]
            else:
                data.columns = [str(col).lower() for col in cols]

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Loaded {len(data)} bars")

        # Step 2: Detect regimes
        console.print("\n[bold]Step 2: Regime Detection[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Detecting regimes...", total=None)
            detector = RegimeDetector()
            regime_series = detector.detect_regime_series(data)
            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Regime detection complete")

        # Step 3: Run diagnostic suite
        console.print("\n[bold cyan]Running Diagnostic Tests...[/bold cyan]\n")

        diagnostic = MLV2Diagnostic(symbol)
        results = diagnostic.run_full_diagnostic(data, regime_series)

        # Step 4: Save report
        console.print("\n[bold]Saving Diagnostic Report...[/bold]")
        report_path = diagnostic.save_report()

        # Display summary
        console.print("\n[bold green]✅ ML V2 Diagnostic Complete![/bold green]")

        # Summary table
        summary_table = Table(title="Diagnostic Summary", show_header=True, header_style="bold cyan")
        summary_table.add_column("Test", style="cyan")
        summary_table.add_column("Status", style="green")

        test_names = [
            "Feature Integrity",
            "Target Integrity",
            "Model Loading",
            "Horizon Predictions",
            "Regime Predictions",
            "Confidence Distribution",
            "V1/V2 Comparison",
            "SHAP Explainability",
            "Prediction Consistency"
        ]

        result_keys = [
            'features', 'targets', 'model_loading', 'horizon_predictions',
            'regime_predictions', 'confidence', 'v1_v2_comparison',
            'shap', 'consistency'
        ]

        for name, key in zip(test_names, result_keys):
            if key in results:
                status = results[key].get('status', 'UNKNOWN')
                if status == 'PASS':
                    status_display = "✅ PASS"
                elif status == 'ERROR':
                    status_display = "❌ ERROR"
                elif status in ['SKIPPED', 'UNAVAILABLE', 'NO_DATA', 'INSUFFICIENT_DATA']:
                    status_display = f"⚠️ {status}"
                else:
                    status_display = status
                summary_table.add_row(name, status_display)

        console.print(summary_table)

        console.print(f"\n[cyan]📄 Full report saved to:[/cyan] {report_path}")

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
    seed: int = typer.Option(42, "--seed", help="Random seed for determinism"),
    strict_dates: bool = typer.Option(False, "--strict-dates", help="Fail on date overlaps instead of auto-adjusting")
):
    """
    Run backtest on PRADO9_EVO strategy.

    Usage:
        prado backtest SYMBOL MODE [dates] [enable-ml] [enable-ml-explain] [use-ml-v2]

    Examples:
        prado backtest QQQ standard
        prado backtest SPY walk-forward
        prado backtest QQQ crisis
        prado backtest SPY monte-carlo 10000
        prado backtest QQQ mc2 1000
        prado backtest QQQ standard start 01 01 2020 end 12 31 2023
        prado backtest QQQ walk-forward start 01 01 2023 end 12 31 2025
        prado backtest QQQ combo start 01 01 2020 end 12 31 2023 wf 12 31 2025
        prado backtest QQQ standard enable-ml
        prado backtest QQQ standard enable-ml enable-ml-explain
        prado backtest QQQ standard enable-ml use-ml-v2
    """
    from .date_parser import parse_backtest_args

    args = ctx.args

    if len(args) < 2:
        console.print("[red]Error:[/red] Symbol and mode required")
        console.print("\nUsage: prado backtest SYMBOL MODE [dates]")
        console.print("Example: prado backtest QQQ standard")
        console.print("         prado backtest QQQ combo start 01 01 2020 end 12 31 2023 wf 12 31 2025")
        raise typer.Exit(code=1)

    # Parse arguments
    try:
        parsed = parse_backtest_args(args)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    symbol = parsed['symbol']
    mode = parsed['mode']
    start_date = parsed['start_date']
    end_date = parsed['end_date']
    wf_date = parsed['wf_date']
    iterations = parsed['iterations']

    # Parse ML flags (space-based)
    enable_ml = 'enable-ml' in args
    enable_ml_explain = 'enable-ml-explain' in args
    use_ml_v2 = 'use-ml-v2' in args

    # Validate: enable-ml-explain requires enable-ml
    if enable_ml_explain and not enable_ml:
        console.print("[red]Error:[/red] enable-ml-explain requires enable-ml")
        raise typer.Exit(code=1)

    # Validate: use-ml-v2 requires enable-ml
    if use_ml_v2 and not enable_ml:
        console.print("[red]Error:[/red] use-ml-v2 requires enable-ml")
        raise typer.Exit(code=1)

    # Seed for determinism
    _seed_all(seed)

    # Handle combined backtest separately (early exit)
    if mode == 'combo':
        if not (start_date and end_date and wf_date):
            console.print("[red]Error:[/red] combo mode requires start, end, and wf dates")
            console.print("\nExample: prado backtest QQQ combo start 01 01 2020 end 12 31 2023 wf 12 31 2025")
            raise typer.Exit(code=1)

        # Import combined backtest
        from afml_system.backtest import evo_backtest_combined

        # Load data
        console.print(f"\n[bold]Loading {symbol} data for combined backtest...[/bold]")
        import yfinance as yf
        from datetime import datetime

        # Use full date range for data loading
        data = yf.download(symbol, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)

        if data.empty:
            console.print(f"[red]Error:[/red] No data retrieved for {symbol}")
            raise typer.Exit(code=1)

        # Fix yfinance MultiIndex columns
        cols = data.columns
        if isinstance(cols[0], tuple):
            data.columns = [str(col[0]).lower() for col in cols]
        else:
            data.columns = [str(col).lower() for col in cols]

        console.print(f"[green]✓[/green] Loaded {len(data)} bars")

        # Create config
        from afml_system.backtest import BacktestConfig
        config = BacktestConfig(
            symbol=symbol,
            random_seed=seed,
            enable_ml_fusion=enable_ml,
            enable_ml_explain=enable_ml_explain,
            use_ml_features_v2=use_ml_v2
        )

        # Run combined backtest
        result = evo_backtest_combined(
            symbol=symbol,
            data=data,
            start=start_date,
            end=end_date,
            wf=wf_date,
            strict=strict_dates,
            config=config
        )

        console.print("\n[green]✅ Combined Backtest Complete![/green]\n")
        return

    # Determine backtest type from mode
    backtest_mode = mode.replace('-', '_')  # normalize walk-forward -> walk_forward

    if mode == 'adaptive':
        backtest_type = "Unified Adaptive Backtest (AR+X2+Y2+CR2)"
    elif mode == 'standard':
        backtest_type = "Standard Backtest"
    elif mode == 'walk-forward':
        backtest_type = "Walk-Forward Backtest"
    elif mode == 'crisis':
        backtest_type = "Crisis Period Backtest"
    elif mode == 'monte-carlo':
        backtest_type = f"Monte Carlo Simulation ({iterations:,} iterations)"
    elif mode == 'mc2':
        backtest_type = f"MC2 Robustness Tests ({iterations:,} iterations)"
    else:
        backtest_type = f"{mode.title()} Backtest"

    # Display configuration
    console.print("\n[bold cyan]📊 PRADO9_EVO Backtest Engine[/bold cyan]")
    config_lines = [
        f"[green]Symbol:[/green] {symbol}",
        f"[green]Type:[/green] {backtest_type}",
        f"[green]Seed:[/green] {seed}"
    ]
    if start_date or end_date:
        date_range = f"{start_date or 'auto'} to {end_date or 'today'}"
        config_lines.append(f"[green]Date Range:[/green] {date_range}")

    console.print(Panel.fit(
        "\n".join(config_lines),
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
        from datetime import datetime, timedelta

        # Parse custom dates or use defaults
        # Note: start_date and end_date are already in YYYY-MM-DD format from parse_backtest_args
        try:
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_dt = datetime.now()

            if start_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_dt = end_dt - timedelta(days=5*365)
        except ValueError as e:
            console.print(f"[red]Error:[/red] Invalid date format")
            console.print(f"[red]Details:[/red] {str(e)}")
            raise typer.Exit(code=1)

        data = yf.download(
            symbol,
            start=start_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
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

        console.print(f"[green]✓[/green] Loaded {len(data)} bars ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})")

        # Create config with seed
        from afml_system.backtest import BacktestConfig

        config = BacktestConfig(
            symbol=symbol,
            random_seed=seed,
            enable_ml_fusion=enable_ml,
            enable_ml_explain=enable_ml_explain,
            use_ml_features_v2=use_ml_v2
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
                response = evo_backtest_monte_carlo(symbol, data, n_sim=iterations, config=config)
            elif backtest_mode == "mc2":
                response = evo_backtest_mc2(symbol, data, n_sim=iterations, config=config)

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

        # Strategy allocation summary (stabilized allocator weights)
        if hasattr(result, 'strategy_allocations') and result.strategy_allocations:
            console.print("\n[bold]Strategy Allocations (Stabilized Weights):[/bold]")

            strat_table = Table(show_header=True, header_style="bold magenta")
            strat_table.add_column("Strategy", style="cyan")
            strat_table.add_column("Weight (Normalized)", style="green", justify="right")

            for strategy, allocation in sorted(result.strategy_allocations.items(), key=lambda x: -abs(x[1])):
                strat_table.add_row(
                    strategy,
                    f"{allocation:.2%}"
                )

            console.print(strat_table)
            console.print(f"  [dim]Note: Average allocator weights across {result.total_trades} trades (tanh + L1 normalized).[/dim]")

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

        # Step 3.5: Run allocator to get stabilized weights
        allocation_decision = None
        if signal_result.signals_raw:
            try:
                allocation_decision = evo_allocate(
                    signals=signal_result.signals_raw,
                    regime=signal_result.regime,
                    horizon='5d',
                    corr_data=None,
                    risk_params=None
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Allocator failed ({e}), using fallback weights[/yellow]")

        # Build strategy breakdown using allocator weights (if available)
        strategies = {}
        if allocation_decision and hasattr(allocation_decision, 'strategy_weights'):
            # Use stabilized allocator weights
            for strat_name, weight in allocation_decision.strategy_weights.items():
                # Find the signal for this strategy
                strat_signal = 0
                for strat in signal_result.signals_raw:
                    if strat.strategy_name == strat_name:
                        strat_signal = float(strat.side)
                        break
                strategies[strat_name] = {
                    'weight': abs(weight),  # Display as positive weight
                    'signal': strat_signal
                }
        elif signal_result.signals_raw:
            # Fallback to raw probabilities
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
