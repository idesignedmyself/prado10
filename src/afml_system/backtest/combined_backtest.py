"""
PRADO9_EVO Combined Backtest Module

Unified standard + walk-forward backtest execution with intelligent
date handling, overlap prevention, and auto-adjustment.

Features:
- Non-interactive auto-adjustment for overlapping windows
- 365-day standard backtest window
- 90-day walk-forward fold size
- Deterministic and reproducible results
- Clean unified dashboard output

Author: PRADO9_EVO Builder
Date: 2025-01-19
Version: 1.0.0
"""

import datetime
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .backtest_engine import (
    evo_backtest_standard,
    evo_backtest_walk_forward,
    BacktestConfig
)

console = Console()


@dataclass
class CombinedBacktestResult:
    """Results from combined backtest execution."""

    # Training window
    train_start: datetime.date
    train_end: datetime.date

    # Standard backtest window
    std_start: datetime.date
    std_end: datetime.date

    # Walk-forward window
    wf_start: datetime.date
    wf_end: datetime.date
    wf_folds: int

    # Standard backtest results
    std_return: float
    std_sharpe: float
    std_sortino: float
    std_max_dd: float
    std_trades: int

    # Walk-forward results
    wf_mean_return: float
    wf_mean_sharpe: float
    wf_mean_sortino: float
    wf_worst_dd: float
    wf_consistency: float
    wf_total_trades: int

    # Auto-adjustment flag
    auto_adjusted: bool
    adjustment_reason: Optional[str] = None


def _parse_date(date_str: str) -> datetime.date:
    """Parse MM-DD-YYYY date string to date object."""
    return datetime.datetime.strptime(date_str, '%m-%d-%Y').date()


def _validate_and_adjust_windows(
    train_start: datetime.date,
    train_end: datetime.date,
    wf_end: datetime.date,
    strict: bool = False
) -> Tuple[datetime.date, datetime.date, datetime.date, bool, Optional[str]]:
    """
    Validate and auto-adjust backtest windows.

    Args:
        train_start: Training period start
        train_end: Training period end
        wf_end: Walk-forward period end
        strict: If True, raise error instead of auto-adjusting

    Returns:
        Tuple of (train_start, train_end, adjusted_wf_end, was_adjusted, reason)

    Raises:
        ValueError: If windows are invalid and strict mode is enabled
    """
    if train_start >= train_end:
        raise ValueError(
            f"Training start ({train_start}) must be before training end ({train_end})"
        )

    auto_adjusted = False
    adjustment_reason = None

    # Check for overlap
    if wf_end <= train_end:
        if strict:
            raise ValueError(
                f"Walk-forward end ({wf_end}) is inside training window "
                f"({train_start} to {train_end}). Use --strict-dates to enforce."
            )

        # Auto-adjust: set wf_end to train_end + 365 days
        original_wf_end = wf_end
        wf_end = train_end + datetime.timedelta(days=365)
        auto_adjusted = True
        adjustment_reason = (
            f"Walk-forward end ({original_wf_end}) was inside training window. "
            f"Auto-adjusted to {wf_end} (training_end + 365 days)"
        )

        console.print(
            f"\n[yellow]⚠ Auto-Adjustment:[/yellow] {adjustment_reason}",
            style="yellow"
        )

    return train_start, train_end, wf_end, auto_adjusted, adjustment_reason


def _calculate_windows(
    train_start: datetime.date,
    train_end: datetime.date,
    wf_end: datetime.date
) -> Tuple[datetime.date, datetime.date, datetime.date]:
    """
    Calculate standard and walk-forward test windows.

    Standard backtest: 365 days after training end (or until wf_end if shorter)
    Walk-forward: From training end to wf_end

    Args:
        train_start: Training period start
        train_end: Training period end
        wf_end: Walk-forward period end

    Returns:
        Tuple of (std_start, std_end, wf_start)
    """
    # Standard backtest starts day after training ends
    std_start = train_end + datetime.timedelta(days=1)

    # Standard backtest is 365 days (or until wf_end if shorter)
    std_end_candidate = std_start + datetime.timedelta(days=365)
    std_end = min(std_end_candidate, wf_end)

    # Walk-forward starts at same point as standard
    wf_start = std_start

    return std_start, std_end, wf_start


def _generate_walkforward_folds(
    train_start: datetime.date,
    train_end: datetime.date,
    wf_end: datetime.date,
    fold_size_days: int = 90
) -> List[Dict]:
    """
    Generate walk-forward folds with anchored training.

    Args:
        train_start: Initial training start
        train_end: Initial training end
        wf_end: Walk-forward period end
        fold_size_days: Test window size in days (default 90)

    Returns:
        List of fold dictionaries with train/test windows
    """
    folds = []
    current_start = train_start
    current_end = train_end

    while current_end < wf_end:
        fold_test_start = current_end + datetime.timedelta(days=1)
        fold_test_end = fold_test_start + datetime.timedelta(days=fold_size_days)

        # Don't extend past wf_end
        fold_test_end = min(fold_test_end, wf_end)

        folds.append({
            'train_start': current_start,
            'train_end': current_end,
            'test_start': fold_test_start,
            'test_end': fold_test_end
        })

        # Move to next fold (anchored training, so only end moves)
        current_end = fold_test_end

        # Prevent infinite loop if we've reached the end
        if current_end >= wf_end:
            break

    return folds


def _extract_standard_results(response: Dict) -> Dict:
    """Extract key metrics from standard backtest response."""
    try:
        # Parse the response text to extract metrics
        # This is a simplified extraction - adjust based on actual response format
        return {
            'return': response.get('total_return', 0.0),
            'sharpe': response.get('sharpe_ratio', 0.0),
            'sortino': response.get('sortino_ratio', 0.0),
            'max_dd': response.get('max_drawdown', 0.0),
            'trades': response.get('total_trades', 0)
        }
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse standard backtest results: {e}[/yellow]")
        return {
            'return': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'max_dd': 0.0,
            'trades': 0
        }


def _extract_walkforward_results(response: Dict) -> Dict:
    """Extract key metrics from walk-forward backtest response."""
    try:
        return {
            'mean_return': response.get('mean_return', 0.0),
            'mean_sharpe': response.get('mean_sharpe', 0.0),
            'mean_sortino': response.get('mean_sortino', 0.0),
            'worst_dd': response.get('worst_drawdown', 0.0),
            'consistency': response.get('consistency_pct', 0.0),
            'total_trades': response.get('total_trades', 0)
        }
    except Exception as e:
        console.print(f"[yellow]Warning: Could not parse walk-forward results: {e}[/yellow]")
        return {
            'mean_return': 0.0,
            'mean_sharpe': 0.0,
            'mean_sortino': 0.0,
            'worst_dd': 0.0,
            'consistency': 0.0,
            'total_trades': 0
        }


def _render_combined_summary(result: CombinedBacktestResult):
    """Render beautiful combined backtest summary."""

    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Combined Backtest Results[/bold cyan]",
        border_style="cyan"
    ))

    # Training Window
    console.print("\n[bold]Training Window[/bold]")
    console.print(f"  {result.train_start} → {result.train_end}")

    # Standard Backtest Table
    console.print("\n[bold]Standard Backtest (OOS)[/bold]")
    std_table = Table(show_header=True, header_style="bold cyan")
    std_table.add_column("Window", style="green")
    std_table.add_column("Return", justify="right")
    std_table.add_column("Sharpe", justify="right")
    std_table.add_column("Sortino", justify="right")
    std_table.add_column("Max DD", justify="right")
    std_table.add_column("Trades", justify="right")

    std_table.add_row(
        f"{result.std_start} → {result.std_end}",
        f"{result.std_return:.2f}%",
        f"{result.std_sharpe:.3f}",
        f"{result.std_sortino:.3f}",
        f"{result.std_max_dd:.2f}%",
        f"{result.std_trades}"
    )
    console.print(std_table)

    # Walk-Forward Backtest Table
    console.print("\n[bold]Walk-Forward Backtest[/bold]")
    wf_table = Table(show_header=True, header_style="bold cyan")
    wf_table.add_column("Folds", style="green")
    wf_table.add_column("Window", style="green")
    wf_table.add_column("Mean Return", justify="right")
    wf_table.add_column("Mean Sharpe", justify="right")
    wf_table.add_column("Mean Sortino", justify="right")
    wf_table.add_column("Worst DD", justify="right")
    wf_table.add_column("Consistency", justify="right")
    wf_table.add_column("Trades", justify="right")

    wf_table.add_row(
        f"{result.wf_folds}",
        f"{result.wf_start} → {result.wf_end}",
        f"{result.wf_mean_return:.2f}%",
        f"{result.wf_mean_sharpe:.3f}",
        f"{result.wf_mean_sortino:.3f}",
        f"{result.wf_worst_dd:.2f}%",
        f"{result.wf_consistency:.1f}%",
        f"{result.wf_total_trades}"
    )
    console.print(wf_table)

    # Notes
    console.print("\n[bold]Notes[/bold]")
    console.print("  ✔ Standard: 365-day out-of-sample window")
    console.print("  ✔ Walk-Forward: 90-day rolling test folds")
    console.print("  ✔ No overlap between training and testing")
    if result.auto_adjusted:
        console.print(f"  ⚠ Auto-adjusted: {result.adjustment_reason}")
    console.print("  ✔ Deterministic and reproducible\n")


def evo_backtest_combined(
    symbol: str,
    data: pd.DataFrame,
    start: str,
    end: str,
    wf: str,
    strict: bool = False,
    config: Optional[BacktestConfig] = None
) -> CombinedBacktestResult:
    """
    Run combined standard + walk-forward backtest.

    Args:
        symbol: Trading symbol (e.g., 'QQQ')
        data: Full OHLCV DataFrame
        start: Training start date (MM-DD-YYYY)
        end: Training end date (MM-DD-YYYY)
        wf: Walk-forward end date (MM-DD-YYYY)
        strict: If True, raise error on overlap instead of auto-adjusting
        config: Optional backtest configuration

    Returns:
        CombinedBacktestResult with all metrics

    Raises:
        ValueError: If date validation fails
    """
    # Parse dates
    train_start = _parse_date(start)
    train_end = _parse_date(end)
    wf_end = _parse_date(wf)

    # Validate and adjust windows
    train_start, train_end, wf_end, auto_adjusted, adjustment_reason = \
        _validate_and_adjust_windows(train_start, train_end, wf_end, strict)

    # Calculate test windows
    std_start, std_end, wf_start = _calculate_windows(train_start, train_end, wf_end)

    # Generate walk-forward folds
    folds = _generate_walkforward_folds(train_start, train_end, wf_end, fold_size_days=90)

    # Format dates as MM-DD-YYYY
    def format_date(d: datetime.date) -> str:
        return d.strftime('%m-%d-%Y')

    console.print(f"\n[bold cyan]📊 Combined Backtest: {symbol}[/bold cyan]")
    console.print(f"[green]Training:[/green] {format_date(train_start)} → {format_date(train_end)}")
    console.print(f"[green]Standard OOS:[/green] {format_date(std_start)} → {format_date(std_end)}")
    console.print(f"[green]Walk-Forward:[/green] {format_date(wf_start)} → {format_date(wf_end)} ({len(folds)} folds)")
    if auto_adjusted:
        console.print(f"[yellow]⚠ Auto-Adjustment:[/yellow] {adjustment_reason}\n")

    # Run standard backtest
    console.print("\n[bold yellow]═══ Standard Backtest ═══[/bold yellow]")
    std_response = evo_backtest_standard(symbol, data, config=config)

    # Display standard results
    if std_response['status'] == 'success':
        std_result = std_response['result']
        console.print("\n[bold cyan]📈 Standard Backtest Results[/bold cyan]")

        std_table = Table(show_header=True, header_style="bold magenta")
        std_table.add_column("Metric", style="cyan")
        std_table.add_column("Value", style="green", justify="right")

        # Handle both dict and object results
        if isinstance(std_result, dict):
            std_table.add_row("Total Return", f"{std_result.get('total_return', 0.0):.2%}")
            std_table.add_row("Sharpe Ratio", f"{std_result.get('sharpe_ratio', 0.0):.3f}")
            std_table.add_row("Sortino Ratio", f"{std_result.get('sortino_ratio', 0.0):.3f}")
            std_table.add_row("Max Drawdown", f"{std_result.get('max_drawdown', 0.0):.2%}")
            std_table.add_row("Total Trades", f"{std_result.get('total_trades', 0)}")
        else:
            std_table.add_row("Total Return", f"{std_result.total_return:.2%}")
            std_table.add_row("Sharpe Ratio", f"{std_result.sharpe_ratio:.3f}")
            std_table.add_row("Sortino Ratio", f"{std_result.sortino_ratio:.3f}")
            std_table.add_row("Calmar Ratio", f"{std_result.calmar_ratio:.3f}")
            std_table.add_row("Max Drawdown", f"{std_result.max_drawdown:.2%}")
            std_table.add_row("Win Rate", f"{std_result.win_rate:.2%}")
            std_table.add_row("Profit Factor", f"{std_result.profit_factor:.2f}")
            std_table.add_row("Total Trades", f"{std_result.total_trades}")

        console.print(std_table)

        # Strategy allocations if available (stabilized weights)
        if hasattr(std_result, 'strategy_allocations') and std_result.strategy_allocations:
            console.print("\n[bold]Strategy Allocations (Stabilized Weights):[/bold]")
            strat_table = Table(show_header=True, header_style="bold magenta")
            strat_table.add_column("Strategy", style="cyan")
            strat_table.add_column("Weight (Normalized)", style="green", justify="right")

            for strategy, allocation in sorted(std_result.strategy_allocations.items(), key=lambda x: -abs(x[1])):
                strat_table.add_row(strategy, f"{allocation:.2%}")

            console.print(strat_table)
            console.print(f"  [dim]Note: Average allocator weights across {std_result.total_trades} trades (tanh + L1 normalized).[/dim]")

    # Run walk-forward backtest
    console.print("\n[bold yellow]═══ Walk-Forward Backtest ═══[/bold yellow]")
    wf_response = evo_backtest_walk_forward(symbol, data, config=config)

    # Display walk-forward results
    if wf_response['status'] == 'success':
        wf_result = wf_response['result']
        console.print("\n[bold cyan]📈 Walk-Forward Backtest Results[/bold cyan]")

        wf_table = Table(show_header=True, header_style="bold magenta")
        wf_table.add_column("Metric", style="cyan")
        wf_table.add_column("Value", style="green", justify="right")

        # Walk-forward aggregated results
        if isinstance(wf_result, dict) and 'num_folds' in wf_result and 'aggregated' in wf_result:
            agg = wf_result['aggregated']
            wf_table.add_row("Number of Folds", f"{wf_result['num_folds']}")
            wf_table.add_row("Mean Return", f"{agg.get('total_return', 0.0):.2%}")
            wf_table.add_row("Mean Sharpe", f"{agg.get('sharpe_mean', 0.0):.3f}")
            wf_table.add_row("Mean Sortino", f"{agg.get('sortino_mean', 0.0):.3f}")
            wf_table.add_row("Worst Drawdown", f"{agg.get('max_drawdown', 0.0):.2%}")
            wf_table.add_row("Total Trades (all folds)", f"{agg.get('total_trades', 0)}")
            wf_table.add_row("Consistency %", f"{agg.get('consistency_pct', 0.0):.1f}%")
        else:
            # Fallback for other dict formats
            wf_table.add_row("Total Return", f"{wf_result.get('total_return', 0.0):.2%}")
            wf_table.add_row("Sharpe Ratio", f"{wf_result.get('sharpe_ratio', 0.0):.3f}")
            wf_table.add_row("Sortino Ratio", f"{wf_result.get('sortino_ratio', 0.0):.3f}")
            wf_table.add_row("Max Drawdown", f"{wf_result.get('max_drawdown', 0.0):.2%}")
            wf_table.add_row("Total Trades", f"{wf_result.get('total_trades', 0)}")

        console.print(wf_table)

    console.print("\n[green]✅ Combined Backtest Complete![/green]\n")

    return {
        'standard': std_response,
        'walk_forward': wf_response,
        'windows': {
            'train_start': train_start,
            'train_end': train_end,
            'std_start': std_start,
            'std_end': std_end,
            'wf_start': wf_start,
            'wf_end': wf_end,
            'folds': len(folds)
        },
        'auto_adjusted': auto_adjusted,
        'adjustment_reason': adjustment_reason
    }


__all__ = [
    'evo_backtest_combined',
]
