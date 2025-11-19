"""
PRADO9_EVO — Rich Terminal Prediction Dashboard

Beautiful real-time prediction visualization using Rich library.

Author: PRADO9_EVO Builder
Date: 2025-01-18
Version: 1.0.0
"""

import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from typing import Dict, Any


class PredictionDashboard:
    """
    Rich-based terminal dashboard for live predictions.

    Displays:
    - Current signal and confidence
    - Position sizing and risk metrics
    - Strategy ensemble breakdown
    - Feature snapshot
    - Risk panel with stop/profit targets
    """

    def __init__(self, engine=None):
        """
        Initialize prediction dashboard.

        Args:
            engine: Optional LiveEngine instance
        """
        self.engine = engine
        self.console = Console()

    def render(self, symbol: str, pred: Dict[str, Any]):
        """
        Render the full prediction dashboard.

        Args:
            symbol: Trading symbol (e.g., "QQQ")
            pred: Prediction dictionary with all required fields
        """
        layout = Layout()

        # MAIN PANELS - 4 rows (added recommendation row)
        layout.split_column(
            Layout(name="recommendation", ratio=4),
            Layout(name="top", ratio=3),
            Layout(name="mid", ratio=3),
            Layout(name="bottom", ratio=4),
        )

        # --------------------
        # (0) RECOMMENDATION PANEL - THE ENTRY
        # --------------------
        signal_str = pred["signal"]
        confidence = pred["confidence"]

        # Determine recommendation and color
        if "LONG" in signal_str and "Strong" in signal_str:
            action = "🟢 ENTER LONG"
            action_style = "bold green"
            entry_text = f"Buy {symbol} | Target Size: {pred['position']['exposure']:.1f}%"
        elif "LONG" in signal_str and "Weak" in signal_str:
            action = "🟡 CONSIDER LONG"
            action_style = "bold yellow"
            entry_text = f"Small Buy {symbol} | Target Size: {pred['position']['exposure']:.1f}%"
        elif "SHORT" in signal_str and "Strong" in signal_str:
            action = "🔴 ENTER SHORT"
            action_style = "bold red"
            entry_text = f"Sell/Short {symbol} | Target Size: {pred['position']['exposure']:.1f}%"
        elif "SHORT" in signal_str and "Weak" in signal_str:
            action = "🟠 CONSIDER SHORT"
            action_style = "bold yellow"
            entry_text = f"Small Sell {symbol} | Target Size: {pred['position']['exposure']:.1f}%"
        else:
            action = "⚪ NO ENTRY"
            action_style = "bold white"
            entry_text = f"Stay in cash | Conflicting signals or low confidence"

        # Get current price if available
        current_price = pred.get("current_price", "N/A")
        stop_loss = pred["risk"]["stop_loss"]
        take_profit = pred["risk"]["take_profit"]

        rec_table = Table(show_header=False, expand=True, box=None)
        rec_table.add_column("Label", style="cyan", width=20)
        rec_table.add_column("Value", style=action_style)

        rec_table.add_row("ACTION", f"[{action_style}]{action}[/{action_style}]")
        rec_table.add_row("ENTRY", entry_text)
        if current_price != "N/A":
            rec_table.add_row("CURRENT PRICE", f"${current_price:.2f}")
        rec_table.add_row("STOP LOSS", f"${stop_loss:.2f}")
        rec_table.add_row("TAKE PROFIT", f"${take_profit:.2f}")

        layout["recommendation"].update(
            Panel(rec_table, title=f"📊 {symbol} TRADING RECOMMENDATION", border_style=action_style.split()[1])
        )

        # --------------------
        # (A) SIGNAL PANEL
        # --------------------
        signal_table = Table(title=f"{symbol} – Signal Analysis", expand=True)
        signal_table.add_column("Field", style="cyan")
        signal_table.add_column("Value", style="green")

        signal_table.add_row("Signal", pred["signal"])
        signal_table.add_row("Confidence", f"{pred['confidence']:.2f}")
        signal_table.add_row("Regime", pred["regime"])
        signal_table.add_row("Top Strategy", pred["top_strategy"])
        signal_table.add_row("Timestamp", str(datetime.datetime.now())[:19])

        layout["top"].update(Panel(signal_table, title="Signal Details"))

        # --------------------
        # (B) POSITION PANEL
        # --------------------
        pos_table = Table(title="Position Sizing", expand=True)
        pos_table.add_column("Field", style="cyan")
        pos_table.add_column("Value", style="yellow")

        pos = pred["position"]
        pos_table.add_row("Exposure %", f"{pos['exposure']:.2f}%")
        pos_table.add_row("Leverage", f"{pos['leverage']:.2f}x")
        pos_table.add_row("Vol Target", f"{pos['vol_target']:.2f}")
        pos_table.add_row("Adjusted Size", f"{pos['adj_size']:.2f}")
        pos_table.add_row("Position Floor", f"{pos['floor']:.2f}")

        layout["mid"].split_row(
            Layout(Panel(pos_table, title="Risk / Position")),
            Layout(name="right_mid")
        )

        # --------------------
        # (C) STRATEGY PANEL
        # --------------------
        strat_table = Table(title="Strategy Ensemble", expand=True)
        strat_table.add_column("Strategy", style="cyan")
        strat_table.add_column("Weight", style="magenta")
        strat_table.add_column("Signal", style="green")

        for s, val in pred["strategies"].items():
            strat_table.add_row(s, f"{val['weight']:.2f}", f"{val['signal']:.3f}")

        layout["right_mid"].update(Panel(strat_table, title="Ensemble Breakdown"))

        # --------------------
        # (D) FEATURE SNAPSHOT
        # --------------------
        feat_table = Table(title="Latest Feature Snapshot", expand=True)
        feat_table.add_column("Feature", style="cyan")
        feat_table.add_column("Value", style="white")

        for k, v in pred["features"].items():
            feat_table.add_row(k, f"{v:.4f}")

        # --------------------
        # (E) RISK PANEL
        # --------------------
        risk_table = Table(title="Risk Panel", expand=True)
        risk_table.add_column("Metric", style="cyan")
        risk_table.add_column("Value", style="red")

        risk = pred["risk"]
        risk_table.add_row("Stop Loss", f"{risk['stop_loss']:.2f}")
        risk_table.add_row("Take Profit", f"{risk['take_profit']:.2f}")
        risk_table.add_row("Expected DD (5d)", f"{risk['exp_dd']:.2f}")
        risk_table.add_row("Crisis Score", f"{risk['crisis_score']:.2f}")

        layout["bottom"].split_row(
            Layout(Panel(feat_table, title="Features", border_style="white")),
            Layout(Panel(risk_table, title="Risk", border_style="red"))
        )

        self.console.print(layout)
