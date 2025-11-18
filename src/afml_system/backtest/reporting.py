"""
PRADO9_EVO Module I — Backtest Reporting

Comprehensive report generation for backtest results.

Author: PRADO9_EVO Builder
Date: 2025-01-17
Version: 1.0.0
"""

from typing import Dict, Any, List
from datetime import datetime
import json
import numpy as np

from .backtest_engine import BacktestResult


# ============================================================================
# CONSTANTS
# ============================================================================

REPORTING_VERSION = '1.0.0'


# ============================================================================
# BACKTEST REPORT BUILDER
# ============================================================================

class BacktestReportBuilder:
    """
    Report builder for backtest results.

    Mini-Sweep I.1F: Enhanced with new metrics, sparklines, risk signatures, JSON export.

    Generates ASCII reports and structured summaries.
    """

    def _compute_enhanced_metrics(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Mini-Sweep I.1F: Compute enhanced metrics for reporting.

        Args:
            result: BacktestResult

        Returns:
            Enhanced metrics dictionary
        """
        metrics = {}

        # Profit Factor (already in BacktestResult, include for completeness)
        metrics['profit_factor'] = result.profit_factor

        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        win_prob = result.win_rate
        loss_prob = 1.0 - result.win_rate
        metrics['expectancy'] = (win_prob * result.avg_win) - (loss_prob * abs(result.avg_loss))

        # Trade Duration Stats (if trades have timestamps)
        if len(result.trades) >= 2:
            durations = []
            for i in range(1, len(result.trades)):
                if 'timestamp' in result.trades[i] and 'timestamp' in result.trades[i-1]:
                    duration = (result.trades[i]['timestamp'] - result.trades[i-1]['timestamp']).total_seconds() / 86400.0  # Days
                    durations.append(duration)

            if len(durations) > 0:
                metrics['trade_duration_mean'] = float(np.mean(durations))
                metrics['trade_duration_std'] = float(np.std(durations))
                metrics['trade_duration_min'] = float(np.min(durations))
                metrics['trade_duration_max'] = float(np.max(durations))
            else:
                metrics['trade_duration_mean'] = 0.0
                metrics['trade_duration_std'] = 0.0
                metrics['trade_duration_min'] = 0.0
                metrics['trade_duration_max'] = 0.0
        else:
            metrics['trade_duration_mean'] = 0.0
            metrics['trade_duration_std'] = 0.0
            metrics['trade_duration_min'] = 0.0
            metrics['trade_duration_max'] = 0.0

        # Volatility of Equity (from equity curve)
        if len(result.equity_curve) > 1:
            equity_values = [e['equity'] for e in result.equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            metrics['equity_volatility'] = float(np.std(returns)) if len(returns) > 0 else 0.0
        else:
            metrics['equity_volatility'] = 0.0

        # Max Runs (longest streak of wins/losses)
        if result.total_trades > 0:
            # Compute win/loss sequence
            trade_results = []
            for i in range(1, len(result.trades)):
                pnl = result.trades[i].get('pnl', 0.0)
                trade_results.append(1 if pnl > 0 else 0)

            if len(trade_results) > 0:
                max_win_run = 0
                max_loss_run = 0
                current_win_run = 0
                current_loss_run = 0

                for is_win in trade_results:
                    if is_win:
                        current_win_run += 1
                        current_loss_run = 0
                        max_win_run = max(max_win_run, current_win_run)
                    else:
                        current_loss_run += 1
                        current_win_run = 0
                        max_loss_run = max(max_loss_run, current_loss_run)

                metrics['max_win_run'] = max_win_run
                metrics['max_loss_run'] = max_loss_run
            else:
                metrics['max_win_run'] = 0
                metrics['max_loss_run'] = 0
        else:
            metrics['max_win_run'] = 0
            metrics['max_loss_run'] = 0

        return metrics

    def _generate_sparkline(self, values: List[float], width: int = 50) -> str:
        """
        Mini-Sweep I.1F: Generate ASCII sparkline for equity curve.

        Args:
            values: List of numeric values
            width: Width of sparkline (default: 50)

        Returns:
            ASCII sparkline string
        """
        if len(values) == 0:
            return " " * width

        # Resample values to fit width
        if len(values) > width:
            step = len(values) / width
            resampled = [values[int(i * step)] for i in range(width)]
        else:
            resampled = values + [values[-1]] * (width - len(values))

        # Normalize to 0-7 range (8 vertical levels)
        min_val = min(resampled)
        max_val = max(resampled)

        if max_val - min_val < 1e-10:
            normalized = [4] * len(resampled)  # Middle level
        else:
            normalized = [int(7 * (v - min_val) / (max_val - min_val)) for v in resampled]

        # Unicode block characters for sparklines
        chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        sparkline = ''.join([chars[n] for n in normalized])

        return sparkline

    def _compute_risk_signature(self, result: BacktestResult, enhanced_metrics: Dict[str, Any]) -> str:
        """
        Mini-Sweep I.1F: Generate risk signature summary.

        Args:
            result: BacktestResult
            enhanced_metrics: Enhanced metrics dictionary

        Returns:
            Risk signature string
        """
        # Classify risk profile
        sharpe = result.sharpe_ratio
        max_dd = abs(result.max_drawdown)
        vol = enhanced_metrics['equity_volatility']

        # Risk level classification
        if max_dd < 0.10 and vol < 0.02:
            risk_level = "LOW"
        elif max_dd < 0.20 and vol < 0.03:
            risk_level = "MODERATE"
        elif max_dd < 0.30 and vol < 0.05:
            risk_level = "ELEVATED"
        else:
            risk_level = "HIGH"

        # Return/Risk ratio
        if max_dd > 1e-6:
            return_risk_ratio = result.total_return / max_dd
        else:
            return_risk_ratio = 0.0

        signature = f"{risk_level} (DD={max_dd*100:.1f}%, Vol={vol*100:.2f}%, RR={return_risk_ratio:.2f}x)"

        return signature

    def build_standard_report(self, result: BacktestResult) -> str:
        """
        Build standard backtest report.

        Mini-Sweep I.1F: Enhanced with new metrics, sparklines, and risk signatures.

        Args:
            result: BacktestResult

        Returns:
            ASCII report string
        """
        # Mini-Sweep I.1F: Compute enhanced metrics
        enhanced = self._compute_enhanced_metrics(result)

        report = []
        report.append("=" * 80)
        report.append("PRADO9_EVO Standard Backtest Report")
        report.append("=" * 80)
        report.append("")
        report.append(f"Symbol: {result.symbol}")
        report.append(f"Period: {result.start_date} to {result.end_date}")
        report.append("")
        report.append("-" * 80)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 80)
        report.append(f"Initial Equity:    ${result.initial_equity:,.2f}")
        report.append(f"Final Equity:      ${result.final_equity:,.2f}")
        report.append(f"Total Return:      {result.total_return*100:,.2f}%")
        report.append("")
        report.append(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio:     {result.sortino_ratio:.2f}")
        report.append(f"Calmar Ratio:      {result.calmar_ratio:.2f}")
        report.append(f"Max Drawdown:      {result.max_drawdown*100:.2f}%")
        report.append("")

        # Mini-Sweep I.1F: Enhanced metrics
        report.append("-" * 80)
        report.append("ENHANCED METRICS")
        report.append("-" * 80)
        report.append(f"Profit Factor:     {enhanced['profit_factor']:.2f}")
        report.append(f"Expectancy:        ${enhanced['expectancy']:,.2f}")
        report.append(f"Equity Vol:        {enhanced['equity_volatility']*100:.2f}%")
        report.append(f"Max Win Run:       {enhanced['max_win_run']}")
        report.append(f"Max Loss Run:      {enhanced['max_loss_run']}")
        report.append("")

        report.append("-" * 80)
        report.append("TRADE STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Trades:      {result.total_trades}")
        report.append(f"Winning Trades:    {result.winning_trades}")
        report.append(f"Losing Trades:     {result.losing_trades}")
        report.append(f"Win Rate:          {result.win_rate*100:.2f}%")
        report.append(f"Avg Win:           ${result.avg_win:,.2f}")
        report.append(f"Avg Loss:          ${result.avg_loss:,.2f}")

        # Mini-Sweep I.1F: Trade duration stats
        if enhanced['trade_duration_mean'] > 0:
            report.append("")
            report.append(f"Trade Duration (days):")
            report.append(f"  Mean:            {enhanced['trade_duration_mean']:.2f}")
            report.append(f"  Std:             {enhanced['trade_duration_std']:.2f}")
            report.append(f"  Range:           [{enhanced['trade_duration_min']:.2f}, {enhanced['trade_duration_max']:.2f}]")

        report.append("")

        # Mini-Sweep I.1F: Risk signature
        risk_sig = self._compute_risk_signature(result, enhanced)
        report.append("-" * 80)
        report.append("RISK SIGNATURE")
        report.append("-" * 80)
        report.append(f"  {risk_sig}")
        report.append("")

        # Mini-Sweep I.1F: Equity curve sparkline
        if len(result.equity_curve) > 0:
            equity_values = [e['equity'] for e in result.equity_curve]
            sparkline = self._generate_sparkline(equity_values, width=60)
            report.append("-" * 80)
            report.append("EQUITY CURVE")
            report.append("-" * 80)
            report.append(f"  {sparkline}")
            report.append("")

        report.append("-" * 80)
        report.append("REGIME DISTRIBUTION")
        report.append("-" * 80)

        for regime, count in result.regime_counts.items():
            report.append(f"  {regime}: {count}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def build_walk_forward_report(self, wf_results: Dict[str, Any]) -> str:
        """
        Build walk-forward optimization report.

        Args:
            wf_results: Walk-forward results dictionary

        Returns:
            ASCII report string
        """
        report = []
        report.append("=" * 80)
        report.append("PRADO9_EVO Walk-Forward Optimization Report")
        report.append("=" * 80)
        report.append("")
        report.append(f"Symbol: {wf_results['symbol']}")
        report.append(f"Train Window: {wf_results['train_window']} bars")
        report.append(f"Test Window: {wf_results['test_window']} bars")
        report.append(f"Number of Folds: {wf_results['num_folds']}")
        report.append("")
        report.append("-" * 80)
        report.append("AGGREGATED METRICS")
        report.append("-" * 80)

        agg = wf_results['aggregated']

        report.append(f"Sharpe (Mean ± Std): {agg['sharpe_mean']:.2f} ± {agg['sharpe_std']:.2f}")
        report.append(f"Sortino (Mean):      {agg['sortino_mean']:.2f}")
        report.append(f"Calmar (Mean):       {agg['calmar_mean']:.2f}")
        report.append(f"Drawdown (Mean):     {agg['max_drawdown_mean']*100:.2f}%")
        report.append(f"Return (Mean):       {agg['total_return_mean']*100:.2f}%")
        report.append("")
        report.append(f"Consistency:         {agg['consistency_pct']:.1f}%")
        report.append(f"Positive Folds:      {agg['positive_folds']} / {agg['total_folds']}")
        report.append("")
        report.append("-" * 80)
        report.append("FOLD DETAILS")
        report.append("-" * 80)

        for fold in wf_results['folds']:
            report.append(f"Fold {fold['fold_idx']}: Sharpe={fold['sharpe']:.2f}, Return={fold['total_return']*100:.2f}%")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def build_crisis_report(self, crisis_results: Dict[str, Any]) -> str:
        """
        Build crisis stress test report.

        Args:
            crisis_results: Crisis test results dictionary

        Returns:
            ASCII report string
        """
        report = []
        report.append("=" * 80)
        report.append("PRADO9_EVO Crisis Stress Test Report")
        report.append("=" * 80)
        report.append("")
        report.append(f"Symbol: {crisis_results['symbol']}")
        report.append(f"Number of Crises Tested: {crisis_results['num_crises']}")
        report.append("")
        report.append("-" * 80)
        report.append("SUMMARY")
        report.append("-" * 80)

        summary = crisis_results['summary']

        report.append(f"Avg Sharpe:       {summary['avg_sharpe']:.2f}")
        report.append(f"Avg Drawdown:     {summary['avg_drawdown']*100:.2f}%")
        report.append(f"Avg Return:       {summary['avg_return']*100:.2f}%")
        report.append(f"Survival Rate:    {summary['survival_rate']:.1f}%")
        report.append(f"Survived:         {summary['survived']} / {summary['total']}")
        report.append("")
        report.append("-" * 80)
        report.append("CRISIS DETAILS")
        report.append("-" * 80)

        for crisis in crisis_results['crises']:
            survived_str = "✓ SURVIVED" if crisis.get('survived', False) else "✗ FAILED"
            report.append(f"\n{crisis['name']}: {survived_str}")
            report.append(f"  Test Period: {crisis['test_start']} to {crisis['test_end']}")
            report.append(f"  Sharpe: {crisis.get('sharpe', 0.0):.2f}")
            report.append(f"  Drawdown: {crisis.get('max_drawdown', 0.0)*100:.2f}%")
            report.append(f"  Return: {crisis.get('total_return', 0.0)*100:.2f}%")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def build_mc_report(self, mc_results: Dict[str, Any]) -> str:
        """
        Build Monte Carlo skill assessment report.

        Args:
            mc_results: Monte Carlo results dictionary

        Returns:
            ASCII report string
        """
        report = []
        report.append("=" * 80)
        report.append("PRADO9_EVO Monte Carlo Skill Assessment Report")
        report.append("=" * 80)
        report.append("")
        report.append(f"Symbol: {mc_results['symbol']}")
        report.append(f"Simulations: {mc_results['num_simulations']:,}")
        report.append("")
        report.append("-" * 80)
        report.append("SKILL ASSESSMENT")
        report.append("-" * 80)
        report.append(f"Actual Sharpe:        {mc_results['actual_sharpe']:.2f}")
        report.append(f"MC Sharpe (Mean):     {mc_results['mc_sharpe_mean']:.2f}")
        report.append(f"MC Sharpe (Std):      {mc_results['mc_sharpe_std']:.2f}")
        report.append("")
        report.append(f"Skill Percentile:     {mc_results['skill_percentile']:.1f}%")
        report.append(f"P-Value:              {mc_results['p_value']:.4f}")
        report.append(f"Statistically Sig.:   {'YES' if mc_results['significant'] else 'NO'}")
        report.append("")

        dist = mc_results.get('mc_distribution', {})
        if dist:
            report.append("-" * 80)
            report.append("MC DISTRIBUTION")
            report.append("-" * 80)
            report.append(f"  Min:     {dist.get('min', 0.0):.2f}")
            report.append(f"  Q25:     {dist.get('q25', 0.0):.2f}")
            report.append(f"  Median:  {dist.get('median', 0.0):.2f}")
            report.append(f"  Q75:     {dist.get('q75', 0.0):.2f}")
            report.append(f"  Max:     {dist.get('max', 0.0):.2f}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def build_comprehensive_report(
        self,
        standard: BacktestResult,
        walk_forward: Dict[str, Any],
        crisis: Dict[str, Any],
        monte_carlo: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build comprehensive validation report.

        Args:
            standard: Standard backtest result
            walk_forward: Walk-forward results
            crisis: Crisis test results
            monte_carlo: Monte Carlo results

        Returns:
            Comprehensive report dictionary
        """
        # Generate individual reports
        standard_report = self.build_standard_report(standard)
        wf_report = self.build_walk_forward_report(walk_forward)
        crisis_report = self.build_crisis_report(crisis)
        mc_report = self.build_mc_report(monte_carlo)

        # Build comprehensive summary
        comprehensive_summary = []
        comprehensive_summary.append("=" * 80)
        comprehensive_summary.append("PRADO9_EVO COMPREHENSIVE VALIDATION REPORT")
        comprehensive_summary.append("=" * 80)
        comprehensive_summary.append("")
        comprehensive_summary.append(f"Symbol: {standard.symbol}")
        comprehensive_summary.append(f"Generated: {datetime.now()}")
        comprehensive_summary.append("")
        comprehensive_summary.append("-" * 80)
        comprehensive_summary.append("OVERALL METRICS")
        comprehensive_summary.append("-" * 80)
        comprehensive_summary.append(f"Standard Sharpe:      {standard.sharpe_ratio:.2f}")
        comprehensive_summary.append(f"WFO Sharpe:           {walk_forward['aggregated']['sharpe_mean']:.2f} ± {walk_forward['aggregated']['sharpe_std']:.2f}")
        comprehensive_summary.append(f"Crisis Survival:      {crisis['summary']['survival_rate']:.1f}%")
        comprehensive_summary.append(f"MC Skill Percentile:  {monte_carlo['skill_percentile']:.1f}%")
        comprehensive_summary.append("")

        # Determine final verdict
        verdict = self._determine_verdict(standard, walk_forward, crisis, monte_carlo)

        comprehensive_summary.append("-" * 80)
        comprehensive_summary.append("FINAL VERDICT")
        comprehensive_summary.append("-" * 80)
        comprehensive_summary.append(f"Status: {verdict['status']}")
        comprehensive_summary.append(f"Grade:  {verdict['grade']}")
        comprehensive_summary.append("")

        for criterion, passed in verdict['criteria'].items():
            status_str = "✓ PASS" if passed else "✗ FAIL"
            comprehensive_summary.append(f"  {criterion}: {status_str}")

        comprehensive_summary.append("")
        comprehensive_summary.append("=" * 80)

        summary_text = "\n".join(comprehensive_summary)

        # Build comprehensive result
        result = {
            'symbol': standard.symbol,
            'timestamp': datetime.now(),
            'summary': summary_text,
            'verdict': verdict,
            'reports': {
                'standard': standard_report,
                'walk_forward': wf_report,
                'crisis': crisis_report,
                'monte_carlo': mc_report
            },
            'metrics': {
                'standard_sharpe': standard.sharpe_ratio,
                'wfo_sharpe_mean': walk_forward['aggregated']['sharpe_mean'],
                'wfo_sharpe_std': walk_forward['aggregated']['sharpe_std'],
                'wfo_consistency': walk_forward['aggregated']['consistency_pct'],
                'crisis_survival_rate': crisis['summary']['survival_rate'],
                'mc_skill_percentile': monte_carlo['skill_percentile'],
                'mc_p_value': monte_carlo['p_value']
            }
        }

        return result

    def _determine_verdict(
        self,
        standard: BacktestResult,
        walk_forward: Dict[str, Any],
        crisis: Dict[str, Any],
        monte_carlo: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine final validation verdict.

        Args:
            standard: Standard backtest result
            walk_forward: Walk-forward results
            crisis: Crisis test results
            monte_carlo: Monte Carlo results

        Returns:
            Verdict dictionary
        """
        # Define pass criteria
        criteria = {
            'Standard Sharpe > 1.0': standard.sharpe_ratio > 1.0,
            'WFO Sharpe > 1.0': walk_forward['aggregated']['sharpe_mean'] > 1.0,
            'WFO Consistency > 60%': walk_forward['aggregated']['consistency_pct'] > 60.0,
            'Crisis Survival > 50%': crisis['summary']['survival_rate'] > 50.0,
            'MC Skill > 75th percentile': monte_carlo['skill_percentile'] > 75.0,
            'MC p-value < 0.05': monte_carlo['p_value'] < 0.05
        }

        # Count passes
        passed_count = sum(criteria.values())
        total_count = len(criteria)

        # Determine status
        if passed_count == total_count:
            status = "PASS"
            grade = "A"
        elif passed_count >= total_count * 0.8:
            status = "PASS"
            grade = "B"
        elif passed_count >= total_count * 0.6:
            status = "CONDITIONAL PASS"
            grade = "C"
        else:
            status = "FAIL"
            grade = "F"

        verdict = {
            'status': status,
            'grade': grade,
            'criteria': criteria,
            'passed': passed_count,
            'total': total_count,
            'pass_rate': (passed_count / total_count) * 100.0
        }

        return verdict

    def to_json(self, result: BacktestResult, filepath: str = None) -> str:
        """
        Mini-Sweep I.1F: Export backtest result to JSON.

        Args:
            result: BacktestResult
            filepath: Optional filepath to write JSON (if None, returns JSON string)

        Returns:
            JSON string
        """
        # Compute enhanced metrics
        enhanced = self._compute_enhanced_metrics(result)

        # Build exportable structure
        export_data = {
            'symbol': result.symbol,
            'start_date': result.start_date.isoformat() if isinstance(result.start_date, datetime) else str(result.start_date),
            'end_date': result.end_date.isoformat() if isinstance(result.end_date, datetime) else str(result.end_date),
            'initial_equity': result.initial_equity,
            'final_equity': result.final_equity,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'max_drawdown': result.max_drawdown,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'profit_factor': result.profit_factor,
            'regime_counts': result.regime_counts,
            'enhanced_metrics': {
                'expectancy': enhanced['expectancy'],
                'equity_volatility': enhanced['equity_volatility'],
                'max_win_run': enhanced['max_win_run'],
                'max_loss_run': enhanced['max_loss_run'],
                'trade_duration_mean': enhanced['trade_duration_mean'],
                'trade_duration_std': enhanced['trade_duration_std'],
                'trade_duration_min': enhanced['trade_duration_min'],
                'trade_duration_max': enhanced['trade_duration_max']
            },
            'risk_signature': self._compute_risk_signature(result, enhanced),
            'equity_curve': result.equity_curve,
            'trades': result.trades
        }

        # Convert to JSON
        json_str = json.dumps(export_data, indent=2, default=str)

        # Write to file if filepath provided
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str


# ============================================================================
# MINI-SWEEP I.1F TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Reporting Module — Mini-Sweep I.1F Tests")
    print("=" * 80)

    # Setup test data
    from datetime import datetime, timedelta
    import numpy as np

    # Create mock BacktestResult
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)

    # Generate synthetic equity curve
    np.random.seed(42)
    num_points = 100
    equity_curve = []
    equity = 100000.0

    for i in range(num_points):
        equity += np.random.randn() * 2000  # Random walk
        equity_curve.append({
            'timestamp': start_date + timedelta(days=i),
            'equity': equity
        })

    # Generate synthetic trades
    trades = []
    current_equity = 100000.0

    for i in range(30):
        # Alternate wins/losses with some randomness
        pnl = np.random.randn() * 1000
        if i % 3 == 0:  # Force some wins
            pnl = abs(pnl)
        elif i % 5 == 0:  # Force some losses
            pnl = -abs(pnl)

        current_equity += pnl

        trades.append({
            'timestamp': start_date + timedelta(days=i*3),
            'equity_after': current_equity,
            'pnl': pnl
        })

    # Compute trade stats
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    losing_trades = len([t for t in trades if t['pnl'] <= 0])
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0.0
    avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if losing_trades > 0 else 0.0

    gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] <= 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Create synthetic returns and drawdown series
    import pandas as pd
    dates = [start_date + timedelta(days=i) for i in range(num_points)]
    equity_vals = [e['equity'] for e in equity_curve]
    returns_vals = np.diff(equity_vals) / equity_vals[:-1]
    returns_vals = np.concatenate([[0.0], returns_vals])  # Prepend 0 for first day

    # Compute drawdown
    cummax = np.maximum.accumulate(equity_vals)
    drawdown_vals = (equity_vals - cummax) / cummax

    returns_series = pd.Series(returns_vals, index=dates)
    drawdown_series = pd.Series(drawdown_vals, index=dates)
    equity_series = pd.Series(equity_vals, index=dates)

    # Create BacktestResult
    mock_result = BacktestResult(
        symbol='TEST',
        start_date=start_date,
        end_date=end_date,
        initial_equity=100000.0,
        final_equity=current_equity,
        total_return=(current_equity - 100000.0) / 100000.0,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=3.0,
        max_drawdown=-0.15,
        total_trades=30,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / 30.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        equity_curve=equity_curve,
        returns=returns_series,
        drawdown=drawdown_series,
        regime_counts={'BULL': 15, 'BEAR': 10, 'NORMAL': 5},
        trades=trades,
        metrics={}
    )

    builder = BacktestReportBuilder()

    # ========================================================================
    # TEST 1: Mini-Sweep I.1F - Enhanced metrics and sparklines
    # ========================================================================
    print("\n[TEST 1] Mini-Sweep I.1F: Enhanced metrics, sparklines, risk signatures")
    print("-" * 80)

    # Compute enhanced metrics
    enhanced = builder._compute_enhanced_metrics(mock_result)

    # Verify enhanced metrics exist
    assert 'expectancy' in enhanced, "Expectancy metric missing"
    assert 'equity_volatility' in enhanced, "Equity volatility missing"
    assert 'max_win_run' in enhanced, "Max win run missing"
    assert 'max_loss_run' in enhanced, "Max loss run missing"
    assert 'trade_duration_mean' in enhanced, "Trade duration mean missing"

    print(f"  Enhanced Metrics:")
    print(f"    Expectancy:        ${enhanced['expectancy']:,.2f}")
    print(f"    Equity Vol:        {enhanced['equity_volatility']*100:.2f}%")
    print(f"    Max Win Run:       {enhanced['max_win_run']}")
    print(f"    Max Loss Run:      {enhanced['max_loss_run']}")
    print(f"    Trade Dur (mean):  {enhanced['trade_duration_mean']:.2f} days")
    print("  ✓ Enhanced metrics computed")

    # Test sparkline generation
    equity_values = [e['equity'] for e in equity_curve]
    sparkline = builder._generate_sparkline(equity_values, width=50)

    assert len(sparkline) == 50, f"Sparkline should be 50 chars, got {len(sparkline)}"
    print(f"\n  Sparkline (50 chars): {sparkline}")
    print("  ✓ ASCII sparkline generated")

    # Test risk signature
    risk_sig = builder._compute_risk_signature(mock_result, enhanced)
    assert 'DD=' in risk_sig, "Risk signature should contain drawdown"
    assert 'Vol=' in risk_sig, "Risk signature should contain volatility"
    assert 'RR=' in risk_sig, "Risk signature should contain return/risk ratio"

    print(f"\n  Risk Signature: {risk_sig}")
    print("  ✓ Risk signature generated")

    # Build full report with enhancements
    report = builder.build_standard_report(mock_result)

    assert 'ENHANCED METRICS' in report, "Report should contain enhanced metrics section"
    assert 'EQUITY CURVE' in report, "Report should contain equity curve sparkline"
    assert 'RISK SIGNATURE' in report, "Report should contain risk signature"
    assert 'Expectancy' in report, "Report should show expectancy"
    assert 'Max Win Run' in report, "Report should show max win run"

    print("\n  Full Report Preview (first 40 lines):")
    report_lines = report.split('\n')
    for line in report_lines[:40]:
        print(f"    {line}")

    print("  ✓ Enhanced report generated successfully")

    # ========================================================================
    # TEST 2: Mini-Sweep I.1F - JSON export
    # ========================================================================
    print("\n[TEST 2] Mini-Sweep I.1F: JSON export functionality")
    print("-" * 80)

    # Export to JSON string
    json_str = builder.to_json(mock_result)

    # Verify JSON is valid
    import json as json_lib
    parsed = json_lib.loads(json_str)

    assert parsed['symbol'] == 'TEST', "Symbol should match"
    assert 'enhanced_metrics' in parsed, "Should contain enhanced metrics"
    assert 'risk_signature' in parsed, "Should contain risk signature"
    assert 'equity_curve' in parsed, "Should contain equity curve"
    assert 'trades' in parsed, "Should contain trades"

    # Verify enhanced metrics in JSON
    assert 'expectancy' in parsed['enhanced_metrics'], "Enhanced metrics missing expectancy"
    assert 'equity_volatility' in parsed['enhanced_metrics'], "Enhanced metrics missing equity_volatility"
    assert 'max_win_run' in parsed['enhanced_metrics'], "Enhanced metrics missing max_win_run"
    assert 'trade_duration_mean' in parsed['enhanced_metrics'], "Enhanced metrics missing trade_duration_mean"

    print(f"  JSON Export Structure:")
    print(f"    Symbol:              {parsed['symbol']}")
    print(f"    Total Trades:        {parsed['total_trades']}")
    print(f"    Sharpe Ratio:        {parsed['sharpe_ratio']:.2f}")
    print(f"    Enhanced Metrics:    {len(parsed['enhanced_metrics'])} fields")
    print(f"    Equity Curve Points: {len(parsed['equity_curve'])}")
    print(f"    Risk Signature:      {parsed['risk_signature']}")
    print("  ✓ JSON export valid")

    # Test file export
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, 'backtest_result.json')
        builder.to_json(mock_result, filepath=json_path)

        # Verify file exists and is valid
        assert os.path.exists(json_path), "JSON file should be created"

        with open(json_path, 'r') as f:
            file_content = f.read()
            file_parsed = json_lib.loads(file_content)

        assert file_parsed['symbol'] == 'TEST', "File content should match"
        print(f"  ✓ JSON file export successful ({len(file_content)} bytes)")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL REPORTING TESTS PASSED (2 TESTS)")
    print("=" * 80)
    print("\nMini-Sweep I.1F Enhancements:")
    print("  ✓ New Metrics:")
    print("    - Profit Factor (already in BacktestResult)")
    print("    - Expectancy")
    print("    - Trade Duration Stats (mean, std, min, max)")
    print("    - Volatility of Equity")
    print("    - Max Runs (longest win/loss streaks)")
    print("  ✓ ASCII Sparklines (equity curve visualization)")
    print("  ✓ Risk Signature Summaries (risk level classification)")
    print("  ✓ to_json() Export (full result serialization)")
    print("\nReporting Module: PRODUCTION READY")
    print("=" * 80)
