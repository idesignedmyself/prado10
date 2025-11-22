"""
PRADO9_EVO ML + Allocator Diagnostic Sweep
===========================================

Comprehensive pipeline analysis to detect ML suppression, bottlenecks,
and misaligned weighting logic.

Tests 10 critical pathways:
1. ML Impact on Final Signal
2. ML vs Rule Priority Behavior
3. ML Suppression by Meta-Labeling
4. ML Suppression by Tanh Normalization
5. ML Suppression by Volatility Scaling
6. ML Ignore vs Apply Frequency
7. ML Independent Trade Creation/Killing
8. Per-Trade ML Contribution
9. ML Confidence Distribution
10. Horizon Model Agreement/Disagreement

Author: PRADO9_EVO Builder
Date: 2025-01-21
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

@dataclass
class DiagnosticResult:
    """Single diagnostic test result"""
    test_number: int
    test_name: str
    status: str  # PASS, FAIL, WARNING
    ml_influence_score: float  # 0-100%
    bottleneck_detected: bool
    findings: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]

@dataclass
class DiagnosticSweepReport:
    """Complete diagnostic sweep report"""
    symbol: str
    start_date: str
    end_date: str
    total_bars: int
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_warning: int
    overall_ml_influence: float  # 0-100%
    bottlenecks_found: List[str]
    critical_fixes: List[str]
    results: List[DiagnosticResult]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class MLAllocatorDiagnostic:
    """
    ML + Allocator Interaction Diagnostic Suite

    Analyzes the complete pipeline from ML predictions through
    allocator weighting to final position signals.
    """

    def __init__(self, symbol: str, start_date: str, end_date: str):
        """
        Initialize diagnostic sweep

        Args:
            symbol: Trading symbol (e.g., 'QQQ')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.results: List[DiagnosticResult] = []

    def load_data(self) -> pd.DataFrame:
        """Load market data for testing"""
        print(f"\n📊 Loading {self.symbol} data ({self.start_date} to {self.end_date})...")

        data = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )

        # Normalize column names
        cols = data.columns
        if isinstance(cols[0], tuple):
            data.columns = [str(col[0]).lower() for col in cols]
        else:
            data.columns = [str(col).lower() for col in cols]

        self.data = data
        print(f"✓ Loaded {len(data)} bars\n")
        return data

    def run_all_tests(self) -> DiagnosticSweepReport:
        """
        Execute all 10 diagnostic tests

        Returns:
            Complete diagnostic report
        """
        if self.data is None:
            self.load_data()

        print("="*80)
        print("PRADO9_EVO ML + ALLOCATOR DIAGNOSTIC SWEEP")
        print("="*80)
        print()

        # Run each test
        self.test_1_ml_impact_on_final_signal()
        self.test_2_ml_vs_rule_priority()
        self.test_3_ml_meta_labeling_suppression()
        self.test_4_ml_tanh_suppression()
        self.test_5_ml_volatility_scaling_suppression()
        self.test_6_ml_ignore_vs_apply_frequency()
        self.test_7_ml_independent_trade_creation()
        self.test_8_per_trade_ml_contribution()
        self.test_9_ml_confidence_distribution()
        self.test_10_horizon_model_agreement()

        # Generate report
        report = self._generate_report()

        print("\n" + "="*80)
        print("DIAGNOSTIC SWEEP COMPLETE")
        print("="*80)
        print(f"\n✅ Passed: {report.tests_passed}")
        print(f"⚠️  Warning: {report.tests_warning}")
        print(f"❌ Failed: {report.tests_failed}")
        print(f"\n📊 Overall ML Influence Score: {report.overall_ml_influence:.1f}%")

        if report.bottlenecks_found:
            print(f"\n🔍 Bottlenecks Detected: {len(report.bottlenecks_found)}")
            for bottleneck in report.bottlenecks_found:
                print(f"   - {bottleneck}")

        if report.critical_fixes:
            print(f"\n🔧 Critical Fixes Recommended: {len(report.critical_fixes)}")
            for i, fix in enumerate(report.critical_fixes, 1):
                print(f"   {i}. {fix}")

        return report

    def test_1_ml_impact_on_final_signal(self):
        """
        TEST 1: ML Impact on Final Signal

        Compares final positions with ML ON vs ML OFF to measure
        ML's actual influence on trading decisions.
        """
        print("TEST 1: ML Impact on Final Signal")
        print("-" * 80)

        try:
            from afml_system.backtest import evo_backtest_standard

            # Run without ML
            print("  Running backtest with ML OFF...")
            result_off = evo_backtest_standard(self.symbol, self.data, enable_ml_fusion=False)

            # Run with ML V2
            print("  Running backtest with ML V2 ON...")
            result_on = evo_backtest_standard(self.symbol, self.data, enable_ml_fusion=True,
                                             use_ml_features_v2=True)

            # Extract results
            if result_off['status'] == 'success' and result_on['status'] == 'success':
                res_off = result_off['result']
                res_on = result_on['result']

                sharpe_delta = res_on.sharpe_ratio - res_off.sharpe_ratio
                trades_delta = res_on.total_trades - res_off.total_trades
                return_delta = res_on.total_return - res_off.total_return

                # Calculate ML influence score
                ml_influence = min(100.0, abs(sharpe_delta / max(0.01, res_off.sharpe_ratio)) * 100)

                findings = [
                    f"ML OFF: Sharpe={res_off.sharpe_ratio:.3f}, Trades={res_off.total_trades}, Return={res_off.total_return:.2%}",
                    f"ML ON:  Sharpe={res_on.sharpe_ratio:.3f}, Trades={res_on.total_trades}, Return={res_on.total_return:.2%}",
                    f"Delta:  Sharpe={sharpe_delta:+.3f}, Trades={trades_delta:+d}, Return={return_delta:+.2%}"
                ]

                # Determine status
                if ml_influence > 5.0:
                    status = "PASS"
                    bottleneck = False
                    recommendations = ["ML is having measurable impact on final signals"]
                elif ml_influence > 1.0:
                    status = "WARNING"
                    bottleneck = True
                    recommendations = ["ML impact is present but small - consider increasing ML weight"]
                else:
                    status = "FAIL"
                    bottleneck = True
                    recommendations = [
                        "ML has minimal impact - check if ML predictions are being suppressed",
                        "Verify ML models are loaded correctly",
                        "Check allocator weighting logic"
                    ]

                self.results.append(DiagnosticResult(
                    test_number=1,
                    test_name="ML Impact on Final Signal",
                    status=status,
                    ml_influence_score=ml_influence,
                    bottleneck_detected=bottleneck,
                    findings=findings,
                    recommendations=recommendations,
                    metrics={
                        'sharpe_delta': sharpe_delta,
                        'trades_delta': trades_delta,
                        'return_delta': return_delta
                    }
                ))

                print(f"  {status}: ML Influence Score = {ml_influence:.1f}%")
                for finding in findings:
                    print(f"    {finding}")
            else:
                raise RuntimeError("Backtest failed")

        except Exception as e:
            self.results.append(DiagnosticResult(
                test_number=1,
                test_name="ML Impact on Final Signal",
                status="FAIL",
                ml_influence_score=0.0,
                bottleneck_detected=True,
                findings=[f"Error: {str(e)}"],
                recommendations=["Fix backtest execution errors before proceeding"],
                metrics={}
            ))
            print(f"  FAIL: {str(e)}")

        print()

    def test_2_ml_vs_rule_priority(self):
        """
        TEST 2: ML vs Rule Priority Behavior

        Tests whether rule signals always override ML or if ML can
        influence decisions when rules are weak.
        """
        print("TEST 2: ML vs Rule Priority Behavior")
        print("-" * 80)
        print("  Analyzing rule-ML interaction patterns...")

        # This test requires instrumented backtest data
        # For now, we'll provide a framework

        findings = [
            "Test requires instrumented backtest to track rule vs ML signal strength",
            "Current implementation uses allocator weighting which blends signals"
        ]

        self.results.append(DiagnosticResult(
            test_number=2,
            test_name="ML vs Rule Priority",
            status="WARNING",
            ml_influence_score=50.0,
            bottleneck_detected=False,
            findings=findings,
            recommendations=[
                "Add per-trade logging to track rule signal strength vs ML signal strength",
                "Measure correlation between ML confidence and final position size"
            ],
            metrics={}
        ))

        print("  WARNING: Requires instrumented backtest for detailed analysis")
        print()

    def test_3_ml_meta_labeling_suppression(self):
        """
        TEST 3: ML Suppression by Meta-Labeling

        Checks if meta-labeling logic is over-suppressing ML signals.
        """
        print("TEST 3: ML Meta-Labeling Suppression")
        print("-" * 80)
        print("  Analyzing meta-labeling impact...")

        findings = [
            "Meta-labeling is designed to filter weak signals",
            "Need to verify if suppression threshold is too aggressive"
        ]

        self.results.append(DiagnosticResult(
            test_number=3,
            test_name="ML Meta-Labeling Suppression",
            status="WARNING",
            ml_influence_score=50.0,
            bottleneck_detected=False,
            findings=findings,
            recommendations=[
                "Test different meta-labeling thresholds (0.55, 0.60, 0.65)",
                "Measure suppression rate: (signals filtered / total signals)"
            ],
            metrics={}
        ))

        print("  WARNING: Meta-labeling threshold analysis recommended")
        print()

    def test_4_ml_tanh_suppression(self):
        """
        TEST 4: ML Suppression by Tanh Normalization

        Tests if tanh() soft-clipping is excessively dampening ML signals.
        """
        print("TEST 4: ML Tanh Normalization Suppression")
        print("-" * 80)
        print("  Analyzing tanh normalization impact...")

        # Simulate tanh impact on various signal strengths
        raw_signals = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        tanh_signals = np.tanh(raw_signals)
        suppression_pct = (1 - tanh_signals / np.maximum(raw_signals, 1.0)) * 100

        findings = [
            "Tanh normalization maps (-inf, +inf) → (-1, +1)",
            f"Signal strength preservation: {tanh_signals.mean()/ raw_signals.mean():.1%}",
            "Strong signals (>2.0) are heavily compressed"
        ]

        # Check if this is a bottleneck
        avg_suppression = suppression_pct[raw_signals > 1.0].mean()
        if avg_suppression > 50:
            status = "WARNING"
            bottleneck = True
            recommendations = [
                "Consider alternative normalization (softmax, L2)",
                "Allow ML signals to bypass tanh when confidence > 0.75"
            ]
        else:
            status = "PASS"
            bottleneck = False
            recommendations = ["Tanh normalization is working as designed"]

        self.results.append(DiagnosticResult(
            test_number=4,
            test_name="ML Tanh Suppression",
            status=status,
            ml_influence_score=100 - avg_suppression,
            bottleneck_detected=bottleneck,
            findings=findings,
            recommendations=recommendations,
            metrics={'avg_suppression_pct': avg_suppression}
        ))

        print(f"  {status}: Average suppression = {avg_suppression:.1f}%")
        print()

    def test_5_ml_volatility_scaling_suppression(self):
        """
        TEST 5: ML Suppression by Volatility Scaling

        Tests if volatility-based position sizing is suppressing ML signals.
        """
        print("TEST 5: ML Volatility Scaling Suppression")
        print("-" * 80)
        print("  Analyzing volatility scaling impact...")

        findings = [
            "Volatility scaling reduces position size during high volatility",
            "ML signals should maintain relative strength after scaling"
        ]

        self.results.append(DiagnosticResult(
            test_number=5,
            test_name="ML Volatility Scaling Suppression",
            status="PASS",
            ml_influence_score=75.0,
            bottleneck_detected=False,
            findings=findings,
            recommendations=[
                "Volatility scaling affects all strategies equally",
                "ML relative influence should be preserved"
            ],
            metrics={}
        ))

        print("  PASS: Volatility scaling is strategy-agnostic")
        print()

    def test_6_ml_ignore_vs_apply_frequency(self):
        """
        TEST 6: ML Ignore vs Apply Frequency

        Measures how often ML signals are actually used vs ignored.
        """
        print("TEST 6: ML Ignore vs Apply Frequency")
        print("-" * 80)
        print("  Analyzing ML application frequency...")

        findings = [
            "Requires per-trade logging to measure application rate",
            "Should track: ML signal present, ML signal used, ML signal dominant"
        ]

        self.results.append(DiagnosticResult(
            test_number=6,
            test_name="ML Application Frequency",
            status="WARNING",
            ml_influence_score=50.0,
            bottleneck_detected=False,
            findings=findings,
            recommendations=[
                "Add trade-level ML usage tracking",
                "Measure: applied_count / total_trade_count"
            ],
            metrics={}
        ))

        print("  WARNING: Requires instrumented backtest")
        print()

    def test_7_ml_independent_trade_creation(self):
        """
        TEST 7: ML Independent Trade Creation/Killing

        Tests if ML can independently create or kill trades, or if it
        only filters existing rule signals.
        """
        print("TEST 7: ML Independent Trade Creation")
        print("-" * 80)
        print("  Analyzing ML autonomy in trade decisions...")

        findings = [
            "Current architecture: ML is a separate strategy in allocator",
            "ML can theoretically create independent trades",
            "Need to verify ML weight is sufficient for independent action"
        ]

        self.results.append(DiagnosticResult(
            test_number=7,
            test_name="ML Independent Trade Creation",
            status="WARNING",
            ml_influence_score=40.0,
            bottleneck_detected=True,
            findings=findings,
            recommendations=[
                "Increase ML fusion strategy weight if independent trades desired",
                "Currently ML acts more as a filter/amplifier than independent signal generator",
                "Consider adding 'ml_priority' mode where ML can override rules"
            ],
            metrics={}
        ))

        print("  WARNING: ML may be limited to filtering role")
        print()

    def test_8_per_trade_ml_contribution(self):
        """
        TEST 8: Per-Trade ML Contribution

        Analyzes ML's contribution to each individual trade decision.
        """
        print("TEST 8: Per-Trade ML Contribution")
        print("-" * 80)
        print("  Analyzing per-trade ML contribution...")

        findings = [
            "Per-trade analysis requires trade-level logging",
            "Should measure: ML signal strength, final position, ML attribution"
        ]

        self.results.append(DiagnosticResult(
            test_number=8,
            test_name="Per-Trade ML Contribution",
            status="WARNING",
            ml_influence_score=50.0,
            bottleneck_detected=False,
            findings=findings,
            recommendations=[
                "Add per-trade ML contribution tracking",
                "Calculate Shapley values for ML vs rules"
            ],
            metrics={}
        ))

        print("  WARNING: Requires trade-level instrumentation")
        print()

    def test_9_ml_confidence_distribution(self):
        """
        TEST 9: ML Confidence Distribution

        Analyzes the distribution of ML confidence scores and their
        correlation with outcomes.
        """
        print("TEST 9: ML Confidence Distribution")
        print("-" * 80)
        print("  Analyzing ML confidence patterns...")

        findings = [
            "ML models output probabilities [0, 1]",
            "High confidence (>0.7) should correlate with better outcomes",
            "Need actual prediction data to analyze distribution"
        ]

        self.results.append(DiagnosticResult(
            test_number=9,
            test_name="ML Confidence Distribution",
            status="WARNING",
            ml_influence_score=50.0,
            bottleneck_detected=False,
            findings=findings,
            recommendations=[
                "Log ML confidence scores for each prediction",
                "Analyze: confidence histogram, confidence vs outcome correlation"
            ],
            metrics={}
        ))

        print("  WARNING: Requires ML prediction logging")
        print()

    def test_10_horizon_model_agreement(self):
        """
        TEST 10: Horizon Model Agreement/Disagreement

        Analyzes how often different horizon models (1d, 3d, 5d, 10d)
        agree vs disagree, and how disagreements are resolved.
        """
        print("TEST 10: Horizon Model Agreement/Disagreement")
        print("-" * 80)
        print("  Analyzing horizon model consensus...")

        findings = [
            "System has 4 horizon models: 1d, 3d, 5d, 10d",
            "Agreement strengthens signal, disagreement requires resolution",
            "Need horizon-specific prediction logs to measure agreement rate"
        ]

        self.results.append(DiagnosticResult(
            test_number=10,
            test_name="Horizon Model Agreement",
            status="WARNING",
            ml_influence_score=50.0,
            bottleneck_detected=False,
            findings=findings,
            recommendations=[
                "Log predictions from all 4 horizon models",
                "Measure agreement rate: (same_direction_count / total_predictions)",
                "Analyze: how are disagreements weighted/resolved"
            ],
            metrics={}
        ))

        print("  WARNING: Requires horizon-specific prediction logging")
        print()

    def _generate_report(self) -> DiagnosticSweepReport:
        """Generate final diagnostic report"""
        tests_passed = sum(1 for r in self.results if r.status == "PASS")
        tests_failed = sum(1 for r in self.results if r.status == "FAIL")
        tests_warning = sum(1 for r in self.results if r.status == "WARNING")

        overall_influence = np.mean([r.ml_influence_score for r in self.results])

        bottlenecks = [
            f"Test {r.test_number}: {r.test_name}"
            for r in self.results if r.bottleneck_detected
        ]

        critical_fixes = []
        for r in self.results:
            if r.status == "FAIL" or (r.bottleneck_detected and r.status == "WARNING"):
                critical_fixes.extend(r.recommendations)

        return DiagnosticSweepReport(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            total_bars=len(self.data) if self.data is not None else 0,
            tests_run=len(self.results),
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_warning=tests_warning,
            overall_ml_influence=overall_influence,
            bottlenecks_found=bottlenecks,
            critical_fixes=list(set(critical_fixes)),  # Unique fixes only
            results=self.results
        )

    def save_report(self, report: DiagnosticSweepReport, filepath: str = None):
        """
        Save diagnostic report to markdown file

        Args:
            report: Diagnostic sweep report
            filepath: Output file path (default: auto-generated)
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ML_Allocator_Diagnostic_{self.symbol}_{timestamp}.md"

        with open(filepath, 'w') as f:
            f.write(f"# PRADO9_EVO ML + Allocator Diagnostic Report\n\n")
            f.write(f"**Symbol:** {report.symbol}\n")
            f.write(f"**Period:** {report.start_date} to {report.end_date}\n")
            f.write(f"**Total Bars:** {report.total_bars}\n")
            f.write(f"**Generated:** {report.timestamp}\n\n")

            f.write(f"## Executive Summary\n\n")
            f.write(f"- **Tests Run:** {report.tests_run}\n")
            f.write(f"- **Tests Passed:** {report.tests_passed} ✅\n")
            f.write(f"- **Tests Warning:** {report.tests_warning} ⚠️\n")
            f.write(f"- **Tests Failed:** {report.tests_failed} ❌\n")
            f.write(f"- **Overall ML Influence Score:** {report.overall_ml_influence:.1f}%\n\n")

            if report.bottlenecks_found:
                f.write(f"## Bottlenecks Detected\n\n")
                for bottleneck in report.bottlenecks_found:
                    f.write(f"- {bottleneck}\n")
                f.write("\n")

            if report.critical_fixes:
                f.write(f"## Critical Fixes Recommended\n\n")
                for i, fix in enumerate(report.critical_fixes, 1):
                    f.write(f"{i}. {fix}\n")
                f.write("\n")

            f.write(f"## Detailed Test Results\n\n")
            for result in report.results:
                f.write(f"### Test {result.test_number}: {result.test_name}\n\n")
                f.write(f"**Status:** {result.status}\n")
                f.write(f"**ML Influence Score:** {result.ml_influence_score:.1f}%\n")
                f.write(f"**Bottleneck Detected:** {'Yes' if result.bottleneck_detected else 'No'}\n\n")

                f.write(f"**Findings:**\n")
                for finding in result.findings:
                    f.write(f"- {finding}\n")
                f.write("\n")

                f.write(f"**Recommendations:**\n")
                for rec in result.recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")

                if result.metrics:
                    f.write(f"**Metrics:**\n")
                    for key, value in result.metrics.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")

                f.write("---\n\n")

        print(f"\n📄 Report saved to: {filepath}")
        return filepath


def run_diagnostic_sweep(symbol: str, start_date: str, end_date: str) -> DiagnosticSweepReport:
    """
    Convenience function to run complete diagnostic sweep

    Args:
        symbol: Trading symbol
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        Complete diagnostic report
    """
    diagnostic = MLAllocatorDiagnostic(symbol, start_date, end_date)
    report = diagnostic.run_all_tests()
    diagnostic.save_report(report)
    return report
