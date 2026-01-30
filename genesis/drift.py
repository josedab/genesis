"""Drift-Aware Synthetic Data Generation.

This module provides drift detection between datasets and adaptive regeneration
of synthetic data to keep it aligned with evolving real data distributions.

Features:
    - Statistical drift detection (KS test, JS divergence, PSI)
    - Column-level and dataset-level drift metrics
    - Drift-aware generators that adapt to distribution shifts
    - Continuous monitoring for streaming scenarios

Example:
    Quick drift detection::

        from genesis import detect_drift

        report = detect_drift(baseline_df, current_df)

        if report.has_significant_drift:
            print(f"Drift detected in: {report.drifted_columns}")
            print(f"Overall drift score: {report.overall_drift_score:.4f}")

    Drift-aware generation::

        from genesis.drift import DriftAwareGenerator

        generator = DriftAwareGenerator()
        generator.fit(baseline_df)

        # Generate data adapted to current distribution
        synthetic = generator.generate(
            n_samples=1000,
            target_distribution=current_df,
            drift_adaptation="weighted"
        )

Classes:
    DriftType: Types of drift (data, concept, schema, quality).
    DriftSeverity: Severity levels (none, low, medium, high, critical).
    DataDriftDetector: Compares two datasets for distribution shifts.
    DriftAwareGenerator: Generates data adapting to detected drift.
    ContinuousMonitor: Monitors data streams for drift.

Functions:
    detect_drift: Convenience function for drift detection.
    calculate_psi: Calculate Population Stability Index.

Note:
    PSI Interpretation:
    - < 0.10: No significant drift
    - 0.10-0.25: Moderate drift, investigate
    - > 0.25: Significant drift, action required
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class DriftType(str, Enum):
    """Types of data drift.

    Attributes:
        DATA_DRIFT: Feature distribution changes over time.
        CONCEPT_DRIFT: Relationship between features and target changes.
        SCHEMA_DRIFT: Column additions, removals, or type changes.
        QUALITY_DRIFT: Data quality degradation (missing values, outliers).
    """

    DATA_DRIFT = "data_drift"  # Feature distribution change
    CONCEPT_DRIFT = "concept_drift"  # Relationship change
    SCHEMA_DRIFT = "schema_drift"  # Column changes
    QUALITY_DRIFT = "quality_drift"  # Quality degradation


class DriftSeverity(str, Enum):
    """Severity levels for detected drift.

    Attributes:
        NONE: No drift detected.
        LOW: Minor drift, monitor but no action needed.
        MEDIUM: Moderate drift, investigate cause.
        HIGH: Significant drift, consider regeneration.
        CRITICAL: Severe drift, immediate action required.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftMetric:
    """A single drift metric."""

    name: str
    value: float
    threshold: float
    is_drifted: bool
    column: Optional[str] = None

    @property
    def severity(self) -> DriftSeverity:
        """Get severity based on how much threshold is exceeded."""
        if not self.is_drifted:
            return DriftSeverity.NONE

        ratio = self.value / self.threshold
        if ratio < 1.5:
            return DriftSeverity.LOW
        elif ratio < 2.0:
            return DriftSeverity.MEDIUM
        elif ratio < 3.0:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL


@dataclass
class ColumnDriftResult:
    """Drift result for a single column."""

    column: str
    drift_score: float
    is_drifted: bool
    metric_name: str
    threshold: float


@dataclass
class DriftReport:
    """Report of drift detection results."""

    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    metrics: List[DriftMetric]
    drifted_columns: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recommendations: List[str] = field(default_factory=list)

    @property
    def overall_drift_score(self) -> float:
        """Calculate overall drift score as average of all metric values."""
        if not self.metrics:
            return 0.0
        return sum(m.value for m in self.metrics) / len(self.metrics)

    @property
    def has_significant_drift(self) -> bool:
        """Check if there is significant drift (medium or higher severity)."""
        return self.severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]

    @property
    def column_results(self) -> Dict[str, ColumnDriftResult]:
        """Get drift results indexed by column name."""
        results = {}
        for m in self.metrics:
            if m.column:
                results[m.column] = ColumnDriftResult(
                    column=m.column,
                    drift_score=m.value,
                    is_drifted=m.is_drifted,
                    metric_name=m.name,
                    threshold=m.threshold,
                )
        return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "overall_drift_score": self.overall_drift_score,
            "has_significant_drift": self.has_significant_drift,
            "drifted_columns": self.drifted_columns,
            "timestamp": self.timestamp,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "is_drifted": m.is_drifted,
                    "column": m.column,
                }
                for m in self.metrics
            ],
            "recommendations": self.recommendations,
        }

    def to_json(self, path: str) -> None:
        """Save report as JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_html(self, path: str) -> None:
        """Save report as HTML file."""
        severity_class = {
            DriftSeverity.NONE: "none",
            DriftSeverity.LOW: "low",
            DriftSeverity.MEDIUM: "medium",
            DriftSeverity.HIGH: "high",
            DriftSeverity.CRITICAL: "critical",
        }
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Drift Detection Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .severity-none {{ color: green; }}
        .severity-low {{ color: yellowgreen; }}
        .severity-medium {{ color: orange; }}
        .severity-high {{ color: red; }}
        .severity-critical {{ color: darkred; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Drift Detection Report</h1>
    <p><strong>Drift Detected:</strong> {"Yes" if self.drift_detected else "No"}</p>
    <p><strong>Overall Drift Score:</strong> {self.overall_drift_score:.4f}</p>
    <p><strong>Severity:</strong> <span class="severity-{severity_class[self.severity]}">{self.severity.value}</span></p>
    <p><strong>Drift Type:</strong> {self.drift_type.value}</p>
    <p><strong>Timestamp:</strong> {self.timestamp}</p>
    <h2>Drifted Columns ({len(self.drifted_columns)})</h2>
    <ul>{"".join(f"<li>{col}</li>" for col in self.drifted_columns) if self.drifted_columns else "<li>None</li>"}</ul>
    <h2>Metrics</h2>
    <table>
        <tr><th>Column</th><th>Metric</th><th>Value</th><th>Threshold</th><th>Drifted</th></tr>
        {"".join(f'<tr><td>{m.column or "N/A"}</td><td>{m.name}</td><td>{m.value:.4f}</td><td>{m.threshold:.4f}</td><td>{"Yes" if m.is_drifted else "No"}</td></tr>' for m in self.metrics)}
    </table>
    <h2>Recommendations</h2>
    <ul>{"".join(f"<li>{rec}</li>" for rec in self.recommendations) if self.recommendations else "<li>No recommendations</li>"}</ul>
</body>
</html>"""
        with open(path, "w") as f:
            f.write(html)


class DataDriftDetector:
    """Detect data drift between datasets."""

    def __init__(
        self,
        ks_threshold: float = 0.1,
        js_threshold: float = 0.1,
        psi_threshold: float = 0.2,
    ):
        """Initialize detector.

        Args:
            ks_threshold: KS statistic threshold for numeric drift
            js_threshold: JS divergence threshold for categorical drift
            psi_threshold: PSI threshold for overall drift
        """
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold
        self.psi_threshold = psi_threshold

    def detect(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> DriftReport:
        """Detect drift between reference and current data.

        Args:
            reference_data: Baseline/reference dataset
            current_data: Current dataset to compare

        Returns:
            DriftReport with detection results
        """
        metrics = []
        drifted_columns = []

        # Get common columns
        common_cols = set(reference_data.columns) & set(current_data.columns)

        # Check for schema drift
        new_cols = set(current_data.columns) - set(reference_data.columns)
        removed_cols = set(reference_data.columns) - set(current_data.columns)

        if new_cols or removed_cols:
            metrics.append(
                DriftMetric(
                    name="schema_change",
                    value=len(new_cols) + len(removed_cols),
                    threshold=0,
                    is_drifted=True,
                )
            )

        # Check each column
        for col in common_cols:
            ref_col = reference_data[col].dropna()
            cur_col = current_data[col].dropna()

            if len(ref_col) == 0 or len(cur_col) == 0:
                continue

            if pd.api.types.is_numeric_dtype(ref_col):
                metric = self._detect_numeric_drift(col, ref_col, cur_col)
            else:
                metric = self._detect_categorical_drift(col, ref_col, cur_col)

            metrics.append(metric)
            if metric.is_drifted:
                drifted_columns.append(col)

        # Compute overall drift
        drift_detected = len(drifted_columns) > 0

        # Determine severity
        if not drift_detected:
            severity = DriftSeverity.NONE
        else:
            severities = [m.severity for m in metrics if m.is_drifted]
            if DriftSeverity.CRITICAL in severities:
                severity = DriftSeverity.CRITICAL
            elif DriftSeverity.HIGH in severities:
                severity = DriftSeverity.HIGH
            elif len(drifted_columns) > len(common_cols) * 0.3:
                severity = DriftSeverity.HIGH
            elif len(drifted_columns) > len(common_cols) * 0.1:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW

        # Generate recommendations
        recommendations = self._generate_recommendations(drift_detected, severity, drifted_columns)

        return DriftReport(
            drift_detected=drift_detected,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            metrics=metrics,
            drifted_columns=drifted_columns,
            recommendations=recommendations,
        )

    def _detect_numeric_drift(self, col: str, ref: pd.Series, cur: pd.Series) -> DriftMetric:
        """Detect drift in numeric column using KS test."""
        try:
            ks_stat, _ = stats.ks_2samp(ref, cur)
        except Exception:
            ks_stat = 0.0

        return DriftMetric(
            name="ks_statistic",
            value=ks_stat,
            threshold=self.ks_threshold,
            is_drifted=ks_stat > self.ks_threshold,
            column=col,
        )

    def _detect_categorical_drift(self, col: str, ref: pd.Series, cur: pd.Series) -> DriftMetric:
        """Detect drift in categorical column using JS divergence."""
        # Get value counts
        ref_counts = ref.value_counts(normalize=True)
        cur_counts = cur.value_counts(normalize=True)

        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        ref_probs = np.array([ref_counts.get(c, 1e-10) for c in all_categories])
        cur_probs = np.array([cur_counts.get(c, 1e-10) for c in all_categories])

        # Normalize
        ref_probs = ref_probs / ref_probs.sum()
        cur_probs = cur_probs / cur_probs.sum()

        # Compute JS divergence
        m = (ref_probs + cur_probs) / 2
        js_div = 0.5 * (
            np.sum(ref_probs * np.log(ref_probs / m + 1e-10))
            + np.sum(cur_probs * np.log(cur_probs / m + 1e-10))
        )

        return DriftMetric(
            name="js_divergence",
            value=js_div,
            threshold=self.js_threshold,
            is_drifted=js_div > self.js_threshold,
            column=col,
        )

    def _generate_recommendations(
        self,
        drift_detected: bool,
        severity: DriftSeverity,
        drifted_columns: List[str],
    ) -> List[str]:
        """Generate recommendations based on drift."""
        if not drift_detected:
            return ["No action needed - data is stable"]

        recs = []

        if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recs.append("Immediate regeneration recommended")
            recs.append("Review data pipeline for potential issues")
        elif severity == DriftSeverity.MEDIUM:
            recs.append("Consider regenerating synthetic data soon")
            recs.append("Monitor drift trends")
        else:
            recs.append("Continue monitoring, regenerate at next scheduled interval")

        if len(drifted_columns) <= 3:
            recs.append(f"Focus on drifted columns: {', '.join(drifted_columns)}")
        else:
            recs.append(f"Multiple columns drifted ({len(drifted_columns)} total)")

        return recs


class QualityDriftDetector:
    """Detect quality drift in synthetic data."""

    def __init__(
        self,
        quality_threshold: float = 0.1,
        utility_threshold: float = 0.1,
    ):
        """Initialize detector.

        Args:
            quality_threshold: Threshold for quality score change
            utility_threshold: Threshold for ML utility change
        """
        self.quality_threshold = quality_threshold
        self.utility_threshold = utility_threshold

    def detect(
        self,
        reference_quality: Dict[str, float],
        current_quality: Dict[str, float],
    ) -> DriftReport:
        """Detect quality drift.

        Args:
            reference_quality: Baseline quality metrics
            current_quality: Current quality metrics

        Returns:
            DriftReport
        """
        metrics = []
        drifted_metrics = []

        for metric_name in reference_quality:
            if metric_name in current_quality:
                ref_val = reference_quality[metric_name]
                cur_val = current_quality[metric_name]

                change = abs(cur_val - ref_val)
                threshold = self.quality_threshold

                is_drifted = change > threshold

                metrics.append(
                    DriftMetric(
                        name=metric_name,
                        value=change,
                        threshold=threshold,
                        is_drifted=is_drifted,
                    )
                )

                if is_drifted:
                    drifted_metrics.append(metric_name)

        drift_detected = len(drifted_metrics) > 0

        if not drift_detected:
            severity = DriftSeverity.NONE
        elif len(drifted_metrics) > len(reference_quality) * 0.5:
            severity = DriftSeverity.HIGH
        elif len(drifted_metrics) > len(reference_quality) * 0.2:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        return DriftReport(
            drift_detected=drift_detected,
            drift_type=DriftType.QUALITY_DRIFT,
            severity=severity,
            metrics=metrics,
            drifted_columns=drifted_metrics,
            recommendations=[
                "Retrain generator on updated data" if drift_detected else "Quality stable"
            ],
        )


class DriftAwareGenerator:
    """Generator that monitors drift and regenerates when needed."""

    def __init__(
        self,
        generator,
        drift_detector: Optional[DataDriftDetector] = None,
        auto_regenerate: bool = True,
        regenerate_threshold: DriftSeverity = DriftSeverity.MEDIUM,
    ):
        """Initialize drift-aware generator.

        Args:
            generator: Base synthetic data generator
            drift_detector: Drift detector instance
            auto_regenerate: Automatically regenerate on drift
            regenerate_threshold: Minimum severity to trigger regeneration
        """
        self.generator = generator
        self.drift_detector = drift_detector or DataDriftDetector()
        self.auto_regenerate = auto_regenerate
        self.regenerate_threshold = regenerate_threshold

        self._reference_data: Optional[pd.DataFrame] = None
        self._current_synthetic: Optional[pd.DataFrame] = None
        self._drift_history: List[DriftReport] = []
        self._generation_count = 0

    def fit(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> "DriftAwareGenerator":
        """Fit generator and store reference data.

        Args:
            data: Training data
            **kwargs: Additional arguments for generator

        Returns:
            Self for chaining
        """
        self._reference_data = data.copy()
        self.generator.fit(data, **kwargs)
        return self

    def generate(
        self,
        n_samples: int,
        check_drift: bool = True,
        current_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data with drift checking.

        Args:
            n_samples: Number of samples to generate
            check_drift: Whether to check for drift
            current_data: Optional new real data to compare against

        Returns:
            Generated synthetic data
        """
        # Check drift if requested and current data provided
        if check_drift and current_data is not None:
            drift_report = self.check_drift(current_data)

            if self.auto_regenerate and self._should_regenerate(drift_report):
                print(f"Drift detected ({drift_report.severity.value}), regenerating...")
                self.fit(current_data)

        # Generate
        self._current_synthetic = self.generator.generate(n_samples)
        self._generation_count += 1

        return self._current_synthetic

    def check_drift(
        self,
        current_data: pd.DataFrame,
    ) -> DriftReport:
        """Check for drift between reference and current data.

        Args:
            current_data: Current real data

        Returns:
            DriftReport
        """
        if self._reference_data is None:
            raise RuntimeError("Must call fit() before check_drift()")

        report = self.drift_detector.detect(self._reference_data, current_data)
        self._drift_history.append(report)

        return report

    def _should_regenerate(self, report: DriftReport) -> bool:
        """Determine if regeneration is needed."""
        if not report.drift_detected:
            return False

        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]

        return severity_order.index(report.severity) >= severity_order.index(
            self.regenerate_threshold
        )

    def update_reference(self, new_reference: pd.DataFrame) -> None:
        """Update reference data without retraining.

        Args:
            new_reference: New reference data
        """
        self._reference_data = new_reference.copy()

    def get_drift_history(self) -> List[DriftReport]:
        """Get history of drift reports."""
        return self._drift_history

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift history."""
        if not self._drift_history:
            return {"n_checks": 0, "n_drifts": 0}

        n_drifts = sum(1 for r in self._drift_history if r.drift_detected)

        return {
            "n_checks": len(self._drift_history),
            "n_drifts": n_drifts,
            "drift_rate": n_drifts / len(self._drift_history),
            "last_drift": self._drift_history[-1].timestamp if n_drifts > 0 else None,
            "generation_count": self._generation_count,
        }


class ContinuousMonitor:
    """Monitor for continuous drift detection."""

    def __init__(
        self,
        generator,
        check_interval_samples: int = 1000,
        on_drift: Optional[Callable[[DriftReport], None]] = None,
    ):
        """Initialize continuous monitor.

        Args:
            generator: Generator to monitor
            check_interval_samples: Check drift every N samples
            on_drift: Callback when drift detected
        """
        self.generator = generator
        self.check_interval_samples = check_interval_samples
        self.on_drift = on_drift

        self._sample_buffer: List[pd.DataFrame] = []
        self._buffer_size = 0
        self._drift_detector = DataDriftDetector()
        self._reference_data: Optional[pd.DataFrame] = None

    def set_reference(self, data: pd.DataFrame) -> None:
        """Set reference data for comparison."""
        self._reference_data = data.copy()

    def ingest_samples(self, samples: pd.DataFrame) -> Optional[DriftReport]:
        """Ingest new samples and check for drift.

        Args:
            samples: New data samples

        Returns:
            DriftReport if drift check was triggered, None otherwise
        """
        self._sample_buffer.append(samples)
        self._buffer_size += len(samples)

        if self._buffer_size >= self.check_interval_samples:
            return self._check_and_reset()

        return None

    def _check_and_reset(self) -> DriftReport:
        """Check drift and reset buffer."""
        # Combine buffer
        current_data = pd.concat(self._sample_buffer, ignore_index=True)

        # Check drift
        if self._reference_data is not None:
            report = self._drift_detector.detect(self._reference_data, current_data)

            if report.drift_detected and self.on_drift:
                self.on_drift(report)
        else:
            report = DriftReport(
                drift_detected=False,
                drift_type=DriftType.DATA_DRIFT,
                severity=DriftSeverity.NONE,
                metrics=[],
                drifted_columns=[],
            )

        # Reset buffer
        self._sample_buffer = []
        self._buffer_size = 0

        return report


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    verbose: bool = True,
) -> DriftReport:
    """Convenience function to detect drift.

    Args:
        reference_data: Baseline data
        current_data: Current data
        verbose: Print results

    Returns:
        DriftReport
    """
    detector = DataDriftDetector()
    report = detector.detect(reference_data, current_data)

    if verbose:
        print(f"Drift Detected: {report.drift_detected}")
        print(f"Severity: {report.severity.value}")
        if report.drifted_columns:
            print(f"Drifted Columns: {', '.join(report.drifted_columns)}")
        for rec in report.recommendations:
            print(f"  • {rec}")

    return report
