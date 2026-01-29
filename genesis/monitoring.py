"""Drift detection and alerting for synthetic data quality monitoring.

This module provides capabilities to:
- Detect when generator quality degrades over time
- Alert on data drift between training and current data
- Trigger automated retraining when thresholds are exceeded

Example:
    >>> from genesis.monitoring import DriftDetector, DriftAlert
    >>>
    >>> detector = DriftDetector(baseline_data=original_data)
    >>> detector.set_generator(trained_generator)
    >>>
    >>> # Monitor for drift
    >>> result = detector.check(current_data)
    >>> if result.has_drift:
    ...     print(f"Drift detected: {result.drift_score}")
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""

    DATA_DRIFT = "data_drift"  # Distribution shift in input data
    CONCEPT_DRIFT = "concept_drift"  # Relationship changes
    QUALITY_DRIFT = "quality_drift"  # Generator quality degradation
    SCHEMA_DRIFT = "schema_drift"  # Column additions/removals


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DriftMetric:
    """Individual drift metric result."""

    name: str
    column: Optional[str]
    value: float
    threshold: float
    has_drift: bool
    drift_type: DriftType
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "column": self.column,
            "value": self.value,
            "threshold": self.threshold,
            "has_drift": self.has_drift,
            "drift_type": self.drift_type.value,
            "description": self.description,
        }


@dataclass
class DriftResult:
    """Result of drift detection."""

    has_drift: bool
    drift_score: float
    metrics: List[DriftMetric]
    timestamp: str
    baseline_hash: str
    current_hash: str
    columns_with_drift: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_drift": self.has_drift,
            "drift_score": self.drift_score,
            "metrics": [m.to_dict() for m in self.metrics],
            "timestamp": self.timestamp,
            "baseline_hash": self.baseline_hash,
            "current_hash": self.current_hash,
            "columns_with_drift": self.columns_with_drift,
            "summary": self.summary,
        }


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""

    alert_id: str
    timestamp: str
    severity: AlertSeverity
    drift_type: DriftType
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "drift_type": self.drift_type.value,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
        }


class AlertHandler:
    """Base class for alert handlers."""

    def handle(self, alert: DriftAlert) -> None:
        """Handle an alert. Override in subclasses."""
        raise NotImplementedError


class LoggingAlertHandler(AlertHandler):
    """Handler that logs alerts."""

    def handle(self, alert: DriftAlert) -> None:
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.info)

        log_func(f"[{alert.severity.value.upper()}] {alert.message}")


class CallbackAlertHandler(AlertHandler):
    """Handler that calls a custom callback function."""

    def __init__(self, callback: Callable[[DriftAlert], None]):
        self.callback = callback

    def handle(self, alert: DriftAlert) -> None:
        self.callback(alert)


class WebhookAlertHandler(AlertHandler):
    """Handler that sends alerts to a webhook URL."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}

    def handle(self, alert: DriftAlert) -> None:
        try:
            import json
            import urllib.request

            data = json.dumps(alert.to_dict()).encode()
            req = urllib.request.Request(
                self.url,
                data=data,
                headers=self.headers,
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            logger.debug(f"Alert sent to webhook: {self.url}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


class DriftDetector:
    """Detects drift between baseline and current data/quality.

    Supports multiple drift detection methods:
    - Statistical tests (KS, Chi-squared) for distribution drift
    - Correlation comparison for relationship drift
    - Quality metric tracking for generator degradation
    """

    def __init__(
        self,
        baseline_data: Optional[pd.DataFrame] = None,
        ks_threshold: float = 0.1,
        correlation_threshold: float = 0.15,
        quality_threshold: float = 0.1,
        window_size: int = 1000,
    ):
        """Initialize drift detector.

        Args:
            baseline_data: Reference data for comparison
            ks_threshold: KS statistic threshold for drift
            correlation_threshold: Correlation change threshold
            quality_threshold: Quality score drop threshold
            window_size: Number of samples for rolling comparisons
        """
        self.baseline_data = baseline_data
        self.ks_threshold = ks_threshold
        self.correlation_threshold = correlation_threshold
        self.quality_threshold = quality_threshold
        self.window_size = window_size

        self._generator = None
        self._baseline_stats: Optional[Dict[str, Any]] = None
        self._baseline_quality: Optional[float] = None
        self._alert_handlers: List[AlertHandler] = [LoggingAlertHandler()]
        self._history: List[DriftResult] = []

        if baseline_data is not None:
            self._compute_baseline_stats()

    def set_baseline(self, data: pd.DataFrame) -> None:
        """Set baseline data for comparison."""
        self.baseline_data = data
        self._compute_baseline_stats()

    def set_generator(self, generator: Any) -> None:
        """Set generator for quality monitoring."""
        self._generator = generator

    def add_alert_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self._alert_handlers.append(handler)

    def clear_alert_handlers(self) -> None:
        """Remove all alert handlers."""
        self._alert_handlers.clear()

    def check(
        self,
        current_data: pd.DataFrame,
        check_quality: bool = True,
    ) -> DriftResult:
        """Check for drift between baseline and current data.

        Args:
            current_data: Current data to compare against baseline
            check_quality: Whether to also check generator quality

        Returns:
            DriftResult with drift detection results
        """
        if self.baseline_data is None:
            raise ValueError("Baseline data not set. Call set_baseline() first.")

        metrics = []
        columns_with_drift = []

        # Check schema drift
        schema_metrics = self._check_schema_drift(current_data)
        metrics.extend(schema_metrics)

        # Check data drift (distribution shift)
        data_metrics = self._check_data_drift(current_data)
        metrics.extend(data_metrics)
        for m in data_metrics:
            if m.has_drift and m.column:
                columns_with_drift.append(m.column)

        # Check correlation drift
        corr_metrics = self._check_correlation_drift(current_data)
        metrics.extend(corr_metrics)

        # Check quality drift if generator is set
        if check_quality and self._generator is not None:
            quality_metrics = self._check_quality_drift(current_data)
            metrics.extend(quality_metrics)

        # Compute overall drift score
        drift_values = [m.value for m in metrics if m.has_drift]
        drift_score = np.mean(drift_values) if drift_values else 0.0
        has_drift = any(m.has_drift for m in metrics)

        # Create result
        import hashlib

        baseline_hash = hashlib.md5(self.baseline_data.to_csv(index=False).encode()).hexdigest()[:8]
        current_hash = hashlib.md5(current_data.to_csv(index=False).encode()).hexdigest()[:8]

        result = DriftResult(
            has_drift=has_drift,
            drift_score=drift_score,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            baseline_hash=baseline_hash,
            current_hash=current_hash,
            columns_with_drift=list(set(columns_with_drift)),
            summary=self._generate_summary(metrics, has_drift),
        )

        # Store in history
        self._history.append(result)

        # Generate alerts if drift detected
        if has_drift:
            self._generate_alerts(result)

        return result

    def check_quality_only(self) -> DriftResult:
        """Check only generator quality drift without new data."""
        if self._generator is None:
            raise ValueError("Generator not set. Call set_generator() first.")

        metrics = self._check_quality_drift(self.baseline_data)

        drift_score = np.mean([m.value for m in metrics]) if metrics else 0.0
        has_drift = any(m.has_drift for m in metrics)

        result = DriftResult(
            has_drift=has_drift,
            drift_score=drift_score,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            baseline_hash="",
            current_hash="",
            summary=self._generate_summary(metrics, has_drift),
        )

        self._history.append(result)

        if has_drift:
            self._generate_alerts(result)

        return result

    def get_history(self, limit: Optional[int] = None) -> List[DriftResult]:
        """Get drift detection history."""
        if limit:
            return self._history[-limit:]
        return self._history

    def needs_retraining(self) -> Tuple[bool, str]:
        """Determine if generator needs retraining based on history.

        Returns:
            Tuple of (needs_retraining, reason)
        """
        if not self._history:
            return False, "No drift history available"

        recent = self._history[-5:] if len(self._history) >= 5 else self._history
        drift_count = sum(1 for r in recent if r.has_drift)

        if drift_count >= 3:
            return True, f"Drift detected in {drift_count}/{len(recent)} recent checks"

        avg_score = np.mean([r.drift_score for r in recent])
        if avg_score > 0.2:
            return True, f"Average drift score ({avg_score:.3f}) exceeds threshold"

        return False, "Generator quality is stable"

    def _compute_baseline_stats(self) -> None:
        """Compute statistics for baseline data."""
        self._baseline_stats = {
            "columns": list(self.baseline_data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.baseline_data.dtypes.items()},
            "n_rows": len(self.baseline_data),
            "column_stats": {},
        }

        for col in self.baseline_data.columns:
            col_data = self.baseline_data[col]
            if pd.api.types.is_numeric_dtype(col_data):
                self._baseline_stats["column_stats"][col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                }
            else:
                value_counts = col_data.value_counts(normalize=True)
                self._baseline_stats["column_stats"][col] = {
                    "categories": list(value_counts.index),
                    "frequencies": list(value_counts.values),
                }

        # Compute correlation matrix for numeric columns
        numeric_cols = self.baseline_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            self._baseline_stats["correlation"] = self.baseline_data[numeric_cols].corr().values
        else:
            self._baseline_stats["correlation"] = None

    def _check_schema_drift(self, current: pd.DataFrame) -> List[DriftMetric]:
        """Check for schema changes."""
        metrics = []

        baseline_cols = set(self._baseline_stats["columns"])
        current_cols = set(current.columns)

        added = current_cols - baseline_cols
        removed = baseline_cols - current_cols

        if added or removed:
            metrics.append(
                DriftMetric(
                    name="schema_change",
                    column=None,
                    value=len(added) + len(removed),
                    threshold=0,
                    has_drift=True,
                    drift_type=DriftType.SCHEMA_DRIFT,
                    description=f"Added: {added}, Removed: {removed}",
                )
            )

        return metrics

    def _check_data_drift(self, current: pd.DataFrame) -> List[DriftMetric]:
        """Check for distribution drift using KS test."""
        metrics = []

        common_cols = set(self.baseline_data.columns) & set(current.columns)

        for col in common_cols:
            baseline_col = self.baseline_data[col]
            current_col = current[col]

            if pd.api.types.is_numeric_dtype(baseline_col):
                # KS test for numeric columns
                try:
                    statistic, _ = stats.ks_2samp(
                        baseline_col.dropna(),
                        current_col.dropna(),
                    )
                    has_drift = statistic > self.ks_threshold

                    metrics.append(
                        DriftMetric(
                            name="ks_statistic",
                            column=col,
                            value=statistic,
                            threshold=self.ks_threshold,
                            has_drift=has_drift,
                            drift_type=DriftType.DATA_DRIFT,
                            description=f"KS statistic for {col}",
                        )
                    )
                except Exception:
                    pass
            else:
                # Chi-squared test for categorical columns
                try:
                    baseline_counts = baseline_col.value_counts()
                    current_counts = current_col.value_counts()

                    all_cats = set(baseline_counts.index) | set(current_counts.index)
                    baseline_freq = [baseline_counts.get(c, 0) for c in all_cats]
                    current_freq = [current_counts.get(c, 0) for c in all_cats]

                    # Normalize
                    baseline_freq = np.array(baseline_freq) / sum(baseline_freq)
                    current_freq = np.array(current_freq) / sum(current_freq)

                    # Jensen-Shannon divergence
                    m = (baseline_freq + current_freq) / 2
                    js_div = 0.5 * (
                        stats.entropy(baseline_freq + 1e-10, m + 1e-10)
                        + stats.entropy(current_freq + 1e-10, m + 1e-10)
                    )

                    has_drift = js_div > self.ks_threshold

                    metrics.append(
                        DriftMetric(
                            name="js_divergence",
                            column=col,
                            value=js_div,
                            threshold=self.ks_threshold,
                            has_drift=has_drift,
                            drift_type=DriftType.DATA_DRIFT,
                            description=f"JS divergence for {col}",
                        )
                    )
                except Exception:
                    pass

        return metrics

    def _check_correlation_drift(self, current: pd.DataFrame) -> List[DriftMetric]:
        """Check for correlation structure changes."""
        metrics = []

        if self._baseline_stats.get("correlation") is None:
            return metrics

        numeric_cols = current.select_dtypes(include=[np.number]).columns
        baseline_cols = [c for c in self._baseline_stats["columns"] if c in numeric_cols]

        if len(baseline_cols) < 2:
            return metrics

        try:
            current_corr = current[baseline_cols].corr().values
            baseline_corr = self._baseline_stats["correlation"]

            # Compute Frobenius norm of difference
            if current_corr.shape == baseline_corr.shape:
                diff = np.abs(current_corr - baseline_corr)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                has_drift = max_diff > self.correlation_threshold

                metrics.append(
                    DriftMetric(
                        name="correlation_max_diff",
                        column=None,
                        value=max_diff,
                        threshold=self.correlation_threshold,
                        has_drift=has_drift,
                        drift_type=DriftType.CONCEPT_DRIFT,
                        description="Maximum correlation matrix difference",
                    )
                )

                metrics.append(
                    DriftMetric(
                        name="correlation_mean_diff",
                        column=None,
                        value=mean_diff,
                        threshold=self.correlation_threshold / 2,
                        has_drift=mean_diff > self.correlation_threshold / 2,
                        drift_type=DriftType.CONCEPT_DRIFT,
                        description="Mean correlation matrix difference",
                    )
                )
        except Exception:
            pass

        return metrics

    def _check_quality_drift(self, reference_data: pd.DataFrame) -> List[DriftMetric]:
        """Check for generator quality degradation."""
        metrics = []

        if self._generator is None:
            return metrics

        try:
            from genesis.evaluation.statistical import compute_statistical_fidelity

            # Generate synthetic data
            synthetic = self._generator.generate(min(len(reference_data), 1000))

            # Compute quality
            result = compute_statistical_fidelity(reference_data, synthetic)
            current_quality = result.overall_score

            # Compare to baseline quality
            if self._baseline_quality is None:
                self._baseline_quality = current_quality

            quality_drop = self._baseline_quality - current_quality
            has_drift = quality_drop > self.quality_threshold

            metrics.append(
                DriftMetric(
                    name="quality_score",
                    column=None,
                    value=current_quality,
                    threshold=self._baseline_quality - self.quality_threshold,
                    has_drift=has_drift,
                    drift_type=DriftType.QUALITY_DRIFT,
                    description=f"Current quality: {current_quality:.3f}, baseline: {self._baseline_quality:.3f}",
                )
            )

        except Exception as e:
            logger.warning(f"Failed to check quality drift: {e}")

        return metrics

    def _generate_summary(self, metrics: List[DriftMetric], has_drift: bool) -> str:
        """Generate human-readable summary."""
        if not has_drift:
            return "No significant drift detected"

        drift_types = {m.drift_type for m in metrics if m.has_drift}
        n_drifted = sum(1 for m in metrics if m.has_drift)

        type_str = ", ".join(t.value for t in drift_types)
        return f"Drift detected: {n_drifted} metrics affected ({type_str})"

    def _generate_alerts(self, result: DriftResult) -> None:
        """Generate and dispatch alerts."""
        import uuid

        # Determine severity
        if result.drift_score > 0.5:
            severity = AlertSeverity.CRITICAL
        elif result.drift_score > 0.3:
            severity = AlertSeverity.ERROR
        elif result.drift_score > 0.1:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        # Determine primary drift type
        drift_types = [m.drift_type for m in result.metrics if m.has_drift]
        primary_type = drift_types[0] if drift_types else DriftType.DATA_DRIFT

        alert = DriftAlert(
            alert_id=str(uuid.uuid4())[:8],
            timestamp=result.timestamp,
            severity=severity,
            drift_type=primary_type,
            message=result.summary,
            details={
                "drift_score": result.drift_score,
                "columns_affected": result.columns_with_drift,
                "n_metrics_drifted": sum(1 for m in result.metrics if m.has_drift),
            },
        )

        for handler in self._alert_handlers:
            try:
                handler.handle(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")


class DriftMonitor:
    """Continuous drift monitoring with scheduling support.

    Example:
        >>> monitor = DriftMonitor(detector)
        >>> monitor.start(interval_seconds=3600)  # Check every hour
    """

    def __init__(
        self,
        detector: DriftDetector,
        data_source: Optional[Callable[[], pd.DataFrame]] = None,
    ):
        """Initialize monitor.

        Args:
            detector: Drift detector to use
            data_source: Callable that returns current data
        """
        self.detector = detector
        self.data_source = data_source
        self._running = False
        self._thread = None

    def check_once(self) -> DriftResult:
        """Run a single drift check."""
        if self.data_source is None:
            raise ValueError("Data source not set")

        current_data = self.data_source()
        return self.detector.check(current_data)

    def start(
        self,
        interval_seconds: int = 3600,
        callback: Optional[Callable[[DriftResult], None]] = None,
    ) -> None:
        """Start continuous monitoring in a background thread.

        Args:
            interval_seconds: Seconds between checks
            callback: Optional callback for each result
        """
        import threading

        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True

        def monitor_loop():
            while self._running:
                try:
                    result = self.check_once()
                    if callback:
                        callback(result)
                except Exception as e:
                    logger.error(f"Drift check failed: {e}")

                time.sleep(interval_seconds)

        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Drift monitor started (interval: {interval_seconds}s)")

    def stop(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Drift monitor stopped")


__all__ = [
    # Core classes
    "DriftDetector",
    "DriftMonitor",
    "DriftResult",
    "DriftMetric",
    "DriftAlert",
    # Types
    "DriftType",
    "AlertSeverity",
    # Handlers
    "AlertHandler",
    "LoggingAlertHandler",
    "CallbackAlertHandler",
    "WebhookAlertHandler",
]
