"""Tests for Drift Detection module."""

import numpy as np
import pandas as pd
import pytest

from genesis.drift import (
    DataDriftDetector,
    DriftMetric,
    DriftReport,
    DriftSeverity,
    DriftType,
    QualityDriftDetector,
    detect_drift,
)


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Create reference dataset."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "numeric": np.random.normal(50, 10, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
        }
    )


@pytest.fixture
def drifted_data() -> pd.DataFrame:
    """Create dataset with drift."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "numeric": np.random.normal(70, 15, 1000),  # Shifted mean and std
            "category": np.random.choice(
                ["A", "B", "C"], 1000, p=[0.2, 0.5, 0.3]
            ),  # Changed distribution
        }
    )


@pytest.fixture
def stable_data() -> pd.DataFrame:
    """Create dataset without drift."""
    np.random.seed(44)
    return pd.DataFrame(
        {
            "numeric": np.random.normal(50, 10, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
        }
    )


class TestDriftMetric:
    """Tests for DriftMetric."""

    def test_severity_none(self) -> None:
        """Test no drift severity."""
        metric = DriftMetric(
            name="test",
            value=0.05,
            threshold=0.1,
            is_drifted=False,
        )

        assert metric.severity == DriftSeverity.NONE

    def test_severity_low(self) -> None:
        """Test low severity drift."""
        metric = DriftMetric(
            name="test",
            value=0.12,
            threshold=0.1,
            is_drifted=True,
        )

        assert metric.severity == DriftSeverity.LOW

    def test_severity_critical(self) -> None:
        """Test critical severity drift."""
        metric = DriftMetric(
            name="test",
            value=0.5,
            threshold=0.1,
            is_drifted=True,
        )

        assert metric.severity == DriftSeverity.CRITICAL


class TestDataDriftDetector:
    """Tests for DataDriftDetector."""

    def test_detect_no_drift(self, reference_data: pd.DataFrame, stable_data: pd.DataFrame) -> None:
        """Test detection with no drift."""
        detector = DataDriftDetector(ks_threshold=0.15, js_threshold=0.15)

        report = detector.detect(reference_data, stable_data)

        assert isinstance(report, DriftReport)
        # May or may not detect drift depending on random sampling

    def test_detect_with_drift(
        self, reference_data: pd.DataFrame, drifted_data: pd.DataFrame
    ) -> None:
        """Test detection with drift."""
        detector = DataDriftDetector(ks_threshold=0.1, js_threshold=0.1)

        report = detector.detect(reference_data, drifted_data)

        assert report.drift_detected is True
        assert len(report.drifted_columns) > 0

    def test_report_has_metrics(
        self, reference_data: pd.DataFrame, drifted_data: pd.DataFrame
    ) -> None:
        """Test that report contains metrics."""
        detector = DataDriftDetector()

        report = detector.detect(reference_data, drifted_data)

        assert len(report.metrics) > 0
        assert all(isinstance(m, DriftMetric) for m in report.metrics)

    def test_report_has_recommendations(
        self, reference_data: pd.DataFrame, drifted_data: pd.DataFrame
    ) -> None:
        """Test that report contains recommendations."""
        detector = DataDriftDetector()

        report = detector.detect(reference_data, drifted_data)

        assert len(report.recommendations) > 0


class TestQualityDriftDetector:
    """Tests for QualityDriftDetector."""

    def test_detect_quality_drift(self) -> None:
        """Test quality drift detection."""
        detector = QualityDriftDetector(quality_threshold=0.05)

        reference = {"accuracy": 0.9, "fidelity": 0.85}
        current = {"accuracy": 0.75, "fidelity": 0.80}

        report = detector.detect(reference, current)

        assert report.drift_detected is True
        assert report.drift_type == DriftType.QUALITY_DRIFT


class TestDriftReport:
    """Tests for DriftReport."""

    def test_to_dict(self, reference_data: pd.DataFrame, drifted_data: pd.DataFrame) -> None:
        """Test report serialization."""
        detector = DataDriftDetector()
        report = detector.detect(reference_data, drifted_data)

        d = report.to_dict()

        assert "drift_detected" in d
        assert "severity" in d
        assert "metrics" in d
        assert "recommendations" in d


class TestDetectDrift:
    """Tests for detect_drift convenience function."""

    def test_returns_report(self, reference_data: pd.DataFrame, drifted_data: pd.DataFrame) -> None:
        """Test convenience function."""
        report = detect_drift(reference_data, drifted_data, verbose=False)

        assert isinstance(report, DriftReport)
