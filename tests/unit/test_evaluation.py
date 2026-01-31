"""Unit tests for evaluation module."""

import numpy as np
import pandas as pd

from genesis.evaluation.evaluator import QualityEvaluator
from genesis.evaluation.ml_utility import compute_ml_utility
from genesis.evaluation.privacy import (
    distance_to_closest_record,
    reidentification_risk,
)
from genesis.evaluation.report import QualityReport
from genesis.evaluation.statistical import (
    chi_squared_test,
    compute_statistical_fidelity,
    kolmogorov_smirnov_test,
)


class TestStatisticalMetrics:
    """Tests for statistical fidelity metrics."""

    def test_ks_test_identical(self):
        data = pd.Series(np.random.randn(100))
        result = kolmogorov_smirnov_test(data, data)

        assert result["statistic"] == 0.0
        assert result["score"] == 1.0

    def test_ks_test_different(self):
        real = pd.Series(np.random.randn(100))
        synthetic = pd.Series(np.random.randn(100) + 5)  # Shifted

        result = kolmogorov_smirnov_test(real, synthetic)
        assert result["statistic"] > 0.5  # Very different
        assert result["score"] < 0.5

    def test_chi_squared_identical(self):
        data = pd.Series(["A", "B", "C"] * 100)
        result = chi_squared_test(data, data)

        assert result["score"] > 0.9

    def test_compute_statistical_fidelity(self, sample_mixed_df):
        # Create slightly perturbed synthetic
        synthetic = sample_mixed_df.copy()
        synthetic["age"] = synthetic["age"] + np.random.randint(-2, 2, len(synthetic))

        result = compute_statistical_fidelity(
            sample_mixed_df,
            synthetic,
            discrete_columns=["gender", "city", "active"],
        )

        assert "column_metrics" in result
        assert "overall" in result
        assert 0 <= result["overall"]["fidelity_score"] <= 1


class TestMLUtility:
    """Tests for ML utility metrics."""

    def test_compute_ml_utility(self, sample_mixed_df):
        # Use slightly modified version as synthetic
        synthetic = sample_mixed_df.copy()
        synthetic["income"] = synthetic["income"] * 1.1

        result = compute_ml_utility(
            sample_mixed_df,
            synthetic,
            target_column="active",
        )

        assert "tstr" in result
        assert "trts" in result
        assert "utility_score" in result


class TestPrivacyMetrics:
    """Tests for privacy metrics."""

    def test_dcr_identical(self, sample_numeric_df):
        result = distance_to_closest_record(sample_numeric_df, sample_numeric_df)

        # Identical data should have DCR of 0
        assert result["min_dcr"] == 0.0

    def test_dcr_different(self, sample_numeric_df):
        # Create very different synthetic
        synthetic = sample_numeric_df.copy()
        synthetic = synthetic + 1000

        result = distance_to_closest_record(sample_numeric_df, synthetic)

        # Should have large DCR
        assert result["mean_dcr"] > 0

    def test_reidentification_risk(self, sample_mixed_df):
        result = reidentification_risk(sample_mixed_df, sample_mixed_df)

        # Identical data should have high risk
        assert result["reidentification_risk"] > 0


class TestQualityReport:
    """Tests for QualityReport."""

    def test_overall_score(self):
        report = QualityReport(
            statistical_fidelity={"overall": {"fidelity_score": 0.9}},
            ml_utility={"utility_score": 0.85},
            privacy_metrics={"overall_privacy_score": 0.95},
        )

        assert report.overall_score > 0
        assert report.fidelity_score == 0.9
        assert report.utility_score == 0.85
        assert report.privacy_score == 0.95

    def test_summary(self):
        report = QualityReport(
            statistical_fidelity={"overall": {"fidelity_score": 0.9}},
            ml_utility={"utility_score": 0.85},
            privacy_metrics={"overall_privacy_score": 0.95},
        )

        summary = report.summary()
        assert "Overall Score" in summary
        assert "Statistical Fidelity" in summary

    def test_to_dict(self):
        report = QualityReport()
        d = report.to_dict()

        assert "overall_score" in d
        assert "fidelity_score" in d
        assert "utility_score" in d
        assert "privacy_score" in d


class TestQualityEvaluator:
    """Tests for QualityEvaluator."""

    def test_evaluate(self, sample_mixed_df):
        # Create synthetic with slight modifications
        synthetic = sample_mixed_df.copy()
        synthetic["income"] = synthetic["income"] * 1.05

        evaluator = QualityEvaluator(sample_mixed_df, synthetic)
        report = evaluator.evaluate(target_column="active")

        assert isinstance(report, QualityReport)
        assert report.overall_score >= 0
