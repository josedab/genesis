"""Tests for fairness-aware generation."""

import numpy as np
import pandas as pd
import pytest

from genesis.fairness import (
    BiasMetrics,
    BiasReport,
    CounterfactualGenerator,
    DemographicParityConstraint,
    FairnessAnalyzer,
    FairnessAudit,
    FairGenerator,
    balance_dataset,
)


@pytest.fixture
def biased_data() -> pd.DataFrame:
    """Create biased sample data."""
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "gender": np.random.choice(["male", "female"], n, p=[0.6, 0.4]),
        "age": np.random.randint(20, 60, n),
        "experience": np.random.randint(0, 20, n),
        "hired": 0,
    })

    # Add bias: males more likely to be hired
    male_mask = data["gender"] == "male"
    data.loc[male_mask, "hired"] = np.random.choice(
        [0, 1], male_mask.sum(), p=[0.3, 0.7]
    )
    data.loc[~male_mask, "hired"] = np.random.choice(
        [0, 1], (~male_mask).sum(), p=[0.6, 0.4]
    )

    return data


class TestBiasMetrics:
    """Tests for BiasMetrics dataclass."""

    def test_fair_when_above_threshold(self) -> None:
        """Test is_fair when above threshold."""
        metrics = BiasMetrics(
            metric_name="demographic_parity",
            value=0.85,
            privileged_value=0.7,
            unprivileged_value=0.595,
            threshold=0.8,
        )

        assert metrics.is_fair is True

    def test_unfair_when_below_threshold(self) -> None:
        """Test is_fair when below threshold."""
        metrics = BiasMetrics(
            metric_name="demographic_parity",
            value=0.5,
            privileged_value=0.8,
            unprivileged_value=0.4,
            threshold=0.8,
        )

        assert metrics.is_fair is False


class TestFairnessAnalyzer:
    """Tests for FairnessAnalyzer."""

    def test_analyze_detects_bias(self, biased_data: pd.DataFrame) -> None:
        """Test that analyzer detects bias in biased data."""
        analyzer = FairnessAnalyzer(
            sensitive_attributes=["gender"],
            outcome_column="hired",
            privileged_groups={"gender": "male"},
        )

        reports = analyzer.analyze(biased_data)

        assert "gender" in reports
        dp = reports["gender"].metrics["demographic_parity"]
        assert dp.value < 0.8  # Should detect bias

    def test_calculates_group_statistics(self, biased_data: pd.DataFrame) -> None:
        """Test group statistics calculation."""
        analyzer = FairnessAnalyzer(
            sensitive_attributes=["gender"],
            outcome_column="hired",
        )

        reports = analyzer.analyze(biased_data)

        assert "male" in reports["gender"].group_statistics
        assert "female" in reports["gender"].group_statistics
        assert "positive_rate" in reports["gender"].group_statistics["male"]

    def test_generates_recommendations(self, biased_data: pd.DataFrame) -> None:
        """Test recommendation generation."""
        analyzer = FairnessAnalyzer(
            sensitive_attributes=["gender"],
            outcome_column="hired",
            privileged_groups={"gender": "male"},
        )

        reports = analyzer.analyze(biased_data)

        assert len(reports["gender"].recommendations) > 0

    def test_handles_missing_attribute(self, biased_data: pd.DataFrame) -> None:
        """Test handling of missing sensitive attribute."""
        analyzer = FairnessAnalyzer(
            sensitive_attributes=["gender", "nonexistent"],
            outcome_column="hired",
        )

        reports = analyzer.analyze(biased_data)

        assert "gender" in reports
        assert "nonexistent" not in reports


class TestDemographicParityConstraint:
    """Tests for DemographicParityConstraint."""

    def test_fit_learns_group_rates(self, biased_data: pd.DataFrame) -> None:
        """Test that fit learns group rates."""
        constraint = DemographicParityConstraint(
            sensitive_attr="gender",
            outcome_col="hired",
        )

        constraint.fit(biased_data)

        assert "male" in constraint._group_rates
        assert "female" in constraint._group_rates

    def test_check_returns_false_for_unfair(self, biased_data: pd.DataFrame) -> None:
        """Test check returns False for unfair data."""
        constraint = DemographicParityConstraint(
            sensitive_attr="gender",
            outcome_col="hired",
            target_ratio=1.0,
            tolerance=0.1,
        )

        is_fair = constraint.check(biased_data)
        assert is_fair is False


class TestFairGenerator:
    """Tests for FairGenerator."""

    def test_resampling_strategy(self, biased_data: pd.DataFrame) -> None:
        """Test resampling strategy."""
        generator = FairGenerator(
            sensitive_attr="gender",
            outcome_col="hired",
            strategy="resampling",
        )

        fair_data = generator.generate(biased_data, n_samples=1000, random_state=42)

        assert len(fair_data) == 1000

        # Check improved balance
        analyzer = FairnessAnalyzer(
            sensitive_attributes=["gender"],
            outcome_column="hired",
            privileged_groups={"gender": "male"},
        )

        reports = analyzer.analyze(fair_data)
        dp = reports["gender"].metrics["demographic_parity"]

        # Should be more fair
        assert dp.value > 0.8

    def test_reweighting_strategy(self, biased_data: pd.DataFrame) -> None:
        """Test reweighting strategy."""
        generator = FairGenerator(
            sensitive_attr="gender",
            strategy="reweighting",
        )

        fair_data = generator.generate(biased_data, n_samples=1000, random_state=42)

        assert len(fair_data) == 1000

    def test_counterfactual_strategy(self, biased_data: pd.DataFrame) -> None:
        """Test counterfactual strategy."""
        generator = FairGenerator(
            sensitive_attr="gender",
            strategy="counterfactual",
        )

        fair_data = generator.generate(biased_data, n_samples=1000, random_state=42)

        assert len(fair_data) == 1000

    def test_unknown_strategy_raises(self, biased_data: pd.DataFrame) -> None:
        """Test unknown strategy raises error."""
        generator = FairGenerator(
            sensitive_attr="gender",
            strategy="unknown_strategy",
        )

        with pytest.raises(ValueError, match="Unknown strategy"):
            generator.generate(biased_data, n_samples=100)


class TestCounterfactualGenerator:
    """Tests for CounterfactualGenerator."""

    def test_flips_sensitive_attribute(self, biased_data: pd.DataFrame) -> None:
        """Test that counterfactuals flip sensitive attribute."""
        generator = CounterfactualGenerator(
            sensitive_attr="gender",
        )

        counterfactuals = generator.generate(biased_data)

        # All gender values should be flipped
        original_males = (biased_data["gender"] == "male").sum()
        cf_males = (counterfactuals["gender"] == "male").sum()

        assert original_males != cf_males

    def test_fit_learns_adjustments(self, biased_data: pd.DataFrame) -> None:
        """Test that fit learns causal adjustments."""
        generator = CounterfactualGenerator(
            sensitive_attr="gender",
            causal_features=["experience"],
        )

        generator.fit(biased_data)

        assert "experience" in generator._adjustments


class TestFairnessAudit:
    """Tests for FairnessAudit."""

    def test_compare_reports_improvement(self, biased_data: pd.DataFrame) -> None:
        """Test comparison reports improvement."""
        # Generate fair data
        generator = FairGenerator(
            sensitive_attr="gender",
            outcome_col="hired",
            strategy="resampling",
        )
        fair_data = generator.generate(biased_data, n_samples=1000, random_state=42)

        audit = FairnessAudit(
            sensitive_attrs=["gender"],
            outcome_col="hired",
        )

        results = audit.compare(biased_data, fair_data)

        assert "original" in results
        assert "synthetic" in results
        assert "improvement" in results
        assert results["improvement"]["gender"] > 0

    def test_generate_report(self, biased_data: pd.DataFrame) -> None:
        """Test human-readable report generation."""
        generator = FairGenerator(
            sensitive_attr="gender",
            outcome_col="hired",
            strategy="resampling",
        )
        fair_data = generator.generate(biased_data, n_samples=1000, random_state=42)

        audit = FairnessAudit(
            sensitive_attrs=["gender"],
            outcome_col="hired",
        )

        report = audit.generate_report(biased_data, fair_data)

        assert "FAIRNESS AUDIT REPORT" in report
        assert "gender" in report


class TestBalanceDataset:
    """Tests for balance_dataset function."""

    def test_oversample_balances_groups(self, biased_data: pd.DataFrame) -> None:
        """Test oversampling balances groups."""
        balanced = balance_dataset(
            biased_data,
            sensitive_attr="gender",
            strategy="oversample",
            random_state=42,
        )

        male_count = (balanced["gender"] == "male").sum()
        female_count = (balanced["gender"] == "female").sum()

        assert male_count == female_count

    def test_undersample_balances_groups(self, biased_data: pd.DataFrame) -> None:
        """Test undersampling balances groups."""
        balanced = balance_dataset(
            biased_data,
            sensitive_attr="gender",
            strategy="undersample",
            random_state=42,
        )

        male_count = (balanced["gender"] == "male").sum()
        female_count = (balanced["gender"] == "female").sum()

        assert male_count == female_count

    def test_unknown_strategy_raises(self, biased_data: pd.DataFrame) -> None:
        """Test unknown strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            balance_dataset(
                biased_data,
                sensitive_attr="gender",
                strategy="unknown",
            )
