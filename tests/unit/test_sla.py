"""Tests for SLA (Service Level Agreement) contracts."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from genesis.sla import (
    MetricCalculator,
    MetricType,
    SLAConfig,
    SLAContract,
    SLAResult,
    SLAValidator,
    create_ci_report,
    create_github_check,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1000),
        "name": [f"Name_{i}" for i in range(1000)],
        "age": np.random.randint(18, 80, 1000),
        "income": np.random.uniform(20000, 150000, 1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Create reference data for comparison."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(1000),
        "name": [f"Name_{i}" for i in range(1000)],
        "age": np.random.randint(18, 80, 1000),
        "income": np.random.uniform(20000, 150000, 1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })


class TestMetricType:
    """Tests for MetricType enum."""

    def test_values(self) -> None:
        """Test enum values exist."""
        assert MetricType.UNIQUENESS is not None
        assert MetricType.COMPLETENESS is not None
        assert MetricType.STATISTICAL_SIMILARITY is not None
        assert MetricType.CORRELATION_PRESERVATION is not None


class TestSLAConfig:
    """Tests for SLAConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SLAConfig(
            metric=MetricType.UNIQUENESS,
            threshold=0.95,
        )

        assert config.metric == MetricType.UNIQUENESS
        assert config.threshold == 0.95
        assert config.columns is None
        assert config.strict is True

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = SLAConfig(
            metric=MetricType.COMPLETENESS,
            threshold=0.99,
            columns=["name", "age"],
            strict=False,
        )

        assert config.columns == ["name", "age"]
        assert config.strict is False


class TestSLAResult:
    """Tests for SLAResult dataclass."""

    def test_passed_result(self) -> None:
        """Test passed SLA result."""
        result = SLAResult(
            metric=MetricType.UNIQUENESS,
            value=0.98,
            threshold=0.95,
            passed=True,
        )

        assert result.passed is True
        assert result.value > result.threshold

    def test_failed_result(self) -> None:
        """Test failed SLA result."""
        result = SLAResult(
            metric=MetricType.UNIQUENESS,
            value=0.90,
            threshold=0.95,
            passed=False,
            message="Uniqueness below threshold",
        )

        assert result.passed is False
        assert result.message is not None


class TestMetricCalculator:
    """Tests for MetricCalculator."""

    def test_uniqueness(self, sample_data: pd.DataFrame) -> None:
        """Test uniqueness calculation."""
        calculator = MetricCalculator()
        uniqueness = calculator.calculate_uniqueness(sample_data, "id")

        assert uniqueness == 1.0  # All IDs are unique

    def test_uniqueness_with_duplicates(self) -> None:
        """Test uniqueness with duplicate values."""
        data = pd.DataFrame({
            "id": [1, 2, 2, 3, 3, 3, 4, 5],
        })

        calculator = MetricCalculator()
        uniqueness = calculator.calculate_uniqueness(data, "id")

        # 5 unique out of 8
        assert uniqueness == 5 / 8

    def test_completeness(self, sample_data: pd.DataFrame) -> None:
        """Test completeness calculation."""
        calculator = MetricCalculator()
        completeness = calculator.calculate_completeness(sample_data, "name")

        assert completeness == 1.0  # No nulls

    def test_completeness_with_nulls(self) -> None:
        """Test completeness with null values."""
        data = pd.DataFrame({
            "value": [1, 2, None, 4, None],
        })

        calculator = MetricCalculator()
        completeness = calculator.calculate_completeness(data, "value")

        assert completeness == 3 / 5

    def test_range_coverage(self, sample_data: pd.DataFrame) -> None:
        """Test range coverage calculation."""
        calculator = MetricCalculator()
        coverage = calculator.calculate_range_coverage(
            sample_data, "age", min_val=18, max_val=80
        )

        assert 0 < coverage <= 1.0

    def test_distribution_similarity(
        self,
        sample_data: pd.DataFrame,
        reference_data: pd.DataFrame,
    ) -> None:
        """Test distribution similarity calculation."""
        calculator = MetricCalculator()
        similarity = calculator.calculate_distribution_similarity(
            sample_data, reference_data, "age"
        )

        # Same seed, should be very similar
        assert similarity > 0.9


class TestSLAContract:
    """Tests for SLAContract."""

    def test_create_from_dict(self) -> None:
        """Test creating contract from dictionary."""
        config = {
            "name": "test_contract",
            "version": "1.0",
            "slas": [
                {"metric": "uniqueness", "threshold": 0.95, "columns": ["id"]},
                {"metric": "completeness", "threshold": 0.99},
            ],
        }

        contract = SLAContract.from_dict(config)

        assert contract.name == "test_contract"
        assert len(contract.sla_configs) == 2

    def test_create_from_yaml_string(self) -> None:
        """Test creating contract from YAML string."""
        yaml_content = """
        name: test_contract
        version: "1.0"
        slas:
          - metric: uniqueness
            threshold: 0.95
            columns:
              - id
          - metric: completeness
            threshold: 0.99
        """

        contract = SLAContract.from_yaml(yaml_content)

        assert contract.name == "test_contract"
        assert len(contract.sla_configs) == 2


class TestSLAValidator:
    """Tests for SLAValidator."""

    @pytest.fixture
    def contract(self) -> SLAContract:
        """Create sample contract."""
        return SLAContract.from_dict({
            "name": "test",
            "version": "1.0",
            "slas": [
                {"metric": "uniqueness", "threshold": 0.95, "columns": ["id"]},
                {"metric": "completeness", "threshold": 0.95},
            ],
        })

    def test_validate_passes(
        self,
        contract: SLAContract,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test validation passes for good data."""
        validator = SLAValidator(contract)
        results = validator.validate(sample_data)

        assert all(r.passed for r in results)

    def test_validate_fails_uniqueness(self, contract: SLAContract) -> None:
        """Test validation fails for data with duplicates."""
        data = pd.DataFrame({
            "id": [1, 1, 2, 2, 3],  # Many duplicates
            "name": ["A", "B", "C", "D", "E"],
        })

        validator = SLAValidator(contract)
        results = validator.validate(data)

        uniqueness_result = next(
            r for r in results if r.metric == MetricType.UNIQUENESS
        )
        assert uniqueness_result.passed is False

    def test_validate_all_returns_bool(
        self,
        contract: SLAContract,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test validate_all returns boolean."""
        validator = SLAValidator(contract)
        passed = validator.validate_all(sample_data)

        assert isinstance(passed, bool)

    def test_generate_report(
        self,
        contract: SLAContract,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test report generation."""
        validator = SLAValidator(contract)
        results = validator.validate(sample_data)
        report = validator.generate_report(results)

        assert "SLA Validation Report" in report
        assert "uniqueness" in report.lower()


class TestCIIntegration:
    """Tests for CI/CD integration helpers."""

    @pytest.fixture
    def sample_results(self) -> list:
        """Create sample results."""
        return [
            SLAResult(MetricType.UNIQUENESS, 0.98, 0.95, True),
            SLAResult(MetricType.COMPLETENESS, 0.99, 0.95, True),
        ]

    def test_create_ci_report(self, sample_results: list) -> None:
        """Test CI report creation."""
        report = create_ci_report(sample_results)

        assert "passed" in report
        assert "results" in report
        assert report["passed"] is True

    def test_create_ci_report_failure(self) -> None:
        """Test CI report with failure."""
        results = [
            SLAResult(MetricType.UNIQUENESS, 0.80, 0.95, False),
        ]

        report = create_ci_report(results)
        assert report["passed"] is False

    def test_create_github_check(self, sample_results: list) -> None:
        """Test GitHub check creation."""
        check = create_github_check(sample_results, "test-run")

        assert "name" in check
        assert "conclusion" in check
        assert check["conclusion"] == "success"

    def test_create_github_check_failure(self) -> None:
        """Test GitHub check with failure."""
        results = [
            SLAResult(MetricType.UNIQUENESS, 0.80, 0.95, False),
        ]

        check = create_github_check(results, "test-run")
        assert check["conclusion"] == "failure"
