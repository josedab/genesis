"""Tests for Synthetic Data Benchmarking Suite."""

import pytest
import pandas as pd
import numpy as np

from genesis.benchmarking import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkDatasetType,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    CompetitorBenchmark,
    DatasetLoader,
    MetricCategory,
    MetricResult,
    load_benchmark_dataset,
    run_benchmark,
)


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset."""

    def test_create_dataset(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        dataset = BenchmarkDataset(
            name="test",
            data=df,
            discrete_columns=["b"],
        )
        assert dataset.name == "test"
        assert dataset.n_rows == 3
        assert dataset.n_columns == 2

    def test_train_test_split(self):
        df = pd.DataFrame({"a": range(100)})
        dataset = BenchmarkDataset(name="test", data=df)
        train, test = dataset.get_train_test_split(test_size=0.2)
        assert len(test) == 20
        assert len(train) == 80


class TestMetricResult:
    """Tests for MetricResult."""

    def test_create_metric(self):
        metric = MetricResult(
            name="test_metric",
            value=0.85,
            category=MetricCategory.FIDELITY,
        )
        assert metric.name == "test_metric"
        assert metric.value == 0.85


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_success_property(self):
        result = BenchmarkResult(method="ctgan", dataset="adult")
        assert result.success is True

        result_failed = BenchmarkResult(method="ctgan", dataset="adult", error="Failed")
        assert result_failed.success is False

    def test_get_metric(self):
        result = BenchmarkResult(
            method="ctgan",
            dataset="adult",
            metrics=[
                MetricResult("fidelity", 0.9, MetricCategory.FIDELITY),
                MetricResult("privacy", 0.8, MetricCategory.PRIVACY),
            ],
        )
        assert result.get_metric("fidelity").value == 0.9
        assert result.get_metric("unknown") is None

    def test_overall_score(self):
        result = BenchmarkResult(
            method="ctgan",
            dataset="adult",
            metrics=[
                MetricResult("fidelity", 0.9, MetricCategory.FIDELITY),
                MetricResult("privacy", 0.8, MetricCategory.PRIVACY),
            ],
        )
        score = result.get_overall_score()
        assert 0 <= score <= 1

    def test_to_dict(self):
        result = BenchmarkResult(method="ctgan", dataset="adult")
        d = result.to_dict()
        assert d["method"] == "ctgan"
        assert d["dataset"] == "adult"
        assert "overall_score" in d


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        real = pd.DataFrame({
            "numeric": np.random.normal(0, 1, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        })
        synthetic = pd.DataFrame({
            "numeric": np.random.normal(0.1, 1.1, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        })
        return real, synthetic

    def test_column_correlation(self, sample_data):
        real, synthetic = sample_data
        metrics = BenchmarkMetrics()
        result = metrics._column_correlation(real, synthetic)
        assert result.name == "column_correlation"
        assert 0 <= result.value <= 1

    def test_marginal_distribution(self, sample_data):
        real, synthetic = sample_data
        metrics = BenchmarkMetrics()
        result = metrics._marginal_distribution(real, synthetic)
        assert result.name == "marginal_distribution"
        assert 0 <= result.value <= 1

    def test_pairwise_correlation(self):
        np.random.seed(42)
        real = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
        })
        synthetic = pd.DataFrame({
            "a": np.random.normal(0, 1, 100),
            "b": np.random.normal(0, 1, 100),
        })
        metrics = BenchmarkMetrics()
        result = metrics._pairwise_correlation(real, synthetic)
        assert result.name == "pairwise_correlation"
        assert 0 <= result.value <= 1

    def test_statistical_similarity(self, sample_data):
        real, synthetic = sample_data
        metrics = BenchmarkMetrics()
        result = metrics._statistical_similarity(real, synthetic)
        assert result.name == "statistical_similarity"
        assert "column_correlation" in result.details

    def test_distance_to_closest_record(self, sample_data):
        real, synthetic = sample_data
        metrics = BenchmarkMetrics()
        result = metrics._distance_to_closest_record(real, synthetic)
        assert result.name == "dcr"
        assert 0 <= result.value <= 1

    def test_compute_all_metrics(self, sample_data):
        real, synthetic = sample_data
        metrics = BenchmarkMetrics()
        results = metrics.compute(real, synthetic)
        assert len(results) > 0
        assert all(isinstance(r, MetricResult) for r in results)


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_load_adult(self):
        loader = DatasetLoader()
        dataset = loader.load("adult")
        assert dataset.name == "adult"
        assert dataset.n_rows > 0
        assert len(dataset.discrete_columns) > 0

    def test_load_credit(self):
        loader = DatasetLoader()
        dataset = loader.load("credit")
        assert dataset.name == "credit"
        assert dataset.target_column == "credit_risk"

    def test_load_california(self):
        loader = DatasetLoader()
        dataset = loader.load("california")
        assert dataset.name == "california"
        assert dataset.n_columns > 0

    def test_load_unknown_generates_synthetic(self):
        loader = DatasetLoader()
        dataset = loader.load("unknown_dataset")
        assert dataset.n_rows == 10000
        assert "target" in dataset.data.columns


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_run_single_method(self):
        config = BenchmarkConfig(n_samples=100)
        runner = BenchmarkRunner(config)
        loader = DatasetLoader()
        dataset = loader.load("credit")
        
        # Use a simple test - just check the runner works
        # Full benchmark would require fitting a generator
        result = runner.run("gaussian_copula", dataset)
        assert result.dataset == "credit"


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_create_suite(self):
        suite = BenchmarkSuite()
        assert suite.loader is not None
        assert suite.runner is not None

    def test_generate_markdown_leaderboard(self):
        suite = BenchmarkSuite()
        results = [
            BenchmarkResult(
                method="ctgan",
                dataset="adult",
                metrics=[
                    MetricResult("statistical_similarity", 0.85, MetricCategory.FIDELITY),
                    MetricResult("dcr", 0.9, MetricCategory.PRIVACY),
                    MetricResult("ml_efficacy", 0.8, MetricCategory.UTILITY),
                    MetricResult("throughput", 1000, MetricCategory.PERFORMANCE),
                ],
                fit_time=10.5,
                generate_time=2.3,
            ),
        ]
        md = suite._generate_markdown_leaderboard(results)
        assert "# Synthetic Data Benchmark Leaderboard" in md
        assert "ctgan" in md

    def test_generate_html_leaderboard(self):
        suite = BenchmarkSuite()
        results = [
            BenchmarkResult(
                method="tvae",
                dataset="credit",
                metrics=[
                    MetricResult("statistical_similarity", 0.8, MetricCategory.FIDELITY),
                ],
                fit_time=5.0,
            ),
        ]
        html = suite._generate_html_leaderboard(results)
        assert "<html>" in html
        assert "tvae" in html
        assert "Leaderboard" in html


class TestCompetitorBenchmark:
    """Tests for CompetitorBenchmark."""

    def test_check_sdv(self):
        benchmark = CompetitorBenchmark()
        # Just check it doesn't error
        assert benchmark.sdv_available in [True, False]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_benchmark_dataset(self):
        dataset = load_benchmark_dataset("adult")
        assert isinstance(dataset, BenchmarkDataset)
        assert dataset.name == "adult"
