"""Tests for federated synthetic data generation."""

import numpy as np
import pandas as pd
import pytest

from genesis.federated import (
    DataSite,
    FederatedGenerator,
    FederatedTrainingSimulator,
    ModelAggregator,
    SecureAggregator,
    SiteConfig,
    create_federated_generator,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 500),
            "income": np.random.normal(50000, 15000, 500),
            "region": np.random.choice(["North", "South", "East", "West"], 500),
        }
    )


@pytest.fixture
def site_datasets(sample_data: pd.DataFrame):
    """Create datasets for multiple sites."""
    # Split data into 3 sites
    n = len(sample_data)
    return {
        "site_a": sample_data.iloc[: n // 3].copy(),
        "site_b": sample_data.iloc[n // 3 : 2 * n // 3].copy(),
        "site_c": sample_data.iloc[2 * n // 3 :].copy(),
    }


class TestSiteConfig:
    """Tests for SiteConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = SiteConfig(name="test")

        assert config.name == "test"
        assert config.weight == 1.0
        assert config.privacy_budget == 1.0
        assert config.min_samples == 100


class TestDataSite:
    """Tests for DataSite."""

    def test_init(self, sample_data: pd.DataFrame) -> None:
        """Test site initialization."""
        site = DataSite("test_site", sample_data)

        assert site.name == "test_site"
        assert site.n_samples == len(sample_data)

    def test_initialize_and_train(self, sample_data: pd.DataFrame) -> None:
        """Test initializing and training a site."""
        site = DataSite("test_site", sample_data)
        site.initialize(method="gaussian_copula")

        params = site.train_local()

        assert params is not None
        assert "site_name" in params
        assert "n_samples" in params
        assert "numeric_stats" in params

    def test_extracted_params(self, sample_data: pd.DataFrame) -> None:
        """Test that extracted parameters are correct."""
        site = DataSite("test_site", sample_data)
        site.initialize()
        params = site.train_local()

        # Check numeric stats
        assert "age" in params["numeric_stats"]
        assert "income" in params["numeric_stats"]

        age_stats = params["numeric_stats"]["age"]
        assert "mean" in age_stats
        assert "std" in age_stats

        # Check categorical distributions
        assert "region" in params["categorical_distributions"]
        region_dist = params["categorical_distributions"]["region"]
        assert sum(region_dist.values()) == pytest.approx(1.0)


class TestModelAggregator:
    """Tests for ModelAggregator."""

    def test_aggregate_numeric_stats(self) -> None:
        """Test aggregating numeric statistics."""
        aggregator = ModelAggregator()

        site_params = [
            {
                "site_name": "a",
                "n_samples": 100,
                "columns": ["x"],
                "numeric_stats": {"x": {"mean": 10, "std": 2, "min": 5, "max": 15}},
            },
            {
                "site_name": "b",
                "n_samples": 100,
                "columns": ["x"],
                "numeric_stats": {"x": {"mean": 20, "std": 3, "min": 12, "max": 28}},
            },
        ]

        result = aggregator.aggregate(site_params)

        assert result.n_sites == 2
        assert result.total_samples == 200

        # Mean should be weighted average (equal weights here)
        agg_mean = result.parameters["numeric_stats"]["x"]["mean"]
        assert agg_mean == pytest.approx(15, rel=0.1)

    def test_aggregate_categorical(self) -> None:
        """Test aggregating categorical distributions."""
        aggregator = ModelAggregator()

        site_params = [
            {
                "site_name": "a",
                "n_samples": 100,
                "columns": ["cat"],
                "categorical_distributions": {"cat": {"A": 0.6, "B": 0.4}},
            },
            {
                "site_name": "b",
                "n_samples": 100,
                "columns": ["cat"],
                "categorical_distributions": {"cat": {"A": 0.4, "B": 0.6}},
            },
        ]

        result = aggregator.aggregate(site_params)

        # Equal weights, so should average to 0.5 each
        cat_dist = result.parameters["categorical_distributions"]["cat"]
        assert cat_dist["A"] == pytest.approx(0.5, rel=0.1)
        assert cat_dist["B"] == pytest.approx(0.5, rel=0.1)


class TestSecureAggregator:
    """Tests for SecureAggregator with DP."""

    def test_adds_noise(self) -> None:
        """Test that noise is added to aggregated values."""
        aggregator = SecureAggregator(noise_scale=0.5)

        site_params = [
            {
                "site_name": "a",
                "n_samples": 100,
                "columns": ["x"],
                "numeric_stats": {"x": {"mean": 50, "std": 10, "min": 30, "max": 70}},
            },
            {
                "site_name": "b",
                "n_samples": 100,
                "columns": ["x"],
                "numeric_stats": {"x": {"mean": 50, "std": 10, "min": 30, "max": 70}},
            },
        ]

        result = aggregator.aggregate(site_params)

        # With noise, mean might not be exactly 50
        agg_mean = result.parameters["numeric_stats"]["x"]["mean"]
        # But should be close (noise is bounded)
        assert 45 < agg_mean < 55

        # Check that secure flag is set
        assert result.metadata.get("secure_aggregation") is True

    def test_min_sites_requirement(self) -> None:
        """Test minimum sites requirement."""
        aggregator = SecureAggregator(min_sites=3)

        site_params = [
            {"site_name": "a", "n_samples": 100},
            {"site_name": "b", "n_samples": 100},
        ]

        with pytest.raises(ValueError, match="at least 3 sites"):
            aggregator.aggregate(site_params)


class TestFederatedGenerator:
    """Tests for FederatedGenerator."""

    def test_add_site(self, sample_data: pd.DataFrame) -> None:
        """Test adding sites."""
        fed_gen = FederatedGenerator()

        site = DataSite("test", sample_data)
        fed_gen.add_site(site)

        assert len(fed_gen.sites) == 1

    def test_train(self, site_datasets) -> None:
        """Test federated training."""
        fed_gen = create_federated_generator(site_datasets)

        result = fed_gen.train(rounds=1)

        assert fed_gen.is_trained
        assert result.n_sites == 3
        assert result.round_number == 1

    def test_generate(self, site_datasets) -> None:
        """Test generating from trained model."""
        fed_gen = create_federated_generator(site_datasets)
        fed_gen.train(rounds=1)

        synthetic = fed_gen.generate(100, strategy="proportional")

        assert len(synthetic) == 100
        assert "_source_site" in synthetic.columns


class TestFederatedTrainingSimulator:
    """Tests for FederatedTrainingSimulator."""

    def test_setup_from_data(self, sample_data: pd.DataFrame) -> None:
        """Test setting up simulator from data."""
        simulator = FederatedTrainingSimulator(n_sites=3)
        simulator.setup_from_data(sample_data)

        assert len(simulator._sites) == 3

    def test_simulate_training(self, sample_data: pd.DataFrame) -> None:
        """Test simulating federated training."""
        simulator = FederatedTrainingSimulator(n_sites=3)
        simulator.setup_from_data(sample_data)

        results = simulator.simulate_training(n_rounds=2)

        assert results["n_sites"] == 3
        assert len(results["rounds"]) == 2
        assert results["final_model"] is not None

    def test_non_iid_setup(self, sample_data: pd.DataFrame) -> None:
        """Test non-IID partitioning."""
        simulator = FederatedTrainingSimulator(n_sites=4)
        simulator.setup_non_iid(sample_data, partition_column="region")

        assert len(simulator._sites) == 4

        # Sites should have different region distributions
        regions_per_site = []
        for site in simulator._sites:
            regions = site._data["region"].unique()
            regions_per_site.append(set(regions))

        # Not all sites should have all regions (non-IID)
        all_same = all(r == regions_per_site[0] for r in regions_per_site)
        assert not all_same

    def test_generate_synthetic(self, sample_data: pd.DataFrame) -> None:
        """Test generating synthetic data after simulation."""
        simulator = FederatedTrainingSimulator(n_sites=3)
        simulator.setup_from_data(sample_data)
        simulator.simulate_training(n_rounds=1)

        synthetic = simulator.generate_synthetic(100)

        assert len(synthetic) == 100


class TestCreateFederatedGenerator:
    """Tests for create_federated_generator convenience function."""

    def test_creates_generator(self, site_datasets) -> None:
        """Test creating generator from datasets dict."""
        fed_gen = create_federated_generator(site_datasets)

        assert len(fed_gen.sites) == 3
        assert fed_gen.method == "gaussian_copula"

    def test_custom_method(self, site_datasets) -> None:
        """Test with custom method."""
        fed_gen = create_federated_generator(site_datasets, method="gaussian_copula")

        assert fed_gen.method == "gaussian_copula"
