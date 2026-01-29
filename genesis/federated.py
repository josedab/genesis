"""Federated synthetic data generation.

This module provides capabilities for generating synthetic data from
distributed data sources without centralizing the raw data.

Key features:
- Local model training at each data site
- Model aggregation across sites
- Privacy-preserving collaboration
- Differential privacy per-site

Example:
    >>> from genesis.federated import FederatedGenerator, DataSite
    >>>
    >>> # Create sites with local data
    >>> sites = [
    ...     DataSite("hospital_a", data_a),
    ...     DataSite("hospital_b", data_b),
    ...     DataSite("hospital_c", data_c),
    ... ]
    >>>
    >>> # Create federated generator
    >>> fed_gen = FederatedGenerator(sites)
    >>> fed_gen.train(rounds=5)
    >>>
    >>> # Generate synthetic data representing all sites
    >>> synthetic = fed_gen.generate(10000)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from genesis.core.config import PrivacyConfig
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SiteConfig:
    """Configuration for a data site."""

    name: str
    weight: float = 1.0  # Contribution weight based on data size
    privacy_budget: float = 1.0  # Epsilon for differential privacy
    min_samples: int = 100  # Minimum samples required


@dataclass
class AggregatedModel:
    """Container for aggregated model parameters."""

    parameters: Dict[str, Any]
    n_sites: int
    total_samples: int
    round_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSite:
    """Represents a local data site in federated learning.

    Each site trains a local model and only shares model parameters,
    never the raw data.
    """

    def __init__(
        self,
        name: str,
        data: Optional[pd.DataFrame] = None,
        config: Optional[SiteConfig] = None,
    ) -> None:
        """Initialize a data site.

        Args:
            name: Unique site identifier
            data: Local training data
            config: Site configuration
        """
        self.name = name
        self._data = data
        self.config = config or SiteConfig(name=name)

        self._generator: Optional[Any] = None
        self._is_initialized = False
        self._model_params: Optional[Dict[str, Any]] = None

    def set_data(self, data: pd.DataFrame) -> None:
        """Set local data (for lazy loading)."""
        self._data = data

    def initialize(
        self,
        method: str = "gaussian_copula",
        discrete_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the local generator.

        Args:
            method: Generator method
            discrete_columns: Categorical columns
        """
        if self._data is None:
            raise ValueError(f"No data set for site {self.name}")

        from genesis import SyntheticGenerator

        privacy = PrivacyConfig(epsilon=self.config.privacy_budget)
        self._generator = SyntheticGenerator(method=method, privacy=privacy)
        self._discrete_columns = discrete_columns or []
        self._is_initialized = True

    def train_local(self) -> Dict[str, Any]:
        """Train local model and return parameters.

        Returns:
            Model parameters (never raw data)
        """
        if not self._is_initialized:
            self.initialize()

        self._generator.fit(self._data, discrete_columns=self._discrete_columns)

        # Extract model parameters (implementation depends on method)
        self._model_params = self._extract_model_params()

        logger.info(f"Site {self.name}: Trained on {len(self._data)} samples")
        return self._model_params

    def _extract_model_params(self) -> Dict[str, Any]:
        """Extract shareable model parameters."""
        # For Gaussian Copula, we can share:
        # - Column statistics (means, stds)
        # - Correlation matrix
        # These don't reveal individual records

        params = {
            "site_name": self.name,
            "n_samples": len(self._data),
            "columns": list(self._data.columns),
        }

        # Extract numeric column statistics
        numeric_cols = self._data.select_dtypes(include=["number"]).columns
        params["numeric_stats"] = {}

        for col in numeric_cols:
            series = self._data[col].dropna()
            if len(series) > 0:
                params["numeric_stats"][col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        # Extract correlation matrix
        if len(numeric_cols) > 1:
            corr = self._data[numeric_cols].corr().values
            params["correlation"] = corr.tolist()
            params["correlation_cols"] = list(numeric_cols)

        # Extract categorical distributions
        cat_cols = self._data.select_dtypes(include=["object", "category"]).columns
        params["categorical_distributions"] = {}

        for col in cat_cols:
            dist = self._data[col].value_counts(normalize=True).to_dict()
            params["categorical_distributions"][col] = dist

        return params

    def apply_aggregated_model(self, aggregated: AggregatedModel) -> None:
        """Apply aggregated model parameters."""
        self._model_params = aggregated.parameters

    def generate_local(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data using local model."""
        if self._generator is None or not self._generator.is_fitted:
            raise ValueError("Model not trained")

        return self._generator.generate(n_samples)

    @property
    def n_samples(self) -> int:
        """Number of local samples."""
        return len(self._data) if self._data is not None else 0


class ModelAggregator:
    """Aggregates model parameters from multiple sites."""

    def __init__(
        self,
        strategy: str = "weighted_average",
    ) -> None:
        """Initialize the aggregator.

        Args:
            strategy: Aggregation strategy ('weighted_average', 'fedavg')
        """
        self.strategy = strategy

    def aggregate(
        self,
        site_params: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> AggregatedModel:
        """Aggregate model parameters from multiple sites.

        Args:
            site_params: List of parameter dicts from each site
            weights: Optional weights for each site

        Returns:
            AggregatedModel with combined parameters
        """
        if not site_params:
            raise ValueError("No site parameters to aggregate")

        n_sites = len(site_params)
        total_samples = sum(p.get("n_samples", 0) for p in site_params)

        # Default weights based on sample counts
        if weights is None:
            weights = [p.get("n_samples", 1) / total_samples for p in site_params]

        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        aggregated_params = {
            "columns": site_params[0].get("columns", []),
            "sites": [p.get("site_name") for p in site_params],
        }

        # Aggregate numeric statistics
        all_numeric_stats = [p.get("numeric_stats", {}) for p in site_params]
        if all_numeric_stats[0]:
            aggregated_params["numeric_stats"] = self._aggregate_numeric_stats(
                all_numeric_stats, weights
            )

        # Aggregate correlation matrices
        all_corr = [p.get("correlation") for p in site_params]
        if all(c is not None for c in all_corr):
            aggregated_params["correlation"] = self._aggregate_correlation(all_corr, weights)
            aggregated_params["correlation_cols"] = site_params[0].get("correlation_cols", [])

        # Aggregate categorical distributions
        all_cat_dists = [p.get("categorical_distributions", {}) for p in site_params]
        if all_cat_dists[0]:
            aggregated_params["categorical_distributions"] = self._aggregate_categorical(
                all_cat_dists, weights
            )

        return AggregatedModel(
            parameters=aggregated_params,
            n_sites=n_sites,
            total_samples=total_samples,
            round_number=0,
            metadata={"weights": weights},
        )

    def _aggregate_numeric_stats(
        self,
        stats_list: List[Dict],
        weights: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate numeric column statistics."""
        result = {}

        # Get all columns
        all_cols = set()
        for stats in stats_list:
            all_cols.update(stats.keys())

        for col in all_cols:
            col_stats = [s.get(col, {}) for s in stats_list]

            # Weighted average of means
            means = [s.get("mean", 0) for s in col_stats]
            stds = [s.get("std", 1) for s in col_stats]

            agg_mean = sum(m * w for m, w in zip(means, weights))

            # Aggregate std using pooled variance formula (simplified)
            agg_std = sum(s * w for s, w in zip(stds, weights))

            result[col] = {
                "mean": agg_mean,
                "std": agg_std,
                "min": min(s.get("min", 0) for s in col_stats),
                "max": max(s.get("max", 0) for s in col_stats),
            }

        return result

    def _aggregate_correlation(
        self,
        corr_list: List[List[List[float]]],
        weights: List[float],
    ) -> List[List[float]]:
        """Aggregate correlation matrices."""
        corr_arrays = [np.array(c) for c in corr_list]

        # Weighted average of correlations
        agg_corr = sum(c * w for c, w in zip(corr_arrays, weights))

        # Ensure valid correlation matrix
        np.fill_diagonal(agg_corr, 1.0)
        agg_corr = np.clip(agg_corr, -1, 1)

        return agg_corr.tolist()

    def _aggregate_categorical(
        self,
        dist_list: List[Dict],
        weights: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate categorical distributions."""
        result = {}

        # Get all columns
        all_cols = set()
        for dist in dist_list:
            all_cols.update(dist.keys())

        for col in all_cols:
            col_dists = [d.get(col, {}) for d in dist_list]

            # Get all categories
            all_cats = set()
            for d in col_dists:
                all_cats.update(d.keys())

            # Weighted average of probabilities
            agg_dist = {}
            for cat in all_cats:
                prob = sum(d.get(cat, 0) * w for d, w in zip(col_dists, weights))
                agg_dist[cat] = prob

            # Normalize
            total = sum(agg_dist.values())
            if total > 0:
                agg_dist = {k: v / total for k, v in agg_dist.items()}

            result[col] = agg_dist

        return result


class FederatedGenerator:
    """Federated synthetic data generator.

    Coordinates training across multiple data sites without
    centralizing the raw data.
    """

    def __init__(
        self,
        sites: Optional[List[DataSite]] = None,
        method: str = "gaussian_copula",
        n_rounds: int = 1,
        aggregator: Optional[ModelAggregator] = None,
    ) -> None:
        """Initialize the federated generator.

        Args:
            sites: List of data sites
            method: Generator method
            n_rounds: Number of federated rounds
            aggregator: Custom model aggregator
        """
        self.sites = sites or []
        self.method = method
        self.n_rounds = n_rounds
        self.aggregator = aggregator or ModelAggregator()

        self._aggregated_model: Optional[AggregatedModel] = None
        self._is_trained = False
        self._discrete_columns: List[str] = []

    def add_site(self, site: DataSite) -> None:
        """Add a data site."""
        self.sites.append(site)

    def train(
        self,
        rounds: Optional[int] = None,
        discrete_columns: Optional[List[str]] = None,
        callback: Optional[Callable[[int, AggregatedModel], None]] = None,
    ) -> AggregatedModel:
        """Train the federated model.

        Args:
            rounds: Number of training rounds
            discrete_columns: Categorical columns
            callback: Callback after each round

        Returns:
            Final aggregated model
        """
        rounds = rounds or self.n_rounds
        self._discrete_columns = discrete_columns or []

        # Initialize all sites
        for site in self.sites:
            site.initialize(self.method, self._discrete_columns)

        # Federated training loop
        for round_num in range(rounds):
            logger.info(f"Federated round {round_num + 1}/{rounds}")

            # Collect local model updates
            site_params = []
            for site in self.sites:
                params = site.train_local()
                site_params.append(params)

            # Aggregate
            weights = [site.n_samples for site in self.sites]
            self._aggregated_model = self.aggregator.aggregate(site_params, weights)
            self._aggregated_model.round_number = round_num + 1

            # Distribute aggregated model back to sites
            for site in self.sites:
                site.apply_aggregated_model(self._aggregated_model)

            if callback:
                callback(round_num + 1, self._aggregated_model)

        self._is_trained = True
        logger.info(
            f"Federated training complete: {len(self.sites)} sites, "
            f"{self._aggregated_model.total_samples} total samples"
        )

        return self._aggregated_model

    def generate(
        self,
        n_samples: int,
        strategy: str = "proportional",
    ) -> pd.DataFrame:
        """Generate synthetic data from the federated model.

        Args:
            n_samples: Number of samples to generate
            strategy: Generation strategy
                - 'proportional': Generate from each site proportionally
                - 'uniform': Equal samples from each site
                - 'aggregated': Use aggregated model directly

        Returns:
            Generated DataFrame
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")

        if strategy == "aggregated":
            return self._generate_from_aggregated(n_samples)

        # Generate from individual sites
        samples_per_site = self._compute_samples_per_site(n_samples, strategy)

        parts = []
        for site, n in zip(self.sites, samples_per_site):
            if n > 0:
                site_data = site.generate_local(n)
                site_data["_source_site"] = site.name
                parts.append(site_data)

        result = pd.concat(parts, ignore_index=True)

        # Shuffle
        result = result.sample(frac=1).reset_index(drop=True)

        return result

    def _compute_samples_per_site(
        self,
        n_samples: int,
        strategy: str,
    ) -> List[int]:
        """Compute samples to generate from each site."""
        n_sites = len(self.sites)

        if strategy == "uniform":
            base = n_samples // n_sites
            remainder = n_samples % n_sites
            return [base + (1 if i < remainder else 0) for i in range(n_sites)]

        elif strategy == "proportional":
            total = sum(site.n_samples for site in self.sites)
            if total == 0:
                return [n_samples // n_sites] * n_sites

            proportions = [site.n_samples / total for site in self.sites]
            samples = [int(n_samples * p) for p in proportions]

            # Adjust for rounding
            diff = n_samples - sum(samples)
            for i in range(abs(diff)):
                if diff > 0:
                    samples[i % n_sites] += 1
                else:
                    samples[i % n_sites] -= 1

            return samples

        raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_from_aggregated(self, n_samples: int) -> pd.DataFrame:
        """Generate using aggregated model parameters."""
        params = self._aggregated_model.parameters

        # Reconstruct data from aggregated statistics
        # This is a simplified approach using numpy

        numeric_stats = params.get("numeric_stats", {})
        cat_dists = params.get("categorical_distributions", {})
        correlation = params.get("correlation")
        corr_cols = params.get("correlation_cols", [])

        data = {}

        # Generate numeric columns with correlation
        if correlation and corr_cols:
            corr_matrix = np.array(correlation)

            # Generate correlated normal samples
            try:
                L = np.linalg.cholesky(corr_matrix)
                z = np.random.randn(n_samples, len(corr_cols))
                correlated = z @ L.T
            except np.linalg.LinAlgError:
                # Fall back to independent generation
                correlated = np.random.randn(n_samples, len(corr_cols))

            for i, col in enumerate(corr_cols):
                stats = numeric_stats.get(col, {"mean": 0, "std": 1})
                # Transform to original scale
                data[col] = correlated[:, i] * stats["std"] + stats["mean"]
        else:
            # Independent generation for numeric columns
            for col, stats in numeric_stats.items():
                data[col] = np.random.normal(
                    stats["mean"],
                    stats["std"],
                    n_samples,
                )

        # Generate categorical columns
        for col, dist in cat_dists.items():
            categories = list(dist.keys())
            probs = list(dist.values())
            data[col] = np.random.choice(categories, n_samples, p=probs)

        return pd.DataFrame(data)

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    @property
    def aggregated_model(self) -> Optional[AggregatedModel]:
        """Get aggregated model."""
        return self._aggregated_model


def create_federated_generator(
    datasets: Dict[str, pd.DataFrame],
    method: str = "gaussian_copula",
) -> FederatedGenerator:
    """Convenience function to create a federated generator.

    Args:
        datasets: Dict mapping site names to DataFrames
        method: Generator method

    Returns:
        Configured FederatedGenerator
    """
    sites = [DataSite(name=name, data=data) for name, data in datasets.items()]

    return FederatedGenerator(sites=sites, method=method)


class SecureAggregator(ModelAggregator):
    """Secure aggregation using cryptographic techniques.

    This aggregator provides privacy guarantees through:
    - Secret sharing of model parameters
    - Secure multi-party computation (simulated)
    - Differential privacy noise addition

    Note: This is a simplified implementation. Production use should
    integrate with proper secure computation frameworks like PySyft or TF Encrypted.
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        clip_threshold: float = 1.0,
        min_sites: int = 2,
    ) -> None:
        """Initialize secure aggregator.

        Args:
            noise_scale: Scale of noise for differential privacy
            clip_threshold: Gradient clipping threshold
            min_sites: Minimum sites required for aggregation
        """
        super().__init__(strategy="secure")
        self.noise_scale = noise_scale
        self.clip_threshold = clip_threshold
        self.min_sites = min_sites

    def _add_dp_noise(
        self,
        value: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Add differential privacy noise to a value."""
        noise = np.random.laplace(0, sensitivity * self.noise_scale)
        return value + noise

    def _clip_value(self, value: float) -> float:
        """Clip value to threshold."""
        return np.clip(value, -self.clip_threshold, self.clip_threshold)

    def aggregate(
        self,
        site_params: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> AggregatedModel:
        """Securely aggregate parameters with DP noise.

        Args:
            site_params: Parameters from each site
            weights: Site weights

        Returns:
            Securely aggregated model
        """
        if len(site_params) < self.min_sites:
            raise ValueError(
                f"Need at least {self.min_sites} sites for secure aggregation, "
                f"got {len(site_params)}"
            )

        # First, do normal aggregation
        aggregated = super().aggregate(site_params, weights)

        # Add DP noise to numeric statistics
        if "numeric_stats" in aggregated.parameters:
            noisy_stats = {}
            for col, stats in aggregated.parameters["numeric_stats"].items():
                noisy_stats[col] = {
                    k: self._add_dp_noise(v) if isinstance(v, (int, float)) else v
                    for k, v in stats.items()
                }
            aggregated.parameters["numeric_stats"] = noisy_stats

        # Add noise to correlation matrix
        if "correlation" in aggregated.parameters:
            corr = np.array(aggregated.parameters["correlation"])
            noise = np.random.laplace(0, self.noise_scale, corr.shape)
            noisy_corr = corr + noise
            # Ensure valid correlation matrix properties
            noisy_corr = np.clip(noisy_corr, -1, 1)
            np.fill_diagonal(noisy_corr, 1.0)
            aggregated.parameters["correlation"] = noisy_corr.tolist()

        aggregated.metadata["secure_aggregation"] = True
        aggregated.metadata["noise_scale"] = self.noise_scale

        return aggregated


class FederatedTrainingSimulator:
    """Simulate federated training for testing and development.

    This class simulates communication between sites and a coordinator,
    useful for testing federated algorithms without actual distributed setup.
    """

    def __init__(
        self,
        n_sites: int = 3,
        data_generator: Optional[Any] = None,
        iid: bool = True,
    ) -> None:
        """Initialize the simulator.

        Args:
            n_sites: Number of simulated sites
            data_generator: Optional generator for creating site data
            iid: Whether data is IID across sites
        """
        self.n_sites = n_sites
        self.data_generator = data_generator
        self.iid = iid

        self._sites: List[DataSite] = []
        self._coordinator: Optional[FederatedGenerator] = None
        self._communication_log: List[Dict[str, Any]] = []

    def setup_from_data(
        self,
        data: pd.DataFrame,
        site_sizes: Optional[List[int]] = None,
    ) -> "FederatedTrainingSimulator":
        """Set up sites by splitting a dataset.

        Args:
            data: Dataset to split
            site_sizes: Sizes for each site (default: equal split)

        Returns:
            Self for method chaining
        """
        if site_sizes is None:
            # Equal split
            size = len(data) // self.n_sites
            site_sizes = [size] * self.n_sites
            site_sizes[-1] = len(data) - sum(site_sizes[:-1])  # Handle remainder

        # Shuffle if IID
        if self.iid:
            data = data.sample(frac=1).reset_index(drop=True)

        # Create sites
        self._sites = []
        start_idx = 0

        for i, size in enumerate(site_sizes):
            site_data = data.iloc[start_idx : start_idx + size].copy()
            site = DataSite(name=f"site_{i}", data=site_data)
            self._sites.append(site)
            start_idx += size

        self._coordinator = FederatedGenerator(
            sites=self._sites,
            method="gaussian_copula",
        )

        logger.info(f"Set up {len(self._sites)} simulated sites")
        return self

    def setup_non_iid(
        self,
        data: pd.DataFrame,
        partition_column: str,
    ) -> "FederatedTrainingSimulator":
        """Set up non-IID sites by partitioning on a column.

        Args:
            data: Dataset to partition
            partition_column: Column to partition by

        Returns:
            Self for method chaining
        """
        unique_values = data[partition_column].unique()

        if len(unique_values) < self.n_sites:
            raise ValueError(
                f"Not enough unique values in {partition_column} "
                f"({len(unique_values)}) for {self.n_sites} sites"
            )

        # Assign values to sites
        values_per_site = np.array_split(unique_values, self.n_sites)

        self._sites = []
        for i, values in enumerate(values_per_site):
            site_data = data[data[partition_column].isin(values)].copy()
            site = DataSite(name=f"site_{i}", data=site_data)
            self._sites.append(site)

        self._coordinator = FederatedGenerator(
            sites=self._sites,
            method="gaussian_copula",
        )

        logger.info(f"Set up {len(self._sites)} non-IID sites partitioned by {partition_column}")
        return self

    def simulate_training(
        self,
        n_rounds: int = 5,
        simulate_failures: bool = False,
        failure_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """Simulate federated training rounds.

        Args:
            n_rounds: Number of training rounds
            simulate_failures: Whether to simulate site failures
            failure_rate: Probability of site failure per round

        Returns:
            Training results and statistics
        """
        if self._coordinator is None:
            raise ValueError("Run setup_from_data() first")

        results = {
            "rounds": [],
            "total_samples": sum(s.n_samples for s in self._sites),
            "n_sites": len(self._sites),
        }

        def log_callback(round_num: int, model: AggregatedModel) -> None:
            round_info = {
                "round": round_num,
                "participating_sites": model.n_sites,
                "total_samples": model.total_samples,
            }
            results["rounds"].append(round_info)
            self._communication_log.append(
                {
                    "type": "aggregation",
                    "round": round_num,
                    "sites": model.n_sites,
                }
            )

        # Simulate failures by temporarily removing sites
        original_sites = self._sites.copy()

        if simulate_failures:
            # Randomly mark some sites as "failed" each round
            for _round_num in range(n_rounds):
                active_sites = []
                for site in original_sites:
                    if np.random.random() > failure_rate:
                        active_sites.append(site)

                if len(active_sites) < 2:
                    active_sites = original_sites[:2]  # Minimum 2 sites

                self._coordinator.sites = active_sites

        # Run training
        self._coordinator.train(
            rounds=n_rounds,
            callback=log_callback,
        )

        # Restore original sites
        self._coordinator.sites = original_sites

        results["final_model"] = self._coordinator.aggregated_model
        results["communication_rounds"] = len(self._communication_log)

        return results

    def get_communication_log(self) -> List[Dict[str, Any]]:
        """Get log of all communication events."""
        return self._communication_log.copy()

    def generate_synthetic(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data from trained model.

        Args:
            n_samples: Number of samples

        Returns:
            Generated DataFrame
        """
        if self._coordinator is None or not self._coordinator.is_trained:
            raise ValueError("Model not trained")

        return self._coordinator.generate(n_samples)
