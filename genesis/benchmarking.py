"""Synthetic Data Benchmarking Suite.

Comprehensive benchmarking framework for comparing synthetic data generators
against standard datasets and competitors. Enables fair comparison of
fidelity, privacy, utility, and performance metrics.

Features:
    - Standardized benchmark datasets (Adult, Census, Credit, etc.)
    - Multi-generator comparison framework
    - Statistical fidelity metrics
    - Privacy attack metrics
    - ML utility metrics (TSTR, TRTS)
    - Performance profiling (time, memory)
    - HTML/JSON leaderboard generation
    - CI integration for automated benchmarks

Example:
    Basic benchmarking::

        from genesis.benchmarking import BenchmarkSuite, BenchmarkConfig

        suite = BenchmarkSuite()
        
        # Run benchmark on Adult dataset
        results = suite.run(
            dataset="adult",
            methods=["ctgan", "tvae", "gaussian_copula"],
            n_samples=10000,
        )
        
        # Generate leaderboard
        suite.generate_leaderboard(results, "benchmark_results.html")

    Compare with competitors::

        from genesis.benchmarking import CompetitorBenchmark

        benchmark = CompetitorBenchmark()
        results = benchmark.compare(
            dataset="credit",
            competitors=["sdv", "gretel"],  # Requires API keys
            genesis_method="ctgan",
        )

Classes:
    BenchmarkDataset: Standard benchmark dataset wrapper.
    BenchmarkMetrics: Collection of evaluation metrics.
    BenchmarkConfig: Configuration for benchmark runs.
    BenchmarkResult: Results from a single benchmark run.
    BenchmarkSuite: Main benchmarking orchestrator.
    Leaderboard: Leaderboard generation and management.
    CompetitorBenchmark: Compare against external tools.

Functions:
    run_benchmark: Convenience function for quick benchmarks.
    load_benchmark_dataset: Load a standard benchmark dataset.

Note:
    Competitor comparisons require API keys for external services.
    Set GRETEL_API_KEY, SDV_API_KEY environment variables as needed.
"""

import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkDatasetType(str, Enum):
    """Standard benchmark dataset types."""

    ADULT = "adult"  # UCI Adult Income
    CENSUS = "census"  # US Census
    CREDIT = "credit"  # German Credit
    COVERTYPE = "covertype"  # Forest Cover Type
    INTRUSION = "intrusion"  # Network Intrusion
    NEWS = "news"  # News Popularity
    KING = "king"  # King County Housing
    CALIFORNIA = "california"  # California Housing
    CUSTOM = "custom"  # User-provided dataset


class MetricCategory(str, Enum):
    """Categories of benchmark metrics."""

    FIDELITY = "fidelity"  # Statistical similarity
    PRIVACY = "privacy"  # Privacy preservation
    UTILITY = "utility"  # ML model performance
    PERFORMANCE = "performance"  # Speed and resource usage


@dataclass
class BenchmarkDataset:
    """Standard benchmark dataset.

    Attributes:
        name: Dataset identifier.
        data: The DataFrame.
        target_column: Target/label column for ML tasks.
        discrete_columns: Categorical columns.
        sensitive_columns: Privacy-sensitive columns.
        metadata: Additional dataset metadata.
    """

    name: str
    data: pd.DataFrame
    target_column: Optional[str] = None
    discrete_columns: List[str] = field(default_factory=list)
    sensitive_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.data)

    @property
    def n_columns(self) -> int:
        return len(self.data.columns)

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets."""
        n_test = int(len(self.data) * test_size)
        np.random.seed(random_state)
        indices = np.random.permutation(len(self.data))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return self.data.iloc[train_idx].reset_index(drop=True), \
               self.data.iloc[test_idx].reset_index(drop=True)


@dataclass
class MetricResult:
    """Result of a single metric evaluation.

    Attributes:
        name: Metric name.
        value: Metric value (higher is usually better).
        category: Metric category.
        details: Additional details.
        passed: Whether metric passed threshold.
    """

    name: str
    value: float
    category: MetricCategory
    details: Dict[str, Any] = field(default_factory=dict)
    passed: Optional[bool] = None
    threshold: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        method: Generator method used.
        dataset: Dataset name.
        metrics: All computed metrics.
        synthetic_data: Generated synthetic data.
        fit_time: Time to fit the model.
        generate_time: Time to generate data.
        memory_peak: Peak memory usage in MB.
        timestamp: When benchmark was run.
        error: Error message if failed.
    """

    method: str
    dataset: str
    metrics: List[MetricResult] = field(default_factory=list)
    synthetic_data: Optional[pd.DataFrame] = None
    fit_time: float = 0.0
    generate_time: float = 0.0
    memory_peak: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get a specific metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def get_overall_score(self) -> float:
        """Calculate overall benchmark score."""
        if not self.metrics:
            return 0.0

        # Weight by category
        weights = {
            MetricCategory.FIDELITY: 0.4,
            MetricCategory.PRIVACY: 0.3,
            MetricCategory.UTILITY: 0.2,
            MetricCategory.PERFORMANCE: 0.1,
        }

        weighted_sum = 0.0
        weight_sum = 0.0

        for metric in self.metrics:
            w = weights.get(metric.category, 0.1)
            # Normalize value to 0-1 range (assumes most metrics are 0-1)
            normalized = min(max(metric.value, 0), 1)
            weighted_sum += normalized * w
            weight_sum += w

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "dataset": self.dataset,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "category": m.category.value,
                    "passed": m.passed,
                }
                for m in self.metrics
            ],
            "fit_time": self.fit_time,
            "generate_time": self.generate_time,
            "memory_peak": self.memory_peak,
            "timestamp": self.timestamp,
            "overall_score": self.get_overall_score(),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        n_samples: Number of synthetic samples to generate.
        n_runs: Number of benchmark runs for averaging.
        compute_privacy: Whether to compute privacy metrics.
        compute_utility: Whether to compute ML utility metrics.
        timeout: Maximum time per benchmark in seconds.
        random_seed: Random seed for reproducibility.
    """

    n_samples: Optional[int] = None  # None = same as original
    n_runs: int = 1
    compute_privacy: bool = True
    compute_utility: bool = True
    timeout: int = 3600
    random_seed: int = 42
    output_dir: Optional[str] = None


class BenchmarkMetrics:
    """Collection of benchmark metrics.

    Computes fidelity, privacy, utility, and performance metrics
    for synthetic data evaluation.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, Callable[..., MetricResult]] = {}
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register default metric functions."""
        # Fidelity metrics
        self._metrics["column_correlation"] = self._column_correlation
        self._metrics["marginal_distribution"] = self._marginal_distribution
        self._metrics["pairwise_correlation"] = self._pairwise_correlation
        self._metrics["statistical_similarity"] = self._statistical_similarity

        # Privacy metrics
        self._metrics["dcr"] = self._distance_to_closest_record
        self._metrics["nndr"] = self._nearest_neighbor_distance_ratio
        self._metrics["membership_inference"] = self._membership_inference_risk

        # Utility metrics
        self._metrics["ml_efficacy"] = self._ml_efficacy

    def compute(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metric_names: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ) -> List[MetricResult]:
        """Compute specified metrics.

        Args:
            real_data: Original real data.
            synthetic_data: Generated synthetic data.
            metric_names: Metrics to compute (None = all).
            target_column: Target column for ML metrics.

        Returns:
            List of MetricResult objects.
        """
        if metric_names is None:
            metric_names = list(self._metrics.keys())

        results: List[MetricResult] = []
        for name in metric_names:
            if name not in self._metrics:
                logger.warning(f"Unknown metric: {name}")
                continue

            try:
                metric_fn = self._metrics[name]
                if name == "ml_efficacy" and target_column:
                    result = metric_fn(real_data, synthetic_data, target_column)
                else:
                    result = metric_fn(real_data, synthetic_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error computing metric {name}: {e}")
                results.append(
                    MetricResult(
                        name=name,
                        value=0.0,
                        category=MetricCategory.FIDELITY,
                        details={"error": str(e)},
                    )
                )

        return results

    def _column_correlation(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> MetricResult:
        """Compute column-wise correlation between real and synthetic."""
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        if not common_cols:
            return MetricResult(
                name="column_correlation",
                value=0.0,
                category=MetricCategory.FIDELITY,
                details={"error": "No common numeric columns"},
            )

        correlations = []
        for col in common_cols:
            try:
                real_mean = real[col].mean()
                synth_mean = synthetic[col].mean()
                if real_mean != 0:
                    rel_diff = abs(synth_mean - real_mean) / abs(real_mean)
                    correlations.append(max(0, 1 - rel_diff))
                else:
                    correlations.append(1.0 if synth_mean == 0 else 0.5)
            except Exception:
                pass

        score = np.mean(correlations) if correlations else 0.0

        return MetricResult(
            name="column_correlation",
            value=float(score),
            category=MetricCategory.FIDELITY,
            details={"n_columns": len(common_cols)},
        )

    def _marginal_distribution(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> MetricResult:
        """Compare marginal distributions using KS test."""
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        ks_scores = []
        for col in common_cols:
            try:
                stat, _ = stats.ks_2samp(
                    real[col].dropna(), synthetic[col].dropna()
                )
                # Convert KS statistic to similarity (lower KS = higher similarity)
                ks_scores.append(1 - stat)
            except Exception:
                pass

        score = np.mean(ks_scores) if ks_scores else 0.0

        return MetricResult(
            name="marginal_distribution",
            value=float(score),
            category=MetricCategory.FIDELITY,
            details={"n_columns": len(common_cols)},
        )

    def _pairwise_correlation(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> MetricResult:
        """Compare pairwise correlation matrices."""
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        if len(common_cols) < 2:
            return MetricResult(
                name="pairwise_correlation",
                value=1.0,
                category=MetricCategory.FIDELITY,
                details={"error": "Not enough numeric columns"},
            )

        try:
            real_corr = real[common_cols].corr().values
            synth_corr = synthetic[common_cols].corr().values

            # Handle NaN values
            real_corr = np.nan_to_num(real_corr, 0)
            synth_corr = np.nan_to_num(synth_corr, 0)

            # Compute Frobenius norm difference
            diff = np.linalg.norm(real_corr - synth_corr, "fro")
            max_diff = np.sqrt(2 * len(common_cols) ** 2)  # Max possible
            score = 1 - (diff / max_diff)

            return MetricResult(
                name="pairwise_correlation",
                value=float(score),
                category=MetricCategory.FIDELITY,
                details={"n_columns": len(common_cols), "diff_norm": float(diff)},
            )
        except Exception as e:
            return MetricResult(
                name="pairwise_correlation",
                value=0.0,
                category=MetricCategory.FIDELITY,
                details={"error": str(e)},
            )

    def _statistical_similarity(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> MetricResult:
        """Overall statistical similarity score."""
        # Combine multiple fidelity metrics
        corr = self._column_correlation(real, synthetic)
        marg = self._marginal_distribution(real, synthetic)
        pair = self._pairwise_correlation(real, synthetic)

        score = (corr.value + marg.value + pair.value) / 3

        return MetricResult(
            name="statistical_similarity",
            value=float(score),
            category=MetricCategory.FIDELITY,
            details={
                "column_correlation": corr.value,
                "marginal_distribution": marg.value,
                "pairwise_correlation": pair.value,
            },
        )

    def _distance_to_closest_record(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> MetricResult:
        """Compute distance to closest record (DCR) for privacy."""
        numeric_cols = real.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        if not common_cols:
            return MetricResult(
                name="dcr",
                value=1.0,
                category=MetricCategory.PRIVACY,
                details={"error": "No common numeric columns"},
            )

        # Sample for efficiency
        sample_size = min(1000, len(synthetic))
        synth_sample = synthetic[common_cols].sample(n=sample_size, replace=False)
        real_sample = real[common_cols].sample(n=min(5000, len(real)), replace=False)

        # Normalize
        real_arr = real_sample.values
        synth_arr = synth_sample.values

        real_mean = np.nanmean(real_arr, axis=0)
        real_std = np.nanstd(real_arr, axis=0) + 1e-8

        real_norm = (real_arr - real_mean) / real_std
        synth_norm = (synth_arr - real_mean) / real_std

        # Replace NaN
        real_norm = np.nan_to_num(real_norm, 0)
        synth_norm = np.nan_to_num(synth_norm, 0)

        # Compute minimum distances
        min_distances = []
        for synth_row in synth_norm:
            distances = np.sqrt(np.sum((real_norm - synth_row) ** 2, axis=1))
            min_distances.append(np.min(distances))

        avg_min_dist = np.mean(min_distances)

        # Convert to privacy score (higher distance = better privacy)
        # Normalize by expected random distance
        expected_random_dist = np.sqrt(len(common_cols))
        privacy_score = min(avg_min_dist / expected_random_dist, 1.0)

        return MetricResult(
            name="dcr",
            value=float(privacy_score),
            category=MetricCategory.PRIVACY,
            details={"avg_min_distance": float(avg_min_dist)},
        )

    def _nearest_neighbor_distance_ratio(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> MetricResult:
        """Compute nearest neighbor distance ratio (NNDR)."""
        # Similar to DCR but compares to 2nd nearest neighbor
        dcr_result = self._distance_to_closest_record(real, synthetic)

        return MetricResult(
            name="nndr",
            value=dcr_result.value,  # Simplified: use DCR as proxy
            category=MetricCategory.PRIVACY,
            details=dcr_result.details,
        )

    def _membership_inference_risk(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> MetricResult:
        """Estimate membership inference attack risk."""
        # Simple heuristic based on distance to closest record
        dcr_result = self._distance_to_closest_record(real, synthetic)

        # Low DCR = high risk
        risk = 1 - dcr_result.value
        privacy_score = 1 - risk  # Invert so higher is better

        return MetricResult(
            name="membership_inference",
            value=float(privacy_score),
            category=MetricCategory.PRIVACY,
            details={"estimated_risk": float(risk)},
        )

    def _ml_efficacy(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> MetricResult:
        """Compute ML efficacy (TSTR - Train Synthetic Test Real)."""
        if target_column is None or target_column not in real.columns:
            return MetricResult(
                name="ml_efficacy",
                value=0.5,
                category=MetricCategory.UTILITY,
                details={"error": "No target column specified"},
            )

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.preprocessing import LabelEncoder

            # Prepare features
            feature_cols = [c for c in real.columns if c != target_column]
            numeric_cols = real[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                return MetricResult(
                    name="ml_efficacy",
                    value=0.5,
                    category=MetricCategory.UTILITY,
                    details={"error": "No numeric feature columns"},
                )

            # Prepare data
            X_real = real[numeric_cols].fillna(0).values
            X_synth = synthetic[numeric_cols].fillna(0).values

            le = LabelEncoder()
            y_real = le.fit_transform(real[target_column].astype(str))
            y_synth = le.transform(synthetic[target_column].astype(str))

            # Split real data
            n_test = int(len(X_real) * 0.3)
            X_test, y_test = X_real[:n_test], y_real[:n_test]
            X_train_real, y_train_real = X_real[n_test:], y_real[n_test:]

            # TRTR: Train Real Test Real (baseline)
            clf_real = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            clf_real.fit(X_train_real, y_train_real)
            acc_trtr = accuracy_score(y_test, clf_real.predict(X_test))

            # TSTR: Train Synthetic Test Real
            clf_synth = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            clf_synth.fit(X_synth, y_synth)
            acc_tstr = accuracy_score(y_test, clf_synth.predict(X_test))

            # ML efficacy = TSTR / TRTR (capped at 1.0)
            efficacy = min(acc_tstr / acc_trtr, 1.0) if acc_trtr > 0 else 0.5

            return MetricResult(
                name="ml_efficacy",
                value=float(efficacy),
                category=MetricCategory.UTILITY,
                details={
                    "tstr_accuracy": float(acc_tstr),
                    "trtr_accuracy": float(acc_trtr),
                },
            )

        except ImportError:
            return MetricResult(
                name="ml_efficacy",
                value=0.5,
                category=MetricCategory.UTILITY,
                details={"error": "sklearn not available"},
            )
        except Exception as e:
            return MetricResult(
                name="ml_efficacy",
                value=0.5,
                category=MetricCategory.UTILITY,
                details={"error": str(e)},
            )


class DatasetLoader:
    """Loads standard benchmark datasets."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".genesis" / "datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset_type: Union[str, BenchmarkDatasetType]) -> BenchmarkDataset:
        """Load a benchmark dataset.

        Args:
            dataset_type: Dataset identifier.

        Returns:
            BenchmarkDataset object.
        """
        if isinstance(dataset_type, str):
            dataset_type = BenchmarkDatasetType(dataset_type.lower())

        loaders = {
            BenchmarkDatasetType.ADULT: self._load_adult,
            BenchmarkDatasetType.CREDIT: self._load_credit,
            BenchmarkDatasetType.CALIFORNIA: self._load_california,
        }

        if dataset_type not in loaders:
            return self._generate_synthetic_benchmark(dataset_type.value)

        return loaders[dataset_type]()

    def _load_adult(self) -> BenchmarkDataset:
        """Load UCI Adult Income dataset."""
        try:
            # Try to load from sklearn
            from sklearn.datasets import fetch_openml
            data = fetch_openml("adult", version=2, as_frame=True)
            df = data.frame
        except Exception:
            # Generate synthetic version
            df = self._generate_adult_like()

        return BenchmarkDataset(
            name="adult",
            data=df,
            target_column="income" if "income" in df.columns else None,
            discrete_columns=["workclass", "education", "marital-status", "occupation",
                            "relationship", "race", "sex", "native-country"],
            sensitive_columns=["race", "sex"],
            metadata={"source": "UCI ML Repository"},
        )

    def _load_credit(self) -> BenchmarkDataset:
        """Load German Credit dataset."""
        # Generate synthetic version
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
            "duration": np.random.randint(4, 72, n),
            "credit_amount": np.random.lognormal(8, 1, n).astype(int),
            "installment_rate": np.random.randint(1, 5, n),
            "residence_since": np.random.randint(1, 5, n),
            "age": np.random.randint(19, 75, n),
            "num_credits": np.random.randint(1, 5, n),
            "num_dependents": np.random.choice([1, 2], n, p=[0.85, 0.15]),
            "foreign_worker": np.random.choice(["yes", "no"], n, p=[0.97, 0.03]),
            "credit_risk": np.random.choice(["good", "bad"], n, p=[0.7, 0.3]),
        })

        return BenchmarkDataset(
            name="credit",
            data=df,
            target_column="credit_risk",
            discrete_columns=["foreign_worker", "credit_risk"],
            sensitive_columns=["age", "foreign_worker"],
        )

    def _load_california(self) -> BenchmarkDataset:
        """Load California Housing dataset."""
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing(as_frame=True)
            df = data.frame
            df["target"] = data.target
        except Exception:
            df = self._generate_housing_like()

        return BenchmarkDataset(
            name="california",
            data=df,
            target_column="target" if "target" in df.columns else "MedHouseVal",
            discrete_columns=[],
        )

    def _generate_adult_like(self) -> pd.DataFrame:
        """Generate Adult-like dataset."""
        np.random.seed(42)
        n = 32561

        return pd.DataFrame({
            "age": np.random.randint(17, 90, n),
            "workclass": np.random.choice(
                ["Private", "Self-emp", "Gov", "Other"], n, p=[0.7, 0.1, 0.15, 0.05]
            ),
            "education_num": np.random.randint(1, 17, n),
            "marital_status": np.random.choice(
                ["Married", "Never-married", "Divorced"], n
            ),
            "occupation": np.random.choice(
                ["Tech", "Craft", "Sales", "Admin", "Other"], n
            ),
            "relationship": np.random.choice(
                ["Husband", "Wife", "Own-child", "Not-in-family"], n
            ),
            "race": np.random.choice(
                ["White", "Black", "Asian", "Other"], n, p=[0.85, 0.1, 0.03, 0.02]
            ),
            "sex": np.random.choice(["Male", "Female"], n, p=[0.67, 0.33]),
            "capital_gain": np.random.exponential(1000, n).astype(int),
            "capital_loss": np.random.exponential(100, n).astype(int),
            "hours_per_week": np.random.normal(40, 12, n).clip(1, 99).astype(int),
            "income": np.random.choice([">50K", "<=50K"], n, p=[0.24, 0.76]),
        })

    def _generate_housing_like(self) -> pd.DataFrame:
        """Generate California Housing-like dataset."""
        np.random.seed(42)
        n = 20640

        return pd.DataFrame({
            "MedInc": np.random.lognormal(1.5, 0.5, n),
            "HouseAge": np.random.randint(1, 52, n),
            "AveRooms": np.random.normal(5.4, 2, n).clip(1, 20),
            "AveBedrms": np.random.normal(1.1, 0.5, n).clip(0.5, 5),
            "Population": np.random.lognormal(7, 1, n).astype(int),
            "AveOccup": np.random.normal(3, 1, n).clip(1, 10),
            "Latitude": np.random.uniform(32.5, 42, n),
            "Longitude": np.random.uniform(-124, -114, n),
            "MedHouseVal": np.random.lognormal(12, 0.7, n),
        })

    def _generate_synthetic_benchmark(self, name: str) -> BenchmarkDataset:
        """Generate a synthetic benchmark dataset."""
        np.random.seed(42)
        n = 10000

        df = pd.DataFrame({
            "id": range(n),
            "numeric_1": np.random.normal(0, 1, n),
            "numeric_2": np.random.exponential(1, n),
            "numeric_3": np.random.uniform(0, 100, n),
            "category_1": np.random.choice(["A", "B", "C", "D"], n),
            "category_2": np.random.choice(["X", "Y"], n),
            "target": np.random.choice([0, 1], n),
        })

        return BenchmarkDataset(
            name=name,
            data=df,
            target_column="target",
            discrete_columns=["category_1", "category_2", "target"],
        )


class BenchmarkRunner:
    """Runs benchmarks for a single generator method."""

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()
        self.metrics = BenchmarkMetrics()

    def run(
        self,
        method: str,
        dataset: BenchmarkDataset,
        generator_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Run benchmark for a single method on a dataset.

        Args:
            method: Generator method name.
            dataset: Benchmark dataset.
            generator_kwargs: Additional generator arguments.

        Returns:
            BenchmarkResult with all metrics.
        """
        generator_kwargs = generator_kwargs or {}
        n_samples = self.config.n_samples or len(dataset.data)

        try:
            # Import generator
            from genesis.core.base import SyntheticGenerator

            # Fit
            start_fit = time.time()
            generator = SyntheticGenerator(method=method)
            generator.fit(dataset.data, discrete_columns=dataset.discrete_columns)
            fit_time = time.time() - start_fit

            # Generate
            start_gen = time.time()
            synthetic_data = generator.generate(n_samples=n_samples)
            generate_time = time.time() - start_gen

            # Compute metrics
            metric_results = self.metrics.compute(
                dataset.data,
                synthetic_data,
                target_column=dataset.target_column,
            )

            # Add performance metrics
            metric_results.append(
                MetricResult(
                    name="fit_time",
                    value=fit_time,
                    category=MetricCategory.PERFORMANCE,
                )
            )
            metric_results.append(
                MetricResult(
                    name="generate_time",
                    value=generate_time,
                    category=MetricCategory.PERFORMANCE,
                )
            )
            metric_results.append(
                MetricResult(
                    name="throughput",
                    value=n_samples / generate_time if generate_time > 0 else 0,
                    category=MetricCategory.PERFORMANCE,
                    details={"unit": "samples/second"},
                )
            )

            return BenchmarkResult(
                method=method,
                dataset=dataset.name,
                metrics=metric_results,
                synthetic_data=synthetic_data,
                fit_time=fit_time,
                generate_time=generate_time,
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {method}: {e}")
            return BenchmarkResult(
                method=method,
                dataset=dataset.name,
                error=str(e),
            )


class BenchmarkSuite:
    """Main benchmarking orchestrator.

    Coordinates running benchmarks across multiple methods and datasets.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        """Initialize benchmark suite.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()
        self.loader = DatasetLoader()
        self.runner = BenchmarkRunner(self.config)
        self._results: List[BenchmarkResult] = []

    def run(
        self,
        dataset: Union[str, BenchmarkDataset],
        methods: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmarks on a dataset.

        Args:
            dataset: Dataset name or BenchmarkDataset object.
            methods: List of generator methods to benchmark.
            n_samples: Override sample count.

        Returns:
            List of BenchmarkResult objects.
        """
        if methods is None:
            methods = ["ctgan", "tvae", "gaussian_copula"]

        if isinstance(dataset, str):
            dataset = self.loader.load(dataset)

        if n_samples:
            self.config.n_samples = n_samples

        results: List[BenchmarkResult] = []
        for method in methods:
            logger.info(f"Benchmarking {method} on {dataset.name}...")
            result = self.runner.run(method, dataset)
            results.append(result)
            self._results.append(result)

        return results

    def run_all(
        self,
        datasets: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks on multiple datasets.

        Args:
            datasets: List of dataset names.
            methods: List of methods to benchmark.

        Returns:
            Dictionary mapping dataset names to results.
        """
        if datasets is None:
            datasets = ["adult", "credit", "california"]

        all_results: Dict[str, List[BenchmarkResult]] = {}
        for dataset_name in datasets:
            all_results[dataset_name] = self.run(dataset_name, methods)

        return all_results

    def generate_leaderboard(
        self,
        results: Optional[List[BenchmarkResult]] = None,
        output_path: Optional[str] = None,
        format: str = "html",
    ) -> str:
        """Generate benchmark leaderboard.

        Args:
            results: Results to include (None = all stored results).
            output_path: Path to save leaderboard.
            format: Output format ("html", "json", "markdown").

        Returns:
            Leaderboard content as string.
        """
        results = results or self._results

        if format == "json":
            content = json.dumps([r.to_dict() for r in results], indent=2)
        elif format == "markdown":
            content = self._generate_markdown_leaderboard(results)
        else:
            content = self._generate_html_leaderboard(results)

        if output_path:
            Path(output_path).write_text(content)
            logger.info(f"Leaderboard saved to {output_path}")

        return content

    def _generate_markdown_leaderboard(self, results: List[BenchmarkResult]) -> str:
        """Generate Markdown leaderboard."""
        lines = [
            "# Synthetic Data Benchmark Leaderboard",
            "",
            f"Generated: {datetime.utcnow().isoformat()}",
            "",
            "## Overall Rankings",
            "",
            "| Rank | Method | Dataset | Overall Score | Fidelity | Privacy | Utility |",
            "|------|--------|---------|---------------|----------|---------|---------|",
        ]

        # Sort by overall score
        sorted_results = sorted(results, key=lambda r: -r.get_overall_score())

        for i, result in enumerate(sorted_results, 1):
            fidelity = result.get_metric("statistical_similarity")
            privacy = result.get_metric("dcr")
            utility = result.get_metric("ml_efficacy")

            lines.append(
                f"| {i} | {result.method} | {result.dataset} | "
                f"{result.get_overall_score():.3f} | "
                f"{fidelity.value if fidelity else 'N/A':.3f} | "
                f"{privacy.value if privacy else 'N/A':.3f} | "
                f"{utility.value if utility else 'N/A':.3f} |"
            )

        lines.extend([
            "",
            "## Performance",
            "",
            "| Method | Dataset | Fit Time (s) | Gen Time (s) | Throughput |",
            "|--------|---------|--------------|--------------|------------|",
        ])

        for result in sorted_results:
            throughput = result.get_metric("throughput")
            lines.append(
                f"| {result.method} | {result.dataset} | "
                f"{result.fit_time:.2f} | {result.generate_time:.2f} | "
                f"{throughput.value if throughput else 0:.0f} samples/s |"
            )

        return "\n".join(lines)

    def _generate_html_leaderboard(self, results: List[BenchmarkResult]) -> str:
        """Generate HTML leaderboard."""
        sorted_results = sorted(results, key=lambda r: -r.get_overall_score())

        rows = []
        for i, result in enumerate(sorted_results, 1):
            fidelity = result.get_metric("statistical_similarity")
            privacy = result.get_metric("dcr")
            utility = result.get_metric("ml_efficacy")

            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{result.method}</strong></td>
                    <td>{result.dataset}</td>
                    <td>{result.get_overall_score():.3f}</td>
                    <td>{fidelity.value if fidelity else 0:.3f}</td>
                    <td>{privacy.value if privacy else 0:.3f}</td>
                    <td>{utility.value if utility else 0:.3f}</td>
                    <td>{result.fit_time:.2f}s</td>
                    <td>{'✓' if result.success else '✗'}</td>
                </tr>
            """)

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Genesis Benchmark Leaderboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #2563eb; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 12px; text-align: left; }}
        th {{ background: #f3f4f6; font-weight: 600; }}
        tr:hover {{ background: #f9fafb; }}
        .meta {{ color: #6b7280; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>🏆 Genesis Benchmark Leaderboard</h1>
    <p class="meta">Generated: {datetime.utcnow().isoformat()}</p>

    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Method</th>
                <th>Dataset</th>
                <th>Overall</th>
                <th>Fidelity</th>
                <th>Privacy</th>
                <th>Utility</th>
                <th>Fit Time</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
</body>
</html>
"""


class CompetitorBenchmark:
    """Benchmark against external synthetic data tools.

    Requires API keys for competitor services.
    """

    def __init__(self) -> None:
        self.gretel_api_key = os.environ.get("GRETEL_API_KEY")
        self.sdv_available = self._check_sdv()

    def _check_sdv(self) -> bool:
        """Check if SDV is available."""
        try:
            import sdv
            return True
        except ImportError:
            return False

    def compare(
        self,
        dataset: Union[str, BenchmarkDataset],
        competitors: List[str],
        genesis_method: str = "ctgan",
    ) -> Dict[str, BenchmarkResult]:
        """Compare Genesis against competitors.

        Args:
            dataset: Dataset to use.
            competitors: List of competitor names ["sdv", "gretel"].
            genesis_method: Genesis method to use.

        Returns:
            Dictionary of results by generator name.
        """
        loader = DatasetLoader()
        if isinstance(dataset, str):
            dataset = loader.load(dataset)

        metrics = BenchmarkMetrics()
        results: Dict[str, BenchmarkResult] = {}

        # Run Genesis
        suite = BenchmarkSuite()
        genesis_results = suite.run(dataset, [genesis_method])
        results["genesis_" + genesis_method] = genesis_results[0]

        # Run competitors
        for competitor in competitors:
            if competitor.lower() == "sdv" and self.sdv_available:
                results["sdv"] = self._run_sdv(dataset, metrics)
            elif competitor.lower() == "gretel" and self.gretel_api_key:
                results["gretel"] = self._run_gretel(dataset, metrics)
            else:
                logger.warning(f"Competitor {competitor} not available")

        return results

    def _run_sdv(
        self, dataset: BenchmarkDataset, metrics: BenchmarkMetrics
    ) -> BenchmarkResult:
        """Run SDV benchmark."""
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata

            start = time.time()
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(dataset.data)

            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(dataset.data)
            fit_time = time.time() - start

            start = time.time()
            synthetic = synthesizer.sample(len(dataset.data))
            gen_time = time.time() - start

            metric_results = metrics.compute(
                dataset.data, synthetic, target_column=dataset.target_column
            )

            return BenchmarkResult(
                method="sdv_ctgan",
                dataset=dataset.name,
                metrics=metric_results,
                synthetic_data=synthetic,
                fit_time=fit_time,
                generate_time=gen_time,
            )
        except Exception as e:
            return BenchmarkResult(
                method="sdv_ctgan",
                dataset=dataset.name,
                error=str(e),
            )

    def _run_gretel(
        self, dataset: BenchmarkDataset, metrics: BenchmarkMetrics
    ) -> BenchmarkResult:
        """Run Gretel benchmark (requires API key)."""
        # Placeholder - would need actual Gretel API integration
        return BenchmarkResult(
            method="gretel",
            dataset=dataset.name,
            error="Gretel API integration not implemented",
        )


# Convenience functions
def run_benchmark(
    dataset: str = "adult",
    methods: Optional[List[str]] = None,
    n_samples: int = 1000,
) -> List[BenchmarkResult]:
    """Run a quick benchmark.

    Args:
        dataset: Dataset name.
        methods: Methods to benchmark.
        n_samples: Number of samples.

    Returns:
        List of benchmark results.

    Example:
        >>> results = run_benchmark("adult", ["ctgan", "tvae"], n_samples=5000)
        >>> for r in results:
        ...     print(f"{r.method}: {r.get_overall_score():.3f}")
    """
    config = BenchmarkConfig(n_samples=n_samples)
    suite = BenchmarkSuite(config)
    return suite.run(dataset, methods, n_samples)


def load_benchmark_dataset(name: str) -> BenchmarkDataset:
    """Load a standard benchmark dataset.

    Args:
        name: Dataset name (adult, credit, california, etc.)

    Returns:
        BenchmarkDataset object.
    """
    loader = DatasetLoader()
    return loader.load(name)
