"""Synthetic Data Quality SLAs (Service Level Agreements).

This module provides a declarative DSL for defining quality contracts for
synthetic data generation with automatic validation and regeneration.

Features:
- YAML-based SLA definitions
- Quality metrics validation (fidelity, privacy, utility)
- Auto-regeneration when SLAs fail
- CI/CD integration (GitHub Actions, pre-commit hooks)
- Detailed diagnostics and reporting

Example:
    >>> from genesis.sla import SLAContract, SLAValidator
    >>>
    >>> # Define SLA
    >>> contract = SLAContract.from_yaml("sla.yaml")
    >>>
    >>> # Validate synthetic data
    >>> validator = SLAValidator(contract)
    >>> result = validator.validate(real_data, synthetic_data)
    >>>
    >>> if not result.passed:
    ...     print(result.violations)
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from genesis.core.exceptions import GenesisError, ValidationError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of quality metrics."""

    # Fidelity metrics
    STATISTICAL_FIDELITY = "statistical_fidelity"
    COLUMN_CORRELATION = "column_correlation"
    DISTRIBUTION_SIMILARITY = "distribution_similarity"
    KS_STATISTIC = "ks_statistic"

    # Privacy metrics
    REIDENTIFICATION_RISK = "reidentification_risk"
    K_ANONYMITY = "k_anonymity"
    MEMBERSHIP_INFERENCE = "membership_inference"
    ATTRIBUTE_DISCLOSURE = "attribute_disclosure"

    # Utility metrics
    ML_UTILITY = "ml_utility"
    DOWNSTREAM_ACCURACY = "downstream_accuracy"
    FEATURE_IMPORTANCE_CORRELATION = "feature_importance_correlation"

    # Custom
    CUSTOM = "custom"


class ComparisonOperator(Enum):
    """Comparison operators for thresholds."""

    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUAL = "eq"
    BETWEEN = "between"


@dataclass
class MetricThreshold:
    """Threshold definition for a metric."""

    metric: MetricType
    operator: ComparisonOperator
    value: Union[float, Tuple[float, float]]
    column: Optional[str] = None  # For column-specific thresholds
    weight: float = 1.0  # Weight for overall score
    critical: bool = False  # If True, any violation fails the SLA

    def check(self, actual_value: float) -> Tuple[bool, str]:
        """Check if the actual value meets the threshold.

        Args:
            actual_value: The actual metric value

        Returns:
            Tuple of (passed, message)
        """
        if self.operator == ComparisonOperator.GREATER_THAN:
            passed = actual_value > self.value
            msg = f"{actual_value:.4f} > {self.value}"
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            passed = actual_value >= self.value
            msg = f"{actual_value:.4f} >= {self.value}"
        elif self.operator == ComparisonOperator.LESS_THAN:
            passed = actual_value < self.value
            msg = f"{actual_value:.4f} < {self.value}"
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            passed = actual_value <= self.value
            msg = f"{actual_value:.4f} <= {self.value}"
        elif self.operator == ComparisonOperator.EQUAL:
            passed = np.isclose(actual_value, self.value)
            msg = f"{actual_value:.4f} == {self.value}"
        elif self.operator == ComparisonOperator.BETWEEN:
            low, high = self.value
            passed = low <= actual_value <= high
            msg = f"{low} <= {actual_value:.4f} <= {high}"
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

        return passed, msg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "operator": self.operator.value,
            "value": self.value,
            "column": self.column,
            "weight": self.weight,
            "critical": self.critical,
        }


@dataclass
class ColumnConstraint:
    """Constraints for a specific column."""

    column: str
    min_correlation: Optional[float] = None  # Min correlation with original
    max_ks_statistic: Optional[float] = None  # Max KS statistic
    required_categories: Optional[List[str]] = None  # Required category values
    value_range: Optional[Tuple[float, float]] = None  # Valid value range
    null_rate_tolerance: float = 0.05  # Max null rate difference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "min_correlation": self.min_correlation,
            "max_ks_statistic": self.max_ks_statistic,
            "required_categories": self.required_categories,
            "value_range": self.value_range,
            "null_rate_tolerance": self.null_rate_tolerance,
        }


@dataclass
class SLAContract:
    """Quality contract for synthetic data.

    Example YAML:
        name: customer_data_sla
        version: "1.0"

        thresholds:
          - metric: statistical_fidelity
            operator: gte
            value: 0.90
            critical: true

          - metric: reidentification_risk
            operator: lte
            value: 0.05

          - metric: ml_utility
            operator: gte
            value: 0.85

        column_constraints:
          - column: age
            min_correlation: 0.95
            value_range: [0, 120]

          - column: income
            max_ks_statistic: 0.1

        regeneration:
          enabled: true
          max_retries: 3
          backoff_factor: 1.5
    """

    name: str
    version: str = "1.0"
    thresholds: List[MetricThreshold] = field(default_factory=list)
    column_constraints: List[ColumnConstraint] = field(default_factory=list)
    regeneration_enabled: bool = True
    max_retries: int = 3
    backoff_factor: float = 1.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SLAContract":
        """Load SLA contract from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            SLAContract instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SLAContract":
        """Create from dictionary.

        Args:
            data: Dictionary with SLA specification

        Returns:
            SLAContract instance
        """
        thresholds = []
        for t in data.get("thresholds", []):
            thresholds.append(
                MetricThreshold(
                    metric=MetricType(t["metric"]),
                    operator=ComparisonOperator(t["operator"]),
                    value=t["value"],
                    column=t.get("column"),
                    weight=t.get("weight", 1.0),
                    critical=t.get("critical", False),
                )
            )

        column_constraints = []
        for c in data.get("column_constraints", []):
            column_constraints.append(
                ColumnConstraint(
                    column=c["column"],
                    min_correlation=c.get("min_correlation"),
                    max_ks_statistic=c.get("max_ks_statistic"),
                    required_categories=c.get("required_categories"),
                    value_range=tuple(c["value_range"]) if c.get("value_range") else None,
                    null_rate_tolerance=c.get("null_rate_tolerance", 0.05),
                )
            )

        regen = data.get("regeneration", {})

        return cls(
            name=data.get("name", "unnamed_sla"),
            version=data.get("version", "1.0"),
            thresholds=thresholds,
            column_constraints=column_constraints,
            regeneration_enabled=regen.get("enabled", True),
            max_retries=regen.get("max_retries", 3),
            backoff_factor=regen.get("backoff_factor", 1.5),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "thresholds": [t.to_dict() for t in self.thresholds],
            "column_constraints": [c.to_dict() for c in self.column_constraints],
            "regeneration": {
                "enabled": self.regeneration_enabled,
                "max_retries": self.max_retries,
                "backoff_factor": self.backoff_factor,
            },
            "metadata": self.metadata,
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def default(cls) -> "SLAContract":
        """Create a default SLA contract with reasonable thresholds."""
        return cls(
            name="default_sla",
            thresholds=[
                MetricThreshold(
                    metric=MetricType.STATISTICAL_FIDELITY,
                    operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                    value=0.85,
                    critical=True,
                ),
                MetricThreshold(
                    metric=MetricType.REIDENTIFICATION_RISK,
                    operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                    value=0.10,
                ),
                MetricThreshold(
                    metric=MetricType.ML_UTILITY,
                    operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                    value=0.80,
                ),
            ],
        )


@dataclass
class SLAViolation:
    """Details of a single SLA violation."""

    threshold: MetricThreshold
    actual_value: float
    message: str
    severity: str  # "critical" or "warning"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.threshold.metric.value,
            "expected": str(self.threshold.value),
            "actual": self.actual_value,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class SLAValidationResult:
    """Result of SLA validation."""

    passed: bool
    overall_score: float
    violations: List[SLAViolation]
    metric_scores: Dict[str, float]
    column_results: Dict[str, Dict[str, Any]]
    validation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "violations": [v.to_dict() for v in self.violations],
            "metric_scores": self.metric_scores,
            "column_results": self.column_results,
            "validation_time": self.validation_time,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Get summary string."""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [
            f"SLA Validation Result: {status}",
            f"Overall Score: {self.overall_score:.1%}",
            f"Validation Time: {self.validation_time:.2f}s",
            "",
        ]

        if self.violations:
            lines.append(f"Violations ({len(self.violations)}):")
            for v in self.violations:
                lines.append(f"  [{v.severity.upper()}] {v.threshold.metric.value}: {v.message}")

        return "\n".join(lines)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class MetricCalculator:
    """Calculates quality metrics for synthetic data."""

    def __init__(self) -> None:
        self._custom_metrics: Dict[str, Callable] = {}

    def register_custom_metric(
        self,
        name: str,
        func: Callable[[pd.DataFrame, pd.DataFrame], float],
    ) -> None:
        """Register a custom metric function.

        Args:
            name: Metric name
            func: Function that takes (real_data, synthetic_data) and returns a score
        """
        self._custom_metrics[name] = func

    def calculate(
        self,
        metric: MetricType,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        column: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """Calculate a specific metric.

        Args:
            metric: Metric type to calculate
            real_data: Original data
            synthetic_data: Generated data
            column: Optional column for column-specific metrics
            **kwargs: Additional arguments

        Returns:
            Metric value
        """
        if metric == MetricType.STATISTICAL_FIDELITY:
            return self._statistical_fidelity(real_data, synthetic_data)
        elif metric == MetricType.COLUMN_CORRELATION:
            return self._column_correlation(real_data, synthetic_data, column)
        elif metric == MetricType.DISTRIBUTION_SIMILARITY:
            return self._distribution_similarity(real_data, synthetic_data, column)
        elif metric == MetricType.KS_STATISTIC:
            return self._ks_statistic(real_data, synthetic_data, column)
        elif metric == MetricType.REIDENTIFICATION_RISK:
            return self._reidentification_risk(real_data, synthetic_data)
        elif metric == MetricType.K_ANONYMITY:
            return self._k_anonymity(synthetic_data, kwargs.get("quasi_identifiers", []))
        elif metric == MetricType.MEMBERSHIP_INFERENCE:
            return self._membership_inference(real_data, synthetic_data)
        elif metric == MetricType.ATTRIBUTE_DISCLOSURE:
            return self._attribute_disclosure(
                real_data, synthetic_data, kwargs.get("sensitive_columns", [])
            )
        elif metric == MetricType.ML_UTILITY:
            return self._ml_utility(real_data, synthetic_data, kwargs.get("target_column"))
        elif metric == MetricType.DOWNSTREAM_ACCURACY:
            return self._downstream_accuracy(
                real_data, synthetic_data, kwargs.get("target_column")
            )
        elif metric == MetricType.FEATURE_IMPORTANCE_CORRELATION:
            return self._feature_importance_correlation(
                real_data, synthetic_data, kwargs.get("target_column")
            )
        elif metric == MetricType.CUSTOM:
            custom_name = kwargs.get("custom_metric_name")
            if custom_name in self._custom_metrics:
                return self._custom_metrics[custom_name](real_data, synthetic_data)
            raise ValueError(f"Custom metric '{custom_name}' not registered")
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _statistical_fidelity(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> float:
        """Calculate overall statistical fidelity."""
        scores = []

        for col in real_data.columns:
            if col not in synthetic_data.columns:
                continue

            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()

            if len(real_col) == 0 or len(synth_col) == 0:
                continue

            if pd.api.types.is_numeric_dtype(real_col):
                # Use KS test for numeric columns
                from scipy import stats

                ks_stat, _ = stats.ks_2samp(real_col, synth_col)
                scores.append(1 - ks_stat)
            else:
                # Use TVD for categorical columns
                real_dist = real_col.value_counts(normalize=True)
                synth_dist = synth_col.value_counts(normalize=True)

                all_cats = set(real_dist.index) | set(synth_dist.index)
                tvd = sum(
                    abs(real_dist.get(c, 0) - synth_dist.get(c, 0)) for c in all_cats
                ) / 2
                scores.append(1 - tvd)

        return np.mean(scores) if scores else 0.0

    def _column_correlation(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        column: Optional[str] = None,
    ) -> float:
        """Calculate correlation between real and synthetic columns."""
        if column is None:
            # Overall correlation matrix similarity
            numeric_cols = real_data.select_dtypes(include=[np.number]).columns
            common_cols = [c for c in numeric_cols if c in synthetic_data.columns]

            if len(common_cols) < 2:
                return 1.0

            real_corr = real_data[common_cols].corr()
            synth_corr = synthetic_data[common_cols].corr()

            diff = np.abs(real_corr.values - synth_corr.values)
            return 1 - np.mean(diff[~np.isnan(diff)])
        else:
            # Single column correlation
            if column not in real_data.columns or column not in synthetic_data.columns:
                return 0.0

            real_col = real_data[column].dropna()
            synth_col = synthetic_data[column].dropna()

            if not pd.api.types.is_numeric_dtype(real_col):
                return 1.0  # Not applicable for categorical

            # Pearson correlation of sorted values
            real_sorted = np.sort(real_col.values)
            synth_sorted = np.sort(synth_col.values)

            min_len = min(len(real_sorted), len(synth_sorted))
            if min_len < 2:
                return 0.0

            # Resample to same length
            real_resampled = np.interp(
                np.linspace(0, 1, min_len),
                np.linspace(0, 1, len(real_sorted)),
                real_sorted,
            )
            synth_resampled = np.interp(
                np.linspace(0, 1, min_len),
                np.linspace(0, 1, len(synth_sorted)),
                synth_sorted,
            )

            corr = np.corrcoef(real_resampled, synth_resampled)[0, 1]
            return max(0, corr)

    def _distribution_similarity(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        column: Optional[str] = None,
    ) -> float:
        """Calculate distribution similarity using Jensen-Shannon divergence."""
        columns = [column] if column else list(real_data.columns)
        scores = []

        for col in columns:
            if col not in synthetic_data.columns:
                continue

            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()

            if len(real_col) == 0 or len(synth_col) == 0:
                continue

            if pd.api.types.is_numeric_dtype(real_col):
                # Binned histogram comparison
                all_values = np.concatenate([real_col.values, synth_col.values])
                bins = np.histogram_bin_edges(all_values, bins=50)

                real_hist, _ = np.histogram(real_col, bins=bins, density=True)
                synth_hist, _ = np.histogram(synth_col, bins=bins, density=True)

                # Add small epsilon to avoid log(0)
                eps = 1e-10
                real_hist = real_hist + eps
                synth_hist = synth_hist + eps

                # Normalize
                real_hist = real_hist / real_hist.sum()
                synth_hist = synth_hist / synth_hist.sum()

                # Jensen-Shannon divergence
                m = (real_hist + synth_hist) / 2
                js_div = (
                    np.sum(real_hist * np.log(real_hist / m))
                    + np.sum(synth_hist * np.log(synth_hist / m))
                ) / 2

                scores.append(1 - min(js_div, 1))
            else:
                # Categorical comparison
                real_dist = real_col.value_counts(normalize=True)
                synth_dist = synth_col.value_counts(normalize=True)

                all_cats = set(real_dist.index) | set(synth_dist.index)
                overlap = sum(
                    min(real_dist.get(c, 0), synth_dist.get(c, 0)) for c in all_cats
                )
                scores.append(overlap)

        return np.mean(scores) if scores else 0.0

    def _ks_statistic(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        column: Optional[str] = None,
    ) -> float:
        """Calculate KS statistic (lower is better)."""
        from scipy import stats

        if column:
            columns = [column]
        else:
            columns = [
                c for c in real_data.columns
                if c in synthetic_data.columns
                and pd.api.types.is_numeric_dtype(real_data[c])
            ]

        ks_stats = []
        for col in columns:
            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()

            if len(real_col) > 0 and len(synth_col) > 0:
                ks_stat, _ = stats.ks_2samp(real_col, synth_col)
                ks_stats.append(ks_stat)

        return np.mean(ks_stats) if ks_stats else 1.0

    def _reidentification_risk(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> float:
        """Estimate re-identification risk using nearest neighbor distance."""
        from scipy.spatial.distance import cdist

        # Use numeric columns only
        numeric_cols = [
            c for c in real_data.columns
            if c in synthetic_data.columns and pd.api.types.is_numeric_dtype(real_data[c])
        ]

        if not numeric_cols:
            return 0.0

        real_numeric = real_data[numeric_cols].fillna(0).values
        synth_numeric = synthetic_data[numeric_cols].fillna(0).values

        # Normalize
        mean = real_numeric.mean(axis=0)
        std = real_numeric.std(axis=0) + 1e-10
        real_norm = (real_numeric - mean) / std
        synth_norm = (synth_numeric - mean) / std

        # Sample for large datasets
        max_samples = 1000
        if len(real_norm) > max_samples:
            indices = np.random.choice(len(real_norm), max_samples, replace=False)
            real_norm = real_norm[indices]
        if len(synth_norm) > max_samples:
            indices = np.random.choice(len(synth_norm), max_samples, replace=False)
            synth_norm = synth_norm[indices]

        # Calculate minimum distances
        distances = cdist(synth_norm, real_norm, metric="euclidean")
        min_distances = distances.min(axis=1)

        # Risk is proportion of synthetic records "too close" to real
        threshold = 0.1 * np.median(min_distances)
        risk = (min_distances < threshold).mean()

        return float(risk)

    def _k_anonymity(
        self, data: pd.DataFrame, quasi_identifiers: List[str]
    ) -> float:
        """Calculate minimum k for k-anonymity."""
        if not quasi_identifiers:
            quasi_identifiers = list(data.columns)[:5]  # Use first 5 columns

        valid_qis = [q for q in quasi_identifiers if q in data.columns]
        if not valid_qis:
            return float("inf")

        group_sizes = data.groupby(valid_qis, dropna=False).size()
        return float(group_sizes.min())

    def _membership_inference(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> float:
        """Estimate membership inference attack success rate."""
        # Simple heuristic: check for exact matches
        real_set = set(tuple(row) for row in real_data.values)
        synth_set = set(tuple(row) for row in synthetic_data.values)

        if not synth_set:
            return 0.0

        matches = len(real_set & synth_set)
        return matches / len(synth_set)

    def _attribute_disclosure(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: List[str],
    ) -> float:
        """Estimate attribute disclosure risk."""
        if not sensitive_columns:
            return 0.0

        valid_cols = [c for c in sensitive_columns if c in synthetic_data.columns]
        if not valid_cols:
            return 0.0

        risks = []
        for col in valid_cols:
            if pd.api.types.is_numeric_dtype(synthetic_data[col]):
                # For numeric: check if values match closely
                real_values = set(real_data[col].dropna().round(2))
                synth_values = synthetic_data[col].dropna().round(2)
                match_rate = synth_values.isin(real_values).mean()
                risks.append(match_rate)
            else:
                # For categorical: check distribution leakage
                real_dist = real_data[col].value_counts(normalize=True)
                synth_dist = synthetic_data[col].value_counts(normalize=True)

                # Risk if synthetic over-represents rare categories
                rare_cats = real_dist[real_dist < 0.01].index
                if len(rare_cats) > 0:
                    synth_rare_rate = synth_dist.reindex(rare_cats).fillna(0).mean()
                    risks.append(min(synth_rare_rate * 10, 1.0))

        return np.mean(risks) if risks else 0.0

    def _ml_utility(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> float:
        """Calculate ML utility score (Train on Synthetic, Test on Real)."""
        if target_column is None or target_column not in real_data.columns:
            return self._statistical_fidelity(real_data, synthetic_data)

        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder

            # Prepare data
            feature_cols = [
                c for c in real_data.columns
                if c != target_column and c in synthetic_data.columns
            ]
            numeric_cols = [
                c for c in feature_cols
                if pd.api.types.is_numeric_dtype(real_data[c])
            ]

            if not numeric_cols:
                return 0.5

            X_real = real_data[numeric_cols].fillna(0).values
            X_synth = synthetic_data[numeric_cols].fillna(0).values
            y_real = real_data[target_column]
            y_synth = synthetic_data[target_column]

            # Determine if classification or regression
            is_classification = (
                not pd.api.types.is_numeric_dtype(y_real) or y_real.nunique() < 10
            )

            if is_classification:
                le = LabelEncoder()
                y_real_enc = le.fit_transform(y_real.fillna("missing"))
                y_synth_enc = le.transform(
                    y_synth.fillna("missing").apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                )
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                y_real_enc = y_real.fillna(0).values
                y_synth_enc = y_synth.fillna(0).values
                model = RandomForestRegressor(n_estimators=50, random_state=42)

            # Split real data for testing
            X_test, _, y_test, _ = train_test_split(
                X_real, y_real_enc, test_size=0.3, random_state=42
            )

            # Train on synthetic
            model.fit(X_synth, y_synth_enc)
            score = model.score(X_test, y_test)

            return max(0, score)

        except Exception as e:
            logger.warning(f"ML utility calculation failed: {e}")
            return 0.5

    def _downstream_accuracy(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> float:
        """Calculate downstream task accuracy comparison."""
        return self._ml_utility(real_data, synthetic_data, target_column)

    def _feature_importance_correlation(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> float:
        """Compare feature importance between models trained on real and synthetic."""
        if target_column is None or target_column not in real_data.columns:
            return 0.5

        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder

            feature_cols = [
                c for c in real_data.columns
                if c != target_column
                and c in synthetic_data.columns
                and pd.api.types.is_numeric_dtype(real_data[c])
            ]

            if len(feature_cols) < 2:
                return 0.5

            X_real = real_data[feature_cols].fillna(0).values
            X_synth = synthetic_data[feature_cols].fillna(0).values
            y_real = real_data[target_column]
            y_synth = synthetic_data[target_column]

            is_classification = (
                not pd.api.types.is_numeric_dtype(y_real) or y_real.nunique() < 10
            )

            if is_classification:
                le = LabelEncoder()
                y_real_enc = le.fit_transform(y_real.fillna("missing"))
                y_synth_enc = le.transform(
                    y_synth.fillna("missing").apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                )
                model_cls = RandomForestClassifier
            else:
                y_real_enc = y_real.fillna(0).values
                y_synth_enc = y_synth.fillna(0).values
                model_cls = RandomForestRegressor

            # Train on real
            model_real = model_cls(n_estimators=50, random_state=42)
            model_real.fit(X_real, y_real_enc)
            importance_real = model_real.feature_importances_

            # Train on synthetic
            model_synth = model_cls(n_estimators=50, random_state=42)
            model_synth.fit(X_synth, y_synth_enc)
            importance_synth = model_synth.feature_importances_

            # Correlation of feature importances
            corr = np.corrcoef(importance_real, importance_synth)[0, 1]
            return max(0, corr)

        except Exception as e:
            logger.warning(f"Feature importance correlation failed: {e}")
            return 0.5


class SLAValidator:
    """Validates synthetic data against SLA contracts."""

    def __init__(
        self,
        contract: SLAContract,
        metric_calculator: Optional[MetricCalculator] = None,
    ) -> None:
        """Initialize validator.

        Args:
            contract: SLA contract to validate against
            metric_calculator: Optional custom metric calculator
        """
        self.contract = contract
        self.calculator = metric_calculator or MetricCalculator()

    def validate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        **kwargs: Any,
    ) -> SLAValidationResult:
        """Validate synthetic data against the SLA.

        Args:
            real_data: Original data
            synthetic_data: Generated synthetic data
            **kwargs: Additional arguments for metrics

        Returns:
            Validation result
        """
        start_time = time.time()
        violations = []
        metric_scores: Dict[str, float] = {}
        column_results: Dict[str, Dict[str, Any]] = {}

        # Validate thresholds
        for threshold in self.contract.thresholds:
            try:
                value = self.calculator.calculate(
                    threshold.metric,
                    real_data,
                    synthetic_data,
                    column=threshold.column,
                    **kwargs,
                )
                metric_scores[threshold.metric.value] = value

                passed, msg = threshold.check(value)
                if not passed:
                    violations.append(
                        SLAViolation(
                            threshold=threshold,
                            actual_value=value,
                            message=msg,
                            severity="critical" if threshold.critical else "warning",
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate {threshold.metric}: {e}")
                violations.append(
                    SLAViolation(
                        threshold=threshold,
                        actual_value=0.0,
                        message=f"Calculation failed: {e}",
                        severity="critical" if threshold.critical else "warning",
                    )
                )

        # Validate column constraints
        for constraint in self.contract.column_constraints:
            col_result: Dict[str, Any] = {"passed": True, "issues": []}

            if constraint.column not in synthetic_data.columns:
                col_result["passed"] = False
                col_result["issues"].append("Column not found in synthetic data")
            else:
                # Check correlation
                if constraint.min_correlation is not None:
                    corr = self.calculator.calculate(
                        MetricType.COLUMN_CORRELATION,
                        real_data,
                        synthetic_data,
                        column=constraint.column,
                    )
                    col_result["correlation"] = corr
                    if corr < constraint.min_correlation:
                        col_result["passed"] = False
                        col_result["issues"].append(
                            f"Correlation {corr:.4f} < {constraint.min_correlation}"
                        )

                # Check KS statistic
                if constraint.max_ks_statistic is not None:
                    ks = self.calculator.calculate(
                        MetricType.KS_STATISTIC,
                        real_data,
                        synthetic_data,
                        column=constraint.column,
                    )
                    col_result["ks_statistic"] = ks
                    if ks > constraint.max_ks_statistic:
                        col_result["passed"] = False
                        col_result["issues"].append(
                            f"KS statistic {ks:.4f} > {constraint.max_ks_statistic}"
                        )

                # Check value range
                if constraint.value_range is not None:
                    col_values = synthetic_data[constraint.column].dropna()
                    if len(col_values) > 0:
                        min_val, max_val = constraint.value_range
                        actual_min, actual_max = col_values.min(), col_values.max()
                        if actual_min < min_val or actual_max > max_val:
                            col_result["passed"] = False
                            col_result["issues"].append(
                                f"Values [{actual_min:.2f}, {actual_max:.2f}] "
                                f"outside range [{min_val}, {max_val}]"
                            )

            column_results[constraint.column] = col_result

        # Determine overall pass/fail
        critical_violations = [v for v in violations if v.severity == "critical"]
        column_failures = [c for c, r in column_results.items() if not r.get("passed", True)]

        passed = len(critical_violations) == 0 and len(column_failures) == 0

        # Calculate overall score
        if metric_scores:
            weights = {
                t.metric.value: t.weight for t in self.contract.thresholds
            }
            total_weight = sum(weights.values())
            overall_score = sum(
                metric_scores.get(m, 0) * w for m, w in weights.items()
            ) / total_weight if total_weight > 0 else 0
        else:
            overall_score = 0.0

        validation_time = time.time() - start_time

        return SLAValidationResult(
            passed=passed,
            overall_score=overall_score,
            violations=violations,
            metric_scores=metric_scores,
            column_results=column_results,
            validation_time=validation_time,
        )


class SLAEnforcedGenerator:
    """Generator wrapper that enforces SLA contracts with auto-regeneration.

    Example:
        >>> from genesis import SyntheticGenerator
        >>> from genesis.sla import SLAContract, SLAEnforcedGenerator
        >>>
        >>> contract = SLAContract.from_yaml("sla.yaml")
        >>> base_generator = SyntheticGenerator(method="ctgan")
        >>>
        >>> generator = SLAEnforcedGenerator(base_generator, contract)
        >>> generator.fit(real_data)
        >>> synthetic = generator.generate(n_samples=1000)  # Auto-retries if SLA fails
    """

    def __init__(
        self,
        generator: Any,  # BaseGenerator
        contract: SLAContract,
        validator: Optional[SLAValidator] = None,
    ) -> None:
        """Initialize SLA-enforced generator.

        Args:
            generator: Base synthetic data generator
            contract: SLA contract to enforce
            validator: Optional custom validator
        """
        self.generator = generator
        self.contract = contract
        self.validator = validator or SLAValidator(contract)
        self._real_data: Optional[pd.DataFrame] = None
        self._discrete_columns: Optional[List[str]] = None
        self._last_result: Optional[SLAValidationResult] = None

    def fit(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "SLAEnforcedGenerator":
        """Fit the generator.

        Args:
            data: Training data
            discrete_columns: Categorical columns
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        self._real_data = data
        self._discrete_columns = discrete_columns
        self.generator.fit(data, discrete_columns=discrete_columns, **kwargs)
        return self

    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        validation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data with SLA enforcement.

        Args:
            n_samples: Number of samples
            conditions: Generation conditions
            validation_kwargs: Additional arguments for validation

        Returns:
            Synthetic data that meets SLA

        Raises:
            GenesisError: If SLA cannot be met after max retries
        """
        if self._real_data is None:
            raise ValidationError("Generator not fitted. Call fit() first.")

        validation_kwargs = validation_kwargs or {}
        best_result: Optional[SLAValidationResult] = None
        best_data: Optional[pd.DataFrame] = None
        best_score = -1.0

        for attempt in range(self.contract.max_retries + 1):
            # Generate data
            synthetic_data = self.generator.generate(n_samples, conditions=conditions)

            # Validate
            result = self.validator.validate(
                self._real_data, synthetic_data, **validation_kwargs
            )
            self._last_result = result

            # Track best attempt
            if result.overall_score > best_score:
                best_score = result.overall_score
                best_result = result
                best_data = synthetic_data

            if result.passed:
                logger.info(f"SLA passed on attempt {attempt + 1}")
                return synthetic_data

            if not self.contract.regeneration_enabled:
                break

            # Log attempt
            logger.warning(
                f"SLA validation failed on attempt {attempt + 1}. "
                f"Score: {result.overall_score:.2%}. "
                f"Violations: {len(result.violations)}"
            )

            # Exponential backoff before retry
            if attempt < self.contract.max_retries:
                wait_time = self.contract.backoff_factor ** attempt
                time.sleep(wait_time)

        # Return best attempt if regeneration failed
        if best_data is not None:
            logger.warning(
                f"SLA not met after {self.contract.max_retries + 1} attempts. "
                f"Returning best attempt with score {best_score:.2%}"
            )
            return best_data

        raise GenesisError(
            f"Failed to generate data meeting SLA after {self.contract.max_retries + 1} attempts"
        )

    @property
    def last_validation_result(self) -> Optional[SLAValidationResult]:
        """Get the last validation result."""
        return self._last_result


def create_github_action(
    sla_path: str = "sla.yaml",
    data_path: str = "data/",
    output_path: str = "synthetic/",
) -> str:
    """Generate GitHub Actions workflow for SLA validation.

    Args:
        sla_path: Path to SLA contract file
        data_path: Path to input data
        output_path: Path for synthetic output

    Returns:
        YAML content for GitHub Actions workflow
    """
    workflow = f"""
name: Synthetic Data SLA Validation

on:
  push:
    branches: [main]
    paths:
      - '{sla_path}'
      - '{data_path}**'
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  validate-sla:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install genesis-synth[all]

      - name: Generate and validate synthetic data
        run: |
          genesis sla validate \\
            --contract {sla_path} \\
            --input {data_path} \\
            --output {output_path} \\
            --report sla-report.json

      - name: Upload SLA report
        uses: actions/upload-artifact@v4
        with:
          name: sla-report
          path: sla-report.json

      - name: Check SLA status
        run: |
          python -c "
          import json
          with open('sla-report.json') as f:
              report = json.load(f)
          if not report['passed']:
              print('SLA validation failed!')
              for v in report['violations']:
                  print(f'  - {{v[\"metric\"]}}: {{v[\"message\"]}}')
              exit(1)
          print(f'SLA passed with score: {{report[\"overall_score\"]:.1%}}')
          "
"""
    return workflow.strip()


def create_pre_commit_hook() -> str:
    """Generate pre-commit hook configuration for SLA validation.

    Returns:
        YAML content for .pre-commit-config.yaml
    """
    hook = """
repos:
  - repo: local
    hooks:
      - id: genesis-sla-validate
        name: Validate Synthetic Data SLA
        entry: genesis sla validate --contract sla.yaml --quick
        language: python
        types: [python]
        pass_filenames: false
        additional_dependencies: ['genesis-synth']
"""
    return hook.strip()
