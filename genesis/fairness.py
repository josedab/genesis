"""Fairness-Aware Synthetic Data Generation.

This module provides tools for generating synthetic data that addresses
and corrects historical biases, supporting fairness in ML pipelines.

Features:
- Bias detection and measurement
- Demographic parity correction
- Equal opportunity generation
- Counterfactual data augmentation
- Fairness constraints during generation

Example:
    >>> from genesis.fairness import FairnessAnalyzer, FairGenerator
    >>>
    >>> analyzer = FairnessAnalyzer(
    ...     sensitive_attributes=["gender", "race"],
    ...     outcome_column="income"
    ... )
    >>> bias_report = analyzer.analyze(original_df)
    >>> print(bias_report.demographic_parity_ratio)
    >>>
    >>> generator = FairGenerator(
    ...     fairness_constraint="demographic_parity",
    ...     target_ratio=1.0
    ... )
    >>> fair_data = generator.generate(original_df, n_samples=10000)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.exceptions import ConfigurationError, GenesisError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class FairnessMetric(Enum):
    """Fairness metrics supported by the system."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"


@dataclass
class BiasMetrics:
    """Container for bias measurement results."""

    metric_name: str
    value: float
    privileged_value: float
    unprivileged_value: float
    threshold: float = 0.8
    is_fair: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute fairness status."""
        if self.value >= self.threshold:
            self.is_fair = True


@dataclass
class BiasReport:
    """Comprehensive bias analysis report."""

    sensitive_attribute: str
    outcome_column: str
    metrics: Dict[str, BiasMetrics] = field(default_factory=dict)
    group_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sensitive_attribute": self.sensitive_attribute,
            "outcome_column": self.outcome_column,
            "metrics": {k: v.__dict__ for k, v in self.metrics.items()},
            "group_statistics": self.group_statistics,
            "recommendations": self.recommendations,
        }


class FairnessAnalyzer:
    """Analyzes datasets for bias and fairness issues.

    Example:
        >>> analyzer = FairnessAnalyzer(
        ...     sensitive_attributes=["gender"],
        ...     outcome_column="hired",
        ...     privileged_groups={"gender": "male"}
        ... )
        >>> report = analyzer.analyze(df)
        >>> print(report.metrics["demographic_parity"].value)
    """

    def __init__(
        self,
        sensitive_attributes: List[str],
        outcome_column: str,
        privileged_groups: Optional[Dict[str, Any]] = None,
        positive_outcome: Any = 1,
        fairness_threshold: float = 0.8,
    ) -> None:
        """Initialize fairness analyzer.

        Args:
            sensitive_attributes: Columns representing sensitive attributes
            outcome_column: Column with the outcome to analyze
            privileged_groups: Mapping of attribute to privileged value
            positive_outcome: Value representing positive outcome
            fairness_threshold: Minimum ratio for fairness (0.8 = 80%)
        """
        self.sensitive_attributes = sensitive_attributes
        self.outcome_column = outcome_column
        self.privileged_groups = privileged_groups or {}
        self.positive_outcome = positive_outcome
        self.fairness_threshold = fairness_threshold

    def analyze(self, data: pd.DataFrame) -> Dict[str, BiasReport]:
        """Analyze dataset for bias across all sensitive attributes.

        Args:
            data: DataFrame to analyze

        Returns:
            Dictionary mapping attribute names to BiasReports
        """
        reports = {}

        for attr in self.sensitive_attributes:
            if attr not in data.columns:
                logger.warning(f"Sensitive attribute '{attr}' not found in data")
                continue

            report = self._analyze_attribute(data, attr)
            reports[attr] = report

        return reports

    def _analyze_attribute(self, data: pd.DataFrame, attr: str) -> BiasReport:
        """Analyze bias for a single sensitive attribute."""
        report = BiasReport(
            sensitive_attribute=attr,
            outcome_column=self.outcome_column,
        )

        # Get groups
        groups = data[attr].unique()
        privileged = self.privileged_groups.get(attr, groups[0])

        # Calculate group statistics
        for group in groups:
            mask = data[attr] == group
            group_data = data[mask]
            positive_rate = (group_data[self.outcome_column] == self.positive_outcome).mean()
            report.group_statistics[str(group)] = {
                "count": len(group_data),
                "positive_rate": positive_rate,
                "positive_count": (group_data[self.outcome_column] == self.positive_outcome).sum(),
            }

        # Calculate metrics
        report.metrics["demographic_parity"] = self._demographic_parity(data, attr, privileged)
        report.metrics["disparate_impact"] = self._disparate_impact(data, attr, privileged)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _demographic_parity(
        self, data: pd.DataFrame, attr: str, privileged: Any
    ) -> BiasMetrics:
        """Calculate demographic parity ratio."""
        privileged_mask = data[attr] == privileged
        unprivileged_mask = ~privileged_mask

        priv_rate = (data[privileged_mask][self.outcome_column] == self.positive_outcome).mean()
        unpriv_rate = (data[unprivileged_mask][self.outcome_column] == self.positive_outcome).mean()

        ratio = unpriv_rate / priv_rate if priv_rate > 0 else 0

        return BiasMetrics(
            metric_name="demographic_parity",
            value=ratio,
            privileged_value=priv_rate,
            unprivileged_value=unpriv_rate,
            threshold=self.fairness_threshold,
        )

    def _disparate_impact(
        self, data: pd.DataFrame, attr: str, privileged: Any
    ) -> BiasMetrics:
        """Calculate disparate impact ratio (80% rule)."""
        privileged_mask = data[attr] == privileged
        unprivileged_mask = ~privileged_mask

        priv_rate = (data[privileged_mask][self.outcome_column] == self.positive_outcome).mean()
        unpriv_rate = (data[unprivileged_mask][self.outcome_column] == self.positive_outcome).mean()

        ratio = unpriv_rate / priv_rate if priv_rate > 0 else 0

        return BiasMetrics(
            metric_name="disparate_impact",
            value=ratio,
            privileged_value=priv_rate,
            unprivileged_value=unpriv_rate,
            threshold=0.8,  # 80% rule threshold
        )

    def _generate_recommendations(self, report: BiasReport) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []

        dp = report.metrics.get("demographic_parity")
        if dp and not dp.is_fair:
            if dp.value < 0.5:
                recommendations.append(
                    f"Severe bias detected: {report.sensitive_attribute} has "
                    f"demographic parity ratio of {dp.value:.2f}. Consider "
                    "resampling or counterfactual augmentation."
                )
            else:
                recommendations.append(
                    f"Moderate bias detected: {report.sensitive_attribute} has "
                    f"demographic parity ratio of {dp.value:.2f}. Consider "
                    "re-weighting or fair representation learning."
                )

        di = report.metrics.get("disparate_impact")
        if di and di.value < 0.8:
            recommendations.append(
                f"Disparate impact violation: ratio {di.value:.2f} is below "
                "the 80% threshold. This may have legal implications."
            )

        return recommendations


class FairnessConstraint(ABC):
    """Abstract base class for fairness constraints."""

    @abstractmethod
    def check(self, data: pd.DataFrame) -> bool:
        """Check if data satisfies the constraint."""
        pass

    @abstractmethod
    def compute_weight(self, row: pd.Series) -> float:
        """Compute sampling weight for a row."""
        pass


class DemographicParityConstraint(FairnessConstraint):
    """Constraint ensuring demographic parity across groups."""

    def __init__(
        self,
        sensitive_attr: str,
        outcome_col: str,
        target_ratio: float = 1.0,
        tolerance: float = 0.05,
    ) -> None:
        self.sensitive_attr = sensitive_attr
        self.outcome_col = outcome_col
        self.target_ratio = target_ratio
        self.tolerance = tolerance
        self._group_rates: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame) -> "DemographicParityConstraint":
        """Fit constraint to data to learn group rates."""
        for group in data[self.sensitive_attr].unique():
            mask = data[self.sensitive_attr] == group
            self._group_rates[str(group)] = data[mask][self.outcome_col].mean()
        return self

    def check(self, data: pd.DataFrame) -> bool:
        """Check if demographic parity is satisfied."""
        rates = []
        for group in data[self.sensitive_attr].unique():
            mask = data[self.sensitive_attr] == group
            rates.append(data[mask][self.outcome_col].mean())

        if len(rates) < 2:
            return True

        min_rate, max_rate = min(rates), max(rates)
        ratio = min_rate / max_rate if max_rate > 0 else 0

        return abs(ratio - self.target_ratio) <= self.tolerance

    def compute_weight(self, row: pd.Series) -> float:
        """Compute weight to achieve demographic parity."""
        group = str(row[self.sensitive_attr])
        group_rate = self._group_rates.get(group, 0.5)
        avg_rate = np.mean(list(self._group_rates.values()))

        if group_rate == 0:
            return 1.0

        return avg_rate / group_rate


class EqualOpportunityConstraint(FairnessConstraint):
    """Constraint for equal opportunity (equal TPR across groups)."""

    def __init__(
        self,
        sensitive_attr: str,
        outcome_col: str,
        predicted_col: str,
        tolerance: float = 0.05,
    ) -> None:
        self.sensitive_attr = sensitive_attr
        self.outcome_col = outcome_col
        self.predicted_col = predicted_col
        self.tolerance = tolerance
        self._group_tpr: Dict[str, float] = {}

    def fit(self, data: pd.DataFrame) -> "EqualOpportunityConstraint":
        """Fit constraint to learn group TPRs."""
        for group in data[self.sensitive_attr].unique():
            mask = (data[self.sensitive_attr] == group) & (data[self.outcome_col] == 1)
            if mask.sum() > 0:
                self._group_tpr[str(group)] = data[mask][self.predicted_col].mean()
        return self

    def check(self, data: pd.DataFrame) -> bool:
        """Check if equal opportunity is satisfied."""
        tprs = []
        for group in data[self.sensitive_attr].unique():
            mask = (data[self.sensitive_attr] == group) & (data[self.outcome_col] == 1)
            if mask.sum() > 0:
                tprs.append(data[mask][self.predicted_col].mean())

        if len(tprs) < 2:
            return True

        return max(tprs) - min(tprs) <= self.tolerance

    def compute_weight(self, row: pd.Series) -> float:
        """Compute weight for equal opportunity."""
        if row[self.outcome_col] != 1:
            return 1.0

        group = str(row[self.sensitive_attr])
        group_tpr = self._group_tpr.get(group, 0.5)
        avg_tpr = np.mean(list(self._group_tpr.values())) if self._group_tpr else 0.5

        if group_tpr == 0:
            return 1.0

        return avg_tpr / group_tpr


class FairGenerator:
    """Generates synthetic data with fairness constraints.

    Supports multiple strategies:
    - Resampling: Oversample underrepresented groups
    - Reweighting: Assign weights to balance outcomes
    - Counterfactual: Generate counterfactual examples
    - Constrained: Generate with fairness constraints

    Example:
        >>> generator = FairGenerator(
        ...     sensitive_attr="gender",
        ...     outcome_col="hired",
        ...     strategy="resampling"
        ... )
        >>> fair_data = generator.generate(biased_df, n_samples=10000)
    """

    def __init__(
        self,
        sensitive_attr: str,
        outcome_col: Optional[str] = None,
        strategy: str = "resampling",
        target_ratio: float = 1.0,
        constraint: Optional[FairnessConstraint] = None,
    ) -> None:
        """Initialize fair generator.

        Args:
            sensitive_attr: Sensitive attribute column
            outcome_col: Outcome column (required for some strategies)
            strategy: One of 'resampling', 'reweighting', 'counterfactual'
            target_ratio: Target fairness ratio
            constraint: Custom fairness constraint
        """
        self.sensitive_attr = sensitive_attr
        self.outcome_col = outcome_col
        self.strategy = strategy
        self.target_ratio = target_ratio
        self.constraint = constraint

    def generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate fair synthetic data.

        Args:
            data: Original (possibly biased) data
            n_samples: Number of samples to generate
            random_state: Random seed

        Returns:
            Fair synthetic DataFrame
        """
        rng = np.random.default_rng(random_state)

        if self.strategy == "resampling":
            return self._resample(data, n_samples, rng)
        elif self.strategy == "reweighting":
            return self._reweight(data, n_samples, rng)
        elif self.strategy == "counterfactual":
            return self._counterfactual(data, n_samples, rng)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _resample(
        self, data: pd.DataFrame, n_samples: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Generate fair data via stratified resampling."""
        groups = data[self.sensitive_attr].unique()
        samples_per_group = n_samples // len(groups)
        remainder = n_samples % len(groups)

        parts = []
        for i, group in enumerate(groups):
            group_data = data[data[self.sensitive_attr] == group]
            group_n = samples_per_group + (1 if i < remainder else 0)

            if self.outcome_col:
                # Balance outcomes within group
                positive = group_data[group_data[self.outcome_col] == 1]
                negative = group_data[group_data[self.outcome_col] == 0]

                pos_n = group_n // 2
                neg_n = group_n - pos_n

                pos_sample = positive.sample(
                    n=pos_n, replace=True, random_state=int(rng.integers(2**31))
                )
                neg_sample = negative.sample(
                    n=neg_n, replace=True, random_state=int(rng.integers(2**31))
                )

                parts.extend([pos_sample, neg_sample])
            else:
                sample = group_data.sample(
                    n=group_n, replace=True, random_state=int(rng.integers(2**31))
                )
                parts.append(sample)

        result = pd.concat(parts, ignore_index=True)
        return result.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)

    def _reweight(
        self, data: pd.DataFrame, n_samples: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Generate fair data via reweighting."""
        weights = self._compute_weights(data)
        weights = weights / weights.sum()

        indices = rng.choice(
            len(data), size=n_samples, replace=True, p=weights
        )

        return data.iloc[indices].reset_index(drop=True)

    def _compute_weights(self, data: pd.DataFrame) -> np.ndarray:
        """Compute sampling weights for fairness."""
        weights = np.ones(len(data))
        groups = data[self.sensitive_attr].unique()

        # Target equal representation
        target_prop = 1.0 / len(groups)

        for group in groups:
            mask = (data[self.sensitive_attr] == group).values
            current_prop = mask.sum() / len(data)

            if current_prop > 0:
                weight_multiplier = target_prop / current_prop
                weights[mask] *= weight_multiplier

        if self.outcome_col:
            # Also balance outcomes
            for group in groups:
                for outcome in [0, 1]:
                    mask = (
                        (data[self.sensitive_attr] == group)
                        & (data[self.outcome_col] == outcome)
                    ).values

                    if mask.sum() > 0:
                        target = 0.5 * target_prop
                        current = mask.sum() / len(data)
                        weights[mask] *= target / current if current > 0 else 1.0

        return weights

    def _counterfactual(
        self, data: pd.DataFrame, n_samples: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Generate counterfactual augmented data."""
        groups = list(data[self.sensitive_attr].unique())

        # Start with resampled base
        base_samples = n_samples // 2
        base = self._resample(data, base_samples, rng)

        # Generate counterfactuals for remaining samples
        counterfactual_n = n_samples - base_samples
        counterfactuals = []

        for _ in range(counterfactual_n):
            # Sample a random row
            idx = rng.integers(len(data))
            row = data.iloc[idx].copy()

            # Flip the sensitive attribute
            current_group = row[self.sensitive_attr]
            other_groups = [g for g in groups if g != current_group]

            if other_groups:
                row[self.sensitive_attr] = rng.choice(other_groups)

            counterfactuals.append(row)

        cf_df = pd.DataFrame(counterfactuals)
        return pd.concat([base, cf_df], ignore_index=True)


class CounterfactualGenerator:
    """Generates counterfactual examples for fairness analysis.

    Counterfactuals answer: "What would have happened if the sensitive
    attribute had been different?"

    Example:
        >>> generator = CounterfactualGenerator(
        ...     sensitive_attr="gender",
        ...     causal_features=["education", "experience"]
        ... )
        >>> counterfactuals = generator.generate(df)
    """

    def __init__(
        self,
        sensitive_attr: str,
        causal_features: Optional[List[str]] = None,
        preserve_features: Optional[List[str]] = None,
    ) -> None:
        """Initialize counterfactual generator.

        Args:
            sensitive_attr: Sensitive attribute to flip
            causal_features: Features causally affected by sensitive attr
            preserve_features: Features to keep unchanged
        """
        self.sensitive_attr = sensitive_attr
        self.causal_features = causal_features or []
        self.preserve_features = preserve_features or []
        self._adjustments: Dict[str, Dict[str, Callable[[Any], Any]]] = {}

    def fit(self, data: pd.DataFrame) -> "CounterfactualGenerator":
        """Learn adjustments for causal features.

        This learns the distribution shift needed when the sensitive
        attribute changes.
        """
        groups = data[self.sensitive_attr].unique()

        for feature in self.causal_features:
            self._adjustments[feature] = {}

            # Learn mean difference between groups
            group_means = {}
            for group in groups:
                mask = data[self.sensitive_attr] == group
                group_means[str(group)] = data[mask][feature].mean()

            global_mean = data[feature].mean()

            for group in groups:
                diff = global_mean - group_means[str(group)]
                self._adjustments[feature][str(group)] = lambda x, d=diff: x + d

        return self

    def generate(
        self,
        data: pd.DataFrame,
        target_group: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Generate counterfactual versions of the data.

        Args:
            data: Original data
            target_group: Target sensitive attribute value (flips all to this)

        Returns:
            Counterfactual DataFrame
        """
        cf = data.copy()
        groups = list(data[self.sensitive_attr].unique())

        if target_group is None and len(groups) == 2:
            # Binary case: flip all
            group_map = {groups[0]: groups[1], groups[1]: groups[0]}
            cf[self.sensitive_attr] = cf[self.sensitive_attr].map(group_map)
        elif target_group is not None:
            cf[self.sensitive_attr] = target_group
        else:
            # Multi-group: cycle to next group
            group_map = {g: groups[(i + 1) % len(groups)] for i, g in enumerate(groups)}
            cf[self.sensitive_attr] = cf[self.sensitive_attr].map(group_map)

        # Apply causal adjustments
        for feature in self.causal_features:
            if feature in self._adjustments:
                for group in groups:
                    mask = data[self.sensitive_attr] == group
                    if str(group) in self._adjustments[feature]:
                        adj_func = self._adjustments[feature][str(group)]
                        cf.loc[mask, feature] = data.loc[mask, feature].apply(adj_func)

        return cf


class FairnessAudit:
    """Comprehensive fairness audit for synthetic data.

    Compares original and synthetic data across multiple fairness metrics.

    Example:
        >>> audit = FairnessAudit(
        ...     sensitive_attrs=["gender", "race"],
        ...     outcome_col="income"
        ... )
        >>> results = audit.compare(original_df, synthetic_df)
        >>> print(results.summary())
    """

    def __init__(
        self,
        sensitive_attrs: List[str],
        outcome_col: str,
        positive_outcome: Any = 1,
    ) -> None:
        self.sensitive_attrs = sensitive_attrs
        self.outcome_col = outcome_col
        self.positive_outcome = positive_outcome

    def compare(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compare fairness between original and synthetic data."""
        results = {
            "original": {},
            "synthetic": {},
            "improvement": {},
        }

        analyzer = FairnessAnalyzer(
            sensitive_attributes=self.sensitive_attrs,
            outcome_column=self.outcome_col,
            positive_outcome=self.positive_outcome,
        )

        orig_reports = analyzer.analyze(original)
        synth_reports = analyzer.analyze(synthetic)

        for attr in self.sensitive_attrs:
            if attr in orig_reports and attr in synth_reports:
                orig_dp = orig_reports[attr].metrics.get("demographic_parity")
                synth_dp = synth_reports[attr].metrics.get("demographic_parity")

                if orig_dp and synth_dp:
                    results["original"][attr] = orig_dp.value
                    results["synthetic"][attr] = synth_dp.value
                    results["improvement"][attr] = (
                        synth_dp.value - orig_dp.value
                    ) / orig_dp.value if orig_dp.value > 0 else 0

        return results

    def generate_report(
        self,
        original: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> str:
        """Generate human-readable fairness comparison report."""
        comparison = self.compare(original, synthetic)

        lines = [
            "=" * 60,
            "FAIRNESS AUDIT REPORT",
            "=" * 60,
            "",
            "Demographic Parity Comparison:",
            "-" * 40,
        ]

        for attr in self.sensitive_attrs:
            if attr in comparison["original"]:
                orig = comparison["original"][attr]
                synth = comparison["synthetic"][attr]
                impr = comparison["improvement"][attr]

                status = "✓" if synth >= 0.8 else "✗"
                direction = "↑" if impr > 0 else "↓" if impr < 0 else "→"

                lines.append(
                    f"  {attr}:"
                )
                lines.append(
                    f"    Original:  {orig:.3f}"
                )
                lines.append(
                    f"    Synthetic: {synth:.3f} {status}"
                )
                lines.append(
                    f"    Change:    {direction} {impr*100:+.1f}%"
                )
                lines.append("")

        lines.extend([
            "-" * 40,
            "Legend: ✓ = Fair (≥0.8), ✗ = Unfair (<0.8)",
            "=" * 60,
        ])

        return "\n".join(lines)


def balance_dataset(
    data: pd.DataFrame,
    sensitive_attr: str,
    outcome_col: Optional[str] = None,
    strategy: str = "oversample",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience function to balance a dataset for fairness.

    Args:
        data: Input DataFrame
        sensitive_attr: Column with sensitive attribute
        outcome_col: Column with outcome (optional)
        strategy: 'oversample', 'undersample', or 'smote'
        random_state: Random seed

    Returns:
        Balanced DataFrame
    """
    rng = np.random.default_rng(random_state)
    groups = data[sensitive_attr].unique()

    if strategy == "oversample":
        max_size = max(len(data[data[sensitive_attr] == g]) for g in groups)
        parts = []

        for group in groups:
            group_data = data[data[sensitive_attr] == group]
            if len(group_data) < max_size:
                # Oversample to match max size
                extra = group_data.sample(
                    n=max_size - len(group_data),
                    replace=True,
                    random_state=int(rng.integers(2**31)),
                )
                parts.extend([group_data, extra])
            else:
                parts.append(group_data)

        return pd.concat(parts, ignore_index=True)

    elif strategy == "undersample":
        min_size = min(len(data[data[sensitive_attr] == g]) for g in groups)
        parts = []

        for group in groups:
            group_data = data[data[sensitive_attr] == group]
            sampled = group_data.sample(
                n=min_size,
                random_state=int(rng.integers(2**31)),
            )
            parts.append(sampled)

        return pd.concat(parts, ignore_index=True)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
