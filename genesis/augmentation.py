"""Synthetic Data Augmentation for imbalanced datasets.

This module provides targeted augmentation for minority classes using
synthetic data generation with quality guarantees.

Features:
    - Multiple augmentation strategies (oversample, undersample, hybrid)
    - Quality-controlled synthetic sample generation
    - Automatic imbalance detection and planning
    - Support for multi-class and binary targets

Example:
    Basic augmentation with convenience function::

        from genesis import augment_imbalanced

        # Balance an imbalanced dataset
        balanced_df = augment_imbalanced(
            df,
            target_column="fraud_label",
            strategy="oversample"
        )

    Using the class for more control::

        from genesis.augmentation import SyntheticAugmenter, AugmentationPlanner

        # Analyze imbalance
        planner = AugmentationPlanner()
        plan = planner.analyze(df, target_column="label")
        print(f"Imbalance ratio: {plan.imbalance_ratio:.2f}")

        # Augment
        augmenter = SyntheticAugmenter(strategy="oversample")
        augmenter.fit(df, target_column="label")
        balanced = augmenter.augment(target_ratio=1.0)

Classes:
    AugmentationStrategy: Enum of available augmentation strategies.
    AugmentationPlan: Plan for augmentation operations.
    AugmentationPlanner: Analyzes data and creates augmentation plans.
    SyntheticAugmenter: Performs augmentation using synthetic generation.

Functions:
    augment_imbalanced: One-line convenience function for augmentation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


class AugmentationStrategy(str, Enum):
    """Strategy for data augmentation.

    Attributes:
        OVERSAMPLE: Generate synthetic samples for minority classes.
        UNDERSAMPLE: Remove samples from majority classes.
        HYBRID: Combine oversampling minorities with undersampling majorities.
        CONDITIONAL: Use conditional generation targeting specific classes.
    """

    OVERSAMPLE = "oversample"  # Generate more minority samples
    UNDERSAMPLE = "undersample"  # Reduce majority samples
    HYBRID = "hybrid"  # Combine over and undersampling
    CONDITIONAL = "conditional"  # Use conditional generation


@dataclass
class ClassDistribution:
    """Distribution statistics for a target class.

    Attributes:
        class_value: The class label value.
        count: Number of samples in this class.
        ratio: Fraction of total samples (0-1).
        is_minority: Whether this is a minority class.
        target_count: Desired count after augmentation.
    """

    class_value: Any
    count: int
    ratio: float
    is_minority: bool = False
    target_count: Optional[int] = None


@dataclass
class AugmentationPlan:
    """Plan for augmentation operations.

    Contains the analysis results and target distribution for augmentation.

    Attributes:
        target_column: Name of the target/label column.
        original_distribution: Current class distribution statistics.
        planned_distribution: Target counts per class.
        strategy: Augmentation strategy to use.
        total_original: Total samples before augmentation.
        total_planned: Total samples after augmentation.
    """

    target_column: str
    original_distribution: Dict[Any, ClassDistribution]
    planned_distribution: Dict[Any, int]
    strategy: AugmentationStrategy
    total_original: int
    total_planned: int

    def summary(self) -> str:
        """Get human-readable summary of the augmentation plan."""
        lines = [
            f"Augmentation Plan for '{self.target_column}'",
            f"Strategy: {self.strategy.value}",
            f"Original samples: {self.total_original:,}",
            f"Planned samples: {self.total_planned:,}",
            "",
            "Class changes:",
        ]

        for class_val, dist in self.original_distribution.items():
            planned = self.planned_distribution.get(class_val, 0)
            change = planned - dist.count
            symbol = "+" if change > 0 else ""
            lines.append(f"  {class_val}: {dist.count:,} → {planned:,} ({symbol}{change:,})")

        return "\n".join(lines)


@dataclass
class AugmentationResult:
    """Result of augmentation operation."""

    augmented_data: pd.DataFrame
    plan: AugmentationPlan
    synthetic_indices: List[int]
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class ImbalanceAnalyzer:
    """Analyze class imbalance in datasets."""

    def __init__(self, minority_threshold: float = 0.2):
        """Initialize analyzer.

        Args:
            minority_threshold: Classes below this ratio are considered minority
        """
        self.minority_threshold = minority_threshold

    def analyze(self, data: pd.DataFrame, target_column: str) -> Dict[Any, ClassDistribution]:
        """Analyze class distribution.

        Args:
            data: Input DataFrame
            target_column: Column to analyze

        Returns:
            Dictionary mapping class values to distribution stats
        """
        if target_column not in data.columns:
            raise ValueError(f"Column '{target_column}' not found in data")

        value_counts = data[target_column].value_counts()
        total = len(data)
        max_count = value_counts.max()

        distribution = {}
        for class_val, count in value_counts.items():
            ratio = count / total
            is_minority = (count / max_count) < self.minority_threshold

            distribution[class_val] = ClassDistribution(
                class_value=class_val,
                count=count,
                ratio=ratio,
                is_minority=is_minority,
            )

        return distribution

    def get_imbalance_ratio(self, data: pd.DataFrame, target_column: str) -> float:
        """Get imbalance ratio (majority/minority counts).

        Args:
            data: Input DataFrame
            target_column: Target column

        Returns:
            Imbalance ratio
        """
        value_counts = data[target_column].value_counts()
        return value_counts.max() / value_counts.min()

    def get_minority_classes(self, data: pd.DataFrame, target_column: str) -> List[Any]:
        """Get list of minority class values.

        Args:
            data: Input DataFrame
            target_column: Target column

        Returns:
            List of minority class values
        """
        distribution = self.analyze(data, target_column)
        return [class_val for class_val, dist in distribution.items() if dist.is_minority]


class AugmentationPlanner:
    """Plan augmentation operations."""

    def __init__(
        self,
        target_ratio: float = 1.0,
        max_oversample_factor: float = 10.0,
        min_samples_per_class: int = 100,
    ):
        """Initialize planner.

        Args:
            target_ratio: Target ratio of minority to majority (1.0 = balanced)
            max_oversample_factor: Maximum factor to oversample any class
            min_samples_per_class: Minimum samples per class after augmentation
        """
        self.target_ratio = target_ratio
        self.max_oversample_factor = max_oversample_factor
        self.min_samples_per_class = min_samples_per_class
        self.analyzer = ImbalanceAnalyzer()

    def plan(
        self,
        data: pd.DataFrame,
        target_column: str,
        strategy: AugmentationStrategy = AugmentationStrategy.OVERSAMPLE,
    ) -> AugmentationPlan:
        """Create augmentation plan.

        Args:
            data: Input DataFrame
            target_column: Column to balance
            strategy: Augmentation strategy

        Returns:
            AugmentationPlan with target distribution
        """
        distribution = self.analyzer.analyze(data, target_column)

        # Calculate target counts
        if strategy == AugmentationStrategy.OVERSAMPLE:
            planned = self._plan_oversample(distribution)
        elif strategy == AugmentationStrategy.UNDERSAMPLE:
            planned = self._plan_undersample(distribution)
        elif strategy == AugmentationStrategy.HYBRID:
            planned = self._plan_hybrid(distribution)
        else:  # CONDITIONAL
            planned = self._plan_oversample(distribution)  # Same as oversample

        return AugmentationPlan(
            target_column=target_column,
            original_distribution=distribution,
            planned_distribution=planned,
            strategy=strategy,
            total_original=sum(d.count for d in distribution.values()),
            total_planned=sum(planned.values()),
        )

    def _plan_oversample(self, distribution: Dict[Any, ClassDistribution]) -> Dict[Any, int]:
        """Plan oversampling strategy."""
        max_count = max(d.count for d in distribution.values())
        target_count = int(max_count * self.target_ratio)
        target_count = max(target_count, self.min_samples_per_class)

        planned = {}
        for class_val, dist in distribution.items():
            if dist.is_minority:
                # Oversample to target, but respect max factor
                new_count = min(target_count, int(dist.count * self.max_oversample_factor))
                planned[class_val] = max(new_count, dist.count)
            else:
                planned[class_val] = dist.count

        return planned

    def _plan_undersample(self, distribution: Dict[Any, ClassDistribution]) -> Dict[Any, int]:
        """Plan undersampling strategy."""
        min_count = min(d.count for d in distribution.values())
        target_count = max(int(min_count / self.target_ratio), self.min_samples_per_class)

        planned = {}
        for class_val, dist in distribution.items():
            if not dist.is_minority:
                planned[class_val] = min(target_count, dist.count)
            else:
                planned[class_val] = dist.count

        return planned

    def _plan_hybrid(self, distribution: Dict[Any, ClassDistribution]) -> Dict[Any, int]:
        """Plan hybrid strategy (over + under)."""
        counts = [d.count for d in distribution.values()]
        median_count = int(np.median(counts))
        target_count = max(median_count, self.min_samples_per_class)

        planned = {}
        for class_val, dist in distribution.items():
            if dist.is_minority:
                # Oversample minorities
                new_count = min(target_count, int(dist.count * self.max_oversample_factor))
                planned[class_val] = max(new_count, dist.count)
            else:
                # Undersample majorities if way above median
                if dist.count > target_count * 2:
                    planned[class_val] = target_count
                else:
                    planned[class_val] = dist.count

        return planned


class SyntheticAugmenter:
    """Augment datasets using synthetic data generation."""

    def __init__(
        self,
        method: str = "auto",
        target_ratio: float = 1.0,
        quality_threshold: float = 0.8,
        random_state: Optional[int] = None,
    ):
        """Initialize augmenter.

        Args:
            method: Generation method ('auto', 'gaussian_copula', 'ctgan', etc.)
            target_ratio: Target minority/majority ratio
            quality_threshold: Minimum quality score for synthetic samples
            random_state: Random seed
        """
        self.method = method
        self.target_ratio = target_ratio
        self.quality_threshold = quality_threshold
        self.random_state = random_state

        self.planner = AugmentationPlanner(target_ratio=target_ratio)
        self._generator = None

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str,
        discrete_columns: Optional[List[str]] = None,
    ) -> "SyntheticAugmenter":
        """Fit the augmenter on training data.

        Args:
            data: Training data
            target_column: Column to balance
            discrete_columns: Categorical columns

        Returns:
            Self for chaining
        """
        self._data = data
        self._target_column = target_column
        self._discrete_columns = discrete_columns or []

        # Ensure target column is in discrete columns
        if target_column not in self._discrete_columns:
            self._discrete_columns = [target_column] + self._discrete_columns

        # Create generator
        if self.method == "auto":
            from genesis.automl import AutoMLSynthesizer

            self._generator = AutoMLSynthesizer()
        else:
            from genesis.generators.tabular import (
                CTGANGenerator,
                GaussianCopulaGenerator,
            )

            if self.method == "ctgan":
                self._generator = CTGANGenerator()
            else:
                self._generator = GaussianCopulaGenerator()

        # Fit generator
        self._generator.fit(data, discrete_columns=self._discrete_columns)

        return self

    def augment(
        self,
        strategy: AugmentationStrategy = AugmentationStrategy.OVERSAMPLE,
        validate_quality: bool = True,
    ) -> AugmentationResult:
        """Perform augmentation.

        Args:
            strategy: Augmentation strategy
            validate_quality: Whether to validate synthetic sample quality

        Returns:
            AugmentationResult with augmented data
        """
        if self._generator is None:
            raise RuntimeError("Must call fit() before augment()")

        # Create augmentation plan
        plan = self.planner.plan(self._data, self._target_column, strategy=strategy)

        # Generate synthetic samples for minority classes
        synthetic_dfs = []
        synthetic_indices_start = len(self._data)

        for class_val, dist in plan.original_distribution.items():
            target_count = plan.planned_distribution[class_val]
            samples_needed = target_count - dist.count

            if samples_needed > 0:
                # Generate samples for this class
                synthetic = self._generate_class_samples(
                    class_val, samples_needed, validate_quality
                )
                synthetic_dfs.append(synthetic)

        # Combine original and synthetic
        if synthetic_dfs:
            all_synthetic = pd.concat(synthetic_dfs, ignore_index=True)
            synthetic_indices = list(
                range(synthetic_indices_start, synthetic_indices_start + len(all_synthetic))
            )

            # Handle undersampling if needed
            if strategy in [AugmentationStrategy.UNDERSAMPLE, AugmentationStrategy.HYBRID]:
                original = self._undersample(plan)
            else:
                original = self._data.copy()

            augmented = pd.concat([original, all_synthetic], ignore_index=True)
        else:
            if strategy in [AugmentationStrategy.UNDERSAMPLE, AugmentationStrategy.HYBRID]:
                augmented = self._undersample(plan)
            else:
                augmented = self._data.copy()
            synthetic_indices = []

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(augmented)

        return AugmentationResult(
            augmented_data=augmented,
            plan=plan,
            synthetic_indices=synthetic_indices,
            quality_metrics=quality_metrics,
        )

    def _generate_class_samples(
        self,
        class_value: Any,
        n_samples: int,
        validate_quality: bool,
    ) -> pd.DataFrame:
        """Generate synthetic samples for a specific class."""
        # Generate extra samples to filter for quality
        oversample_factor = 2.0 if validate_quality else 1.0
        n_generate = int(n_samples * oversample_factor)

        # Generate samples
        synthetic = self._generator.generate(n_generate)

        # Filter to target class
        class_mask = synthetic[self._target_column] == class_value
        class_samples = synthetic[class_mask]

        # If not enough class samples, regenerate with conditioning
        attempts = 0
        while len(class_samples) < n_samples and attempts < 5:
            additional = self._generator.generate(n_generate)
            additional_class = additional[additional[self._target_column] == class_value]
            class_samples = pd.concat([class_samples, additional_class], ignore_index=True)
            attempts += 1

        # Take required number
        if len(class_samples) > n_samples:
            class_samples = class_samples.sample(n=n_samples, random_state=self.random_state)

        return class_samples.reset_index(drop=True)

    def _undersample(self, plan: AugmentationPlan) -> pd.DataFrame:
        """Undersample majority classes according to plan."""
        samples = []

        for class_val, target_count in plan.planned_distribution.items():
            class_data = self._data[self._data[self._target_column] == class_val]

            if len(class_data) > target_count:
                class_data = class_data.sample(n=target_count, random_state=self.random_state)

            samples.append(class_data)

        return pd.concat(samples, ignore_index=True)

    def _compute_quality_metrics(self, augmented: pd.DataFrame) -> Dict[str, float]:
        """Compute quality metrics for augmented data."""
        metrics = {}

        # Class balance ratio
        value_counts = augmented[self._target_column].value_counts()
        metrics["balance_ratio"] = value_counts.min() / value_counts.max()

        # Original data retention
        metrics["original_retention"] = min(len(self._data), len(augmented)) / len(augmented)

        return metrics


class ConditionalAugmenter:
    """Augmenter using conditional generation for precise class targeting."""

    def __init__(
        self,
        target_ratio: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """Initialize conditional augmenter.

        Args:
            target_ratio: Target minority/majority ratio
            random_state: Random seed
        """
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.planner = AugmentationPlanner(target_ratio=target_ratio)

    def augment(
        self,
        data: pd.DataFrame,
        target_column: str,
        generator_fn: Callable[[int], pd.DataFrame],
        strategy: AugmentationStrategy = AugmentationStrategy.CONDITIONAL,
    ) -> AugmentationResult:
        """Augment data using conditional generation.

        Args:
            data: Original data
            target_column: Column to balance
            generator_fn: Function that generates n samples with conditions
            strategy: Augmentation strategy

        Returns:
            AugmentationResult
        """
        from genesis.generators.conditional import (
            ConditionBuilder,
            GuidedConditionalSampler,
        )

        plan = self.planner.plan(data, target_column, strategy=strategy)

        # Fit conditional sampler
        sampler = GuidedConditionalSampler(strategy="iterative_refinement")
        sampler.fit(data)

        synthetic_dfs = []
        synthetic_indices_start = len(data)

        for class_val, dist in plan.original_distribution.items():
            samples_needed = plan.planned_distribution[class_val] - dist.count

            if samples_needed > 0:
                # Build condition for this class
                conditions = ConditionBuilder().where(target_column).eq(class_val).build()

                # Generate with conditions
                synthetic = sampler.sample(
                    generator_fn=generator_fn,
                    n_samples=samples_needed,
                    conditions=conditions,
                )
                synthetic_dfs.append(synthetic)

        # Combine
        if synthetic_dfs:
            all_synthetic = pd.concat(synthetic_dfs, ignore_index=True)
            synthetic_indices = list(
                range(synthetic_indices_start, synthetic_indices_start + len(all_synthetic))
            )
            augmented = pd.concat([data, all_synthetic], ignore_index=True)
        else:
            augmented = data.copy()
            synthetic_indices = []

        return AugmentationResult(
            augmented_data=augmented,
            plan=plan,
            synthetic_indices=synthetic_indices,
        )


def augment_imbalanced(
    data: pd.DataFrame,
    target_column: str,
    target_ratio: float = 1.0,
    method: str = "auto",
    strategy: AugmentationStrategy = AugmentationStrategy.OVERSAMPLE,
    discrete_columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> AugmentationResult:
    """Convenience function to augment an imbalanced dataset.

    Args:
        data: Input DataFrame
        target_column: Column to balance
        target_ratio: Target minority/majority ratio (1.0 = balanced)
        method: Generation method
        strategy: Augmentation strategy
        discrete_columns: Categorical columns
        random_state: Random seed

    Returns:
        AugmentationResult with balanced data
    """
    augmenter = SyntheticAugmenter(
        method=method,
        target_ratio=target_ratio,
        random_state=random_state,
    )

    augmenter.fit(data, target_column, discrete_columns=discrete_columns)
    return augmenter.augment(strategy=strategy)


def analyze_imbalance(
    data: pd.DataFrame,
    target_column: str,
) -> Dict[str, Any]:
    """Analyze class imbalance in a dataset.

    Args:
        data: Input DataFrame
        target_column: Column to analyze

    Returns:
        Dictionary with imbalance analysis
    """
    analyzer = ImbalanceAnalyzer()
    distribution = analyzer.analyze(data, target_column)

    return {
        "target_column": target_column,
        "n_classes": len(distribution),
        "imbalance_ratio": analyzer.get_imbalance_ratio(data, target_column),
        "minority_classes": analyzer.get_minority_classes(data, target_column),
        "distribution": {
            class_val: {
                "count": dist.count,
                "ratio": dist.ratio,
                "is_minority": dist.is_minority,
            }
            for class_val, dist in distribution.items()
        },
    }
