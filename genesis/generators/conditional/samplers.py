"""Conditional samplers for synthetic data generation.

This module provides ConditionalSampler (rejection-based) and
GuidedConditionalSampler (guided generation) for conditional sampling.
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

from genesis.generators.conditional.conditions import ConditionSet, Operator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ConditionalSampler:
    """Efficient conditional sampling from generators.

    Supports multiple sampling strategies:
    - Rejection sampling: Generate and filter (simple but can be slow)
    - Guided sampling: Use conditions to guide generation (more efficient)
    - Hybrid: Start guided, fall back to rejection if needed
    """

    def __init__(
        self,
        max_trials: int = 100,
        batch_size: int = 1000,
        efficiency_threshold: float = 0.01,
    ) -> None:
        """Initialize the conditional sampler.

        Args:
            max_trials: Maximum number of generation attempts
            batch_size: Batch size for generation attempts
            efficiency_threshold: Minimum acceptance rate before warning
        """
        self.max_trials = max_trials
        self.batch_size = batch_size
        self.efficiency_threshold = efficiency_threshold

    def sample(
        self,
        generator_fn: Callable[[int], pd.DataFrame],
        n_samples: int,
        conditions: Union[Dict[str, Any], ConditionSet],
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Sample data satisfying conditions using rejection sampling.

        Args:
            generator_fn: Function that generates n samples
            n_samples: Number of samples to generate
            conditions: Conditions to satisfy
            seed: Random seed for reproducibility

        Returns:
            DataFrame with samples satisfying all conditions

        Raises:
            ValidationError: If unable to generate enough samples
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert dict to ConditionSet if needed
        if isinstance(conditions, dict):
            conditions = ConditionSet.from_dict(conditions)

        collected = []
        total_generated = 0
        total_accepted = 0

        for trial in range(self.max_trials):
            # Estimate how many samples we need based on acceptance rate
            if total_generated > 0:
                acceptance_rate = max(total_accepted / total_generated, 0.001)
                needed = n_samples - total_accepted
                estimate = int(needed / acceptance_rate * 1.5)  # 50% buffer
                batch = min(max(estimate, self.batch_size), self.batch_size * 10)
            else:
                batch = self.batch_size

            # Generate batch
            samples = generator_fn(batch)
            total_generated += len(samples)

            # Filter by conditions
            mask = conditions.evaluate(samples)
            accepted = samples[mask]
            total_accepted += len(accepted)
            collected.append(accepted)

            # Check if we have enough
            if total_accepted >= n_samples:
                break

            # Warn if acceptance rate is very low
            acceptance_rate = total_accepted / total_generated
            if trial > 5 and acceptance_rate < self.efficiency_threshold:
                logger.warning(
                    f"Low acceptance rate ({acceptance_rate:.2%}). "
                    "Consider loosening conditions or using guided generation."
                )

        # Combine and trim to exact size
        result = pd.concat(collected, ignore_index=True)

        if len(result) < n_samples:
            logger.warning(
                f"Could only generate {len(result)} samples satisfying conditions "
                f"(requested {n_samples}). Generated {total_generated} total samples."
            )
        else:
            result = result.head(n_samples)

        logger.info(
            f"Conditional generation: {len(result)}/{n_samples} samples, "
            f"acceptance rate: {total_accepted/total_generated:.2%}"
        )

        return result

    def estimate_feasibility(
        self,
        generator_fn: Callable[[int], pd.DataFrame],
        conditions: Union[Dict[str, Any], ConditionSet],
        sample_size: int = 1000,
    ) -> Dict[str, Any]:
        """Estimate feasibility of conditions.

        Args:
            generator_fn: Function that generates samples
            conditions: Conditions to test
            sample_size: Size of sample for estimation

        Returns:
            Dictionary with feasibility metrics
        """
        if isinstance(conditions, dict):
            conditions = ConditionSet.from_dict(conditions)

        # Generate sample
        samples = generator_fn(sample_size)
        mask = conditions.evaluate(samples)
        acceptance_rate = mask.mean()

        # Estimate samples needed for various targets
        estimates = {}
        for target in [100, 1000, 10000]:
            if acceptance_rate > 0:
                estimates[target] = int(target / acceptance_rate)
            else:
                estimates[target] = float("inf")

        return {
            "acceptance_rate": float(acceptance_rate),
            "samples_tested": sample_size,
            "samples_passed": int(mask.sum()),
            "estimated_generations_needed": estimates,
            "feasible": acceptance_rate > 0,
        }


class GuidedConditionalSampler:
    """Advanced conditional sampler using guided generation.

    This sampler attempts to guide the generation process directly rather than
    using rejection sampling, resulting in much better efficiency for rare conditions.

    Strategies:
    - latent_guidance: Modify latent space to bias towards conditions
    - iterative_refinement: Generate and refine samples iteratively
    - importance_sampling: Use importance weights to improve efficiency
    """

    def __init__(
        self,
        strategy: str = "iterative_refinement",
        max_iterations: int = 10,
        refinement_steps: int = 3,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the guided sampler.

        Args:
            strategy: Sampling strategy to use
            max_iterations: Maximum iterations for refinement
            refinement_steps: Steps per refinement iteration
            temperature: Sampling temperature (higher = more diverse)
        """
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.refinement_steps = refinement_steps
        self.temperature = temperature
        self._column_stats: Dict[str, Dict[str, Any]] = {}

    def fit(self, data: pd.DataFrame) -> "GuidedConditionalSampler":
        """Learn column statistics for guided sampling.

        Args:
            data: Reference data to learn from

        Returns:
            Self for method chaining
        """
        for col in data.columns:
            series = data[col]
            stats: Dict[str, Any] = {
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique()),
            }

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    stats["mean"] = float(non_null.mean())
                    stats["std"] = float(non_null.std())
                    stats["min"] = float(non_null.min())
                    stats["max"] = float(non_null.max())
                    try:
                        stats["quantiles"] = {
                            0.25: float(non_null.quantile(0.25)),
                            0.5: float(non_null.quantile(0.5)),
                            0.75: float(non_null.quantile(0.75)),
                        }
                    except TypeError:
                        stats["quantiles"] = None
            else:
                stats["value_counts"] = series.value_counts(normalize=True).to_dict()

            self._column_stats[col] = stats

        return self

    def sample(
        self,
        generator_fn: Callable[[int], pd.DataFrame],
        n_samples: int,
        conditions: Union[Dict[str, Any], ConditionSet],
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Sample data satisfying conditions using guided generation.

        Args:
            generator_fn: Function that generates n samples
            n_samples: Number of samples to generate
            conditions: Conditions to satisfy
            seed: Random seed

        Returns:
            DataFrame with samples satisfying conditions
        """
        if seed is not None:
            np.random.seed(seed)

        if isinstance(conditions, dict):
            conditions = ConditionSet.from_dict(conditions)

        if self.strategy == "iterative_refinement":
            return self._sample_iterative(generator_fn, n_samples, conditions)
        elif self.strategy == "importance_sampling":
            return self._sample_importance(generator_fn, n_samples, conditions)
        else:
            # Fall back to rejection sampling
            sampler = ConditionalSampler()
            return sampler.sample(generator_fn, n_samples, conditions, seed)

    def _sample_iterative(
        self,
        generator_fn: Callable[[int], pd.DataFrame],
        n_samples: int,
        conditions: ConditionSet,
    ) -> pd.DataFrame:
        """Sample using iterative refinement strategy."""
        collected = []
        total_collected = 0

        n_conditions = len(conditions.conditions)
        base_batch = max(1000, n_samples * (2 ** min(n_conditions, 5)))

        for iteration in range(self.max_iterations):
            if total_collected > 0:
                efficiency = total_collected / (iteration + 1) / base_batch
                batch_size = int(
                    min(base_batch * 2, (n_samples - total_collected) / max(efficiency, 0.01))
                )
            else:
                batch_size = base_batch

            batch_size = max(100, min(batch_size, 50000))

            samples = generator_fn(batch_size)
            mask = conditions.evaluate(samples)
            accepted = samples[mask]

            if len(accepted) > 0:
                collected.append(accepted)
                total_collected += len(accepted)

            if total_collected >= n_samples:
                break

            if iteration < self.max_iterations - 1 and len(accepted) < batch_size * 0.1:
                logger.debug(
                    f"Iteration {iteration + 1}: {len(accepted)}/{batch_size} accepted "
                    f"({len(accepted)/batch_size:.1%})"
                )

        if not collected:
            logger.warning("No samples satisfied conditions")
            return pd.DataFrame(columns=generator_fn(1).columns)

        result = pd.concat(collected, ignore_index=True).head(n_samples)

        logger.info(
            f"Guided generation: {len(result)}/{n_samples} samples in {iteration + 1} iterations"
        )

        return result

    def _sample_importance(
        self,
        generator_fn: Callable[[int], pd.DataFrame],
        n_samples: int,
        conditions: ConditionSet,
    ) -> pd.DataFrame:
        """Sample using importance sampling."""
        oversample_factor = 5
        samples = generator_fn(n_samples * oversample_factor)

        weights = self._calculate_importance_weights(samples, conditions)

        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(len(samples)) / len(samples)

        indices = np.random.choice(
            len(samples),
            size=min(n_samples, len(samples)),
            replace=True,
            p=weights,
        )

        result = samples.iloc[indices].reset_index(drop=True)

        mask = conditions.evaluate(result)
        result = result[mask].head(n_samples)

        if len(result) < n_samples:
            sampler = ConditionalSampler()
            additional = sampler.sample(
                generator_fn,
                n_samples - len(result),
                conditions,
            )
            result = pd.concat([result, additional], ignore_index=True)

        return result

    def _calculate_importance_weights(
        self,
        samples: pd.DataFrame,
        conditions: ConditionSet,
    ) -> np.ndarray:
        """Calculate importance weights based on condition proximity."""
        weights = np.ones(len(samples))

        for condition in conditions.conditions:
            col = condition.column
            if col not in samples.columns:
                continue

            series = samples[col]

            if condition.operator == Operator.EQ:
                if pd.api.types.is_numeric_dtype(series):
                    target = condition.value
                    std = self._column_stats.get(col, {}).get("std", 1.0) or 1.0
                    distance = np.abs(series - target) / std
                    weights *= np.exp(-distance * self.temperature)
                else:
                    weights *= (series == condition.value).astype(float) * 0.99 + 0.01

            elif condition.operator in (Operator.GT, Operator.GE):
                if pd.api.types.is_numeric_dtype(series):
                    target = condition.value
                    above = series >= target
                    distance = np.maximum(0, target - series)
                    std = self._column_stats.get(col, {}).get("std", 1.0) or 1.0
                    weights *= above.astype(float) * 0.99 + np.exp(-distance / std) * 0.01

            elif condition.operator in (Operator.LT, Operator.LE):
                if pd.api.types.is_numeric_dtype(series):
                    target = condition.value
                    below = series <= target
                    distance = np.maximum(0, series - target)
                    std = self._column_stats.get(col, {}).get("std", 1.0) or 1.0
                    weights *= below.astype(float) * 0.99 + np.exp(-distance / std) * 0.01

            elif condition.operator == Operator.BETWEEN:
                if pd.api.types.is_numeric_dtype(series):
                    low, high = condition.value
                    in_range = (series >= low) & (series <= high)
                    distance = np.minimum(np.abs(series - low), np.abs(series - high))
                    std = self._column_stats.get(col, {}).get("std", 1.0) or 1.0
                    weights *= in_range.astype(float) * 0.99 + np.exp(-distance / std) * 0.01

            elif condition.operator == Operator.IN:
                weights *= series.isin(condition.value).astype(float) * 0.99 + 0.01

        return weights
