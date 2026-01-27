"""Mixin classes for extending generator functionality.

These mixins provide optional capabilities that can be composed
with generators to add features like conditional generation,
upsampling, scenario generation, and quality reporting.

This follows the composition over inheritance principle,
keeping the base generator focused on core fit/generate.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from genesis.evaluation.report import QualityReport


class ConditionalGenerationMixin:
    """Mixin providing conditional generation capabilities.

    Requires the class to have:
        - _is_fitted: bool
        - generate(n_samples, apply_constraints=False) -> DataFrame
    """

    def generate_conditional(
        self,
        n_samples: int,
        conditions: Dict[str, Any],
        max_trials: int = 100,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """Generate synthetic data satisfying conditions.

        Uses rejection sampling with adaptive batch sizing for efficiency.

        Args:
            n_samples: Number of samples to generate
            conditions: Dictionary of conditions to satisfy.
                Supports simple equality ({"col": value}) or operators
                ({"col": (">=", 18), "status": ("in", ["A", "B"])})
            max_trials: Maximum generation attempts
            batch_size: Initial batch size for generation

        Returns:
            DataFrame with samples satisfying all conditions

        Example:
            >>> data = generator.generate_conditional(1000, {"fraud": True})
            >>> data = generator.generate_conditional(1000, {
            ...     "age": (">=", 18),
            ...     "country": ("in", ["US", "UK"]),
            ... })
        """
        from genesis.core.exceptions import NotFittedError
        from genesis.generators.conditional import ConditionalSampler

        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(self.__class__.__name__)

        sampler = ConditionalSampler(max_trials=max_trials, batch_size=batch_size)
        return sampler.sample(
            generator_fn=lambda n: self.generate(n, apply_constraints=False),
            n_samples=n_samples,
            conditions=conditions,
        )


class UpsamplingMixin:
    """Mixin providing class imbalance upsampling capabilities.

    Requires the class to have:
        - _is_fitted: bool
        - Used as 'self' in Upsampler
    """

    def upsample(
        self,
        data: pd.DataFrame,
        target_column: str,
        target_ratio: Optional[float] = None,
        target_counts: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Upsample minority classes using synthetic data.

        Generates synthetic samples to balance class distributions.

        Args:
            data: Original data to augment
            target_column: Column to balance
            target_ratio: Target minimum ratio for each class (e.g., 0.5)
            target_counts: Specific target counts per class

        Returns:
            Combined DataFrame with original + synthetic samples

        Example:
            >>> balanced = generator.upsample(data, "fraud")
            >>> balanced = generator.upsample(
            ...     data, "fraud",
            ...     target_counts={True: 5000, False: 5000}
            ... )
        """
        from genesis.core.exceptions import NotFittedError
        from genesis.generators.conditional import Upsampler

        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(self.__class__.__name__)

        upsampler = Upsampler(self, target_column)
        return upsampler.upsample(data, target_ratio=target_ratio, target_counts=target_counts)


class ScenarioGenerationMixin:
    """Mixin providing batch scenario generation capabilities.

    Requires the class to have:
        - _is_fitted: bool
        - Used as 'self' in ScenarioGenerator
    """

    def generate_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        samples_per_scenario: int = 100,
        include_scenario_id: bool = True,
    ) -> pd.DataFrame:
        """Generate data for multiple scenarios in batch.

        Useful for generating test data covering various edge cases
        or creating stratified synthetic datasets.

        Args:
            scenarios: List of condition dictionaries
            samples_per_scenario: Number of samples per scenario
            include_scenario_id: Whether to add a 'scenario_id' column

        Returns:
            Combined DataFrame with all scenarios

        Example:
            >>> scenarios = [
            ...     {"fraud": True, "amount": (">=", 10000)},
            ...     {"fraud": False, "amount": ("<", 1000)},
            ... ]
            >>> data = generator.generate_scenarios(scenarios, samples_per_scenario=500)
        """
        from genesis.core.exceptions import NotFittedError
        from genesis.generators.conditional import ScenarioGenerator

        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(self.__class__.__name__)

        scenario_gen = ScenarioGenerator(self)
        return scenario_gen.generate_scenarios(
            scenarios=scenarios,
            samples_per_scenario=samples_per_scenario,
            include_scenario_id=include_scenario_id,
        )


class QualityReportMixin:
    """Mixin providing quality evaluation capabilities.

    Requires the class to have:
        - _is_fitted: bool
        - _original_data: Optional[DataFrame]
        - generate(n_samples) -> DataFrame
    """

    def quality_report(self, n_eval_samples: Optional[int] = None) -> "QualityReport":
        """Generate a quality report comparing synthetic to real data.

        Args:
            n_eval_samples: Number of samples to generate for evaluation.
                           Defaults to min(len(original_data), 1000).

        Returns:
            QualityReport with comprehensive metrics

        Raises:
            NotFittedError: If generator has not been fitted
        """
        from genesis.core.exceptions import NotFittedError
        from genesis.evaluation.evaluator import QualityEvaluator

        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(self.__class__.__name__)

        original_data = getattr(self, "_original_data", None)
        if original_data is None:
            raise NotFittedError(
                f"{self.__class__.__name__} does not have original data stored. "
                "Use QualityEvaluator directly for manual evaluation."
            )

        n_eval = n_eval_samples or min(len(original_data), 1000)
        synthetic_sample = self.generate(n_eval)

        evaluator = QualityEvaluator(
            real_data=original_data,
            synthetic_data=synthetic_sample,
        )
        return evaluator.evaluate()


class GeneratorExtensionsMixin(
    ConditionalGenerationMixin,
    UpsamplingMixin,
    ScenarioGenerationMixin,
    QualityReportMixin,
):
    """Combined mixin providing all generator extension capabilities.

    This is a convenience class that bundles all the individual mixins.
    Use this for full-featured generators, or compose individual mixins
    for lighter-weight classes.
    """

    pass
