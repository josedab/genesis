"""Scenario generation for batch conditional data generation.

This module provides ScenarioGenerator for generating data
across multiple scenarios in batch, useful for test data generation.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.generators.conditional.samplers import ConditionalSampler
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ScenarioGenerator:
    """Generate data for multiple scenarios in batch.

    Useful for generating test data covering various edge cases
    or creating stratified synthetic datasets.
    """

    def __init__(self, generator: Any) -> None:
        """Initialize with a fitted generator.

        Args:
            generator: Fitted generator instance
        """
        self.generator = generator
        self.sampler = ConditionalSampler()

    def generate_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        samples_per_scenario: int = 100,
        include_scenario_id: bool = True,
    ) -> pd.DataFrame:
        """Generate data for multiple scenarios.

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
            ...     {"fraud": True, "amount": ("between", (1000, 5000))},
            ... ]
            >>> data = generator.generate_scenarios(scenarios, samples_per_scenario=500)
        """
        results = []

        for i, conditions in enumerate(scenarios):
            logger.info(f"Generating scenario {i + 1}/{len(scenarios)}: {conditions}")

            scenario_data = self.sampler.sample(
                generator_fn=lambda n: self.generator.generate(n),
                n_samples=samples_per_scenario,
                conditions=conditions,
            )

            if include_scenario_id:
                scenario_data = scenario_data.copy()
                scenario_data["scenario_id"] = i

            results.append(scenario_data)

        combined = pd.concat(results, ignore_index=True)
        logger.info(f"Generated {len(combined)} samples across {len(scenarios)} scenarios")
        return combined


def conditional_generate(
    generator: Any,
    n_samples: int,
    conditions: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience function for conditional generation.

    Args:
        generator: Fitted generator
        n_samples: Number of samples to generate
        conditions: Optional conditions dictionary
        **kwargs: Additional arguments for ConditionalSampler

    Returns:
        Generated DataFrame satisfying conditions
    """
    if conditions is None:
        return generator.generate(n_samples)

    sampler = ConditionalSampler(**kwargs)
    return sampler.sample(
        generator_fn=lambda n: generator.generate(n),
        n_samples=n_samples,
        conditions=conditions,
    )
