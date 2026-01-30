"""Upsampling for class imbalance using synthetic data.

This module provides the Upsampler class for addressing class imbalance
by generating synthetic samples for underrepresented classes.
"""

from typing import Any, Dict, Optional

import pandas as pd

from genesis.core.exceptions import ValidationError
from genesis.generators.conditional.samplers import ConditionalSampler
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class Upsampler:
    """Upsample minority classes using synthetic data.

    This class helps address class imbalance by generating synthetic
    samples for underrepresented classes.
    """

    def __init__(
        self,
        generator: Any,  # BaseGenerator but avoid circular import
        target_column: str,
        strategy: str = "uniform",
    ) -> None:
        """Initialize the upsampler.

        Args:
            generator: Fitted generator instance
            target_column: Column to balance
            strategy: Balancing strategy ('uniform', 'proportional', 'custom')
        """
        self.generator = generator
        self.target_column = target_column
        self.strategy = strategy
        self._class_distribution: Optional[Dict[Any, float]] = None
        self._class_counts: Optional[Dict[Any, int]] = None
        self._max_class_count: Optional[int] = None

    def fit(self, data: pd.DataFrame) -> "Upsampler":
        """Analyze class distribution in data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        if self.target_column not in data.columns:
            raise ValidationError(f"Column '{self.target_column}' not found in data")

        counts = data[self.target_column].value_counts()
        self._class_distribution = (counts / len(data)).to_dict()
        self._class_counts = counts.to_dict()
        self._max_class_count = counts.max()

        logger.info(f"Class distribution: {self._class_counts}")
        return self

    def upsample(
        self,
        data: pd.DataFrame,
        target_ratio: Optional[float] = None,
        target_counts: Optional[Dict[Any, int]] = None,
    ) -> pd.DataFrame:
        """Upsample minority classes.

        Args:
            data: Original data to augment
            target_ratio: Target ratio for minority classes (e.g., 0.5 for 50%)
            target_counts: Specific target counts per class

        Returns:
            Combined original + synthetic data
        """
        if self._class_distribution is None:
            self.fit(data)

        current_counts = data[self.target_column].value_counts().to_dict()
        max_count = max(current_counts.values())

        # Determine target counts
        if target_counts is not None:
            targets = target_counts
        elif target_ratio is not None:
            # Target ratio is the minimum class ratio
            total = len(data)
            min_count = int(total * target_ratio / len(current_counts))
            targets = {cls: max(count, min_count) for cls, count in current_counts.items()}
        else:
            # Default: uniform (match the majority class)
            targets = dict.fromkeys(current_counts.keys(), max_count)

        # Generate synthetic samples for each class that needs more
        synthetic_parts = [data]
        sampler = ConditionalSampler()

        for cls, target in targets.items():
            current = current_counts.get(cls, 0)
            needed = target - current

            if needed > 0:
                logger.info(f"Generating {needed} synthetic samples for {self.target_column}={cls}")

                conditions = {self.target_column: cls}
                synthetic = sampler.sample(
                    generator_fn=lambda n: self.generator.generate(n),
                    n_samples=needed,
                    conditions=conditions,
                )
                synthetic_parts.append(synthetic)

        result = pd.concat(synthetic_parts, ignore_index=True)
        logger.info(f"Upsampling complete: {len(data)} → {len(result)} samples")
        return result
