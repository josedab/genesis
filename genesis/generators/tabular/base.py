"""Base class for tabular data generators."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.types import ColumnList
from genesis.utils.transformers import DataTransformer

if TYPE_CHECKING:
    from genesis.core.config import GenesisConfig


class BaseTabularGenerator(BaseGenerator):
    """Base class for tabular data generators.

    This class provides common functionality for all tabular generators,
    including data transformation, constraint handling, and conditional generation.
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
    ) -> None:
        super().__init__(config, privacy)

        self._transformer: Optional[DataTransformer] = None
        self._output_info: List[tuple] = []
        self._column_names: List[str] = []
        self._n_categories_per_col: Dict[str, int] = {}

    @classmethod
    def from_config(cls, genesis_config: "GenesisConfig") -> "BaseTabularGenerator":
        """Create a generator from a unified GenesisConfig.

        Args:
            genesis_config: Unified configuration object

        Returns:
            Configured generator instance

        Example:
            >>> config = GenesisConfig(training={"epochs": 500})
            >>> generator = CTGANGenerator.from_config(config)
        """
        from genesis.core.config import GenesisConfig

        if not isinstance(genesis_config, GenesisConfig):
            raise TypeError(f"Expected GenesisConfig, got {type(genesis_config)}")

        return cls(
            config=genesis_config.training,
            privacy=genesis_config.privacy,
        )

    def _fit_transformer(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[ColumnList] = None,
    ) -> np.ndarray:
        """Fit the data transformer and transform data.

        Args:
            data: Training DataFrame
            discrete_columns: List of discrete column names

        Returns:
            Transformed numpy array
        """
        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_columns)
        self._column_names = list(data.columns)

        # Get output info for loss computation
        self._output_info = []
        for col in self._column_names:
            dim = self._transformer._output_dims[col]
            transformer = self._transformer._transformers[col]

            if hasattr(transformer, "n_modes"):
                # Numerical: 1 continuous + n_modes categorical
                self._output_info.append((1, "tanh"))
                if dim > 1:
                    self._output_info.append((dim - 1, "softmax"))
            else:
                # Categorical: all softmax
                self._output_info.append((dim, "softmax"))

        return self._transformer.transform(data)

    def _inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Transform generated data back to original format.

        Args:
            data: Generated numpy array

        Returns:
            DataFrame in original format
        """
        if self._transformer is None:
            raise RuntimeError("Transformer not fitted")

        return self._transformer.inverse_transform(data)

    def _apply_activation(self, data: np.ndarray) -> np.ndarray:
        """Apply activation functions based on output info.

        Args:
            data: Raw generator output

        Returns:
            Activated output
        """
        result = []
        offset = 0

        for dim, activation in self._output_info:
            slice_data = data[:, offset : offset + dim]

            if activation == "tanh":
                activated = np.tanh(slice_data)
            elif activation == "softmax":
                # Softmax
                exp_data = np.exp(slice_data - slice_data.max(axis=1, keepdims=True))
                activated = exp_data / exp_data.sum(axis=1, keepdims=True)
            else:
                activated = slice_data

            result.append(activated)
            offset += dim

        return np.hstack(result)

    def _sample_from_output(self, data: np.ndarray) -> np.ndarray:
        """Sample discrete values from softmax outputs.

        Args:
            data: Activated generator output

        Returns:
            Sampled output with one-hot categoricals
        """
        result = []
        offset = 0

        for dim, activation in self._output_info:
            slice_data = data[:, offset : offset + dim]

            if activation == "softmax" and dim > 1:
                # Sample from categorical distribution
                sampled = np.zeros_like(slice_data)
                for i in range(len(slice_data)):
                    probs = slice_data[i]
                    probs = np.clip(probs, 1e-10, 1.0)
                    probs = probs / probs.sum()
                    idx = np.random.choice(dim, p=probs)
                    sampled[i, idx] = 1
                result.append(sampled)
            else:
                result.append(slice_data)

            offset += dim

        return np.hstack(result)

    def _prepare_conditional_vector(
        self,
        conditions: Dict[str, Any],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        """Prepare conditional vector for conditional generation.

        Args:
            conditions: Dictionary of column conditions
            n_samples: Number of samples

        Returns:
            Conditional vector or None
        """
        if not conditions:
            return None

        # This is a simplified version - full implementation would use
        # the transformer to encode conditions properly
        cond_vector = np.zeros((n_samples, len(conditions)))

        for i, (col, value) in enumerate(conditions.items()):
            if col in self._column_names:
                # Simple encoding - real implementation would be more sophisticated
                cond_vector[:, i] = hash(str(value)) % 100 / 100.0

        return cond_vector

    @property
    def output_dimensions(self) -> int:
        """Get total output dimensions."""
        if self._transformer is None:
            return 0
        return self._transformer.get_output_dimensions()

    def get_learned_distributions(self) -> Dict[str, Any]:
        """Get information about learned distributions.

        Returns:
            Dictionary with distribution information per column
        """
        if self._transformer is None:
            return {}

        info = {}
        for col in self._column_names:
            transformer = self._transformer._transformers.get(col)
            if transformer is None:
                continue

            col_info = {"dim": self._transformer._output_dims[col]}

            if hasattr(transformer, "_modes"):
                col_info["type"] = "numerical"
                col_info["n_modes"] = len(transformer._modes)
                col_info["modes"] = transformer._modes.tolist()
            elif hasattr(transformer, "_categories"):
                col_info["type"] = "categorical"
                col_info["categories"] = transformer._categories

            info[col] = col_info

        return info

    def sample_conditional(
        self,
        n_samples: int,
        conditions: Dict[str, Any],
        max_iterations: int = 100,
        min_acceptance_rate: float = 0.001,
    ) -> pd.DataFrame:
        """Generate samples with specific conditions.

        Uses rejection sampling with adaptive batch sizing. If the acceptance
        rate is too low, raises an error rather than running indefinitely.

        Args:
            n_samples: Number of samples to generate
            conditions: Dictionary mapping column names to values or ranges
                       e.g., {'gender': 'F', 'age': (25, 35)}
            max_iterations: Maximum number of generation attempts (default: 100)
            min_acceptance_rate: Minimum acceptable ratio of valid samples (default: 0.001)

        Returns:
            DataFrame with samples satisfying conditions

        Raises:
            NotFittedError: If generator is not fitted
            GenerationError: If conditions are too restrictive to satisfy
        """
        from genesis.core.exceptions import GenerationError, NotFittedError

        if not self.is_fitted:
            raise NotFittedError("Generator must be fitted before generating")

        collected = []
        total_generated = 0
        total_accepted = 0
        batch_size = max(n_samples * 5, 1000)  # Initial batch size

        for iteration in range(max_iterations):
            # Generate candidates
            candidates = self.generate(batch_size)
            total_generated += len(candidates)

            # Apply conditions
            mask = pd.Series([True] * len(candidates), index=candidates.index)

            for col, condition in conditions.items():
                if col not in candidates.columns:
                    continue

                if isinstance(condition, tuple) and len(condition) == 2:
                    # Range condition
                    low, high = condition
                    mask &= (candidates[col] >= low) & (candidates[col] <= high)
                elif isinstance(condition, (list, range)):
                    # Value in list
                    mask &= candidates[col].isin(list(condition))
                else:
                    # Exact value
                    mask &= candidates[col] == condition

            filtered = candidates[mask]
            total_accepted += len(filtered)

            if len(filtered) > 0:
                collected.append(filtered)

            # Check if we have enough
            total_collected = sum(len(df) for df in collected)
            if total_collected >= n_samples:
                break

            # Check acceptance rate to avoid infinite loops
            if total_generated > batch_size * 3:  # After a few batches
                acceptance_rate = total_accepted / total_generated
                if acceptance_rate < min_acceptance_rate:
                    raise GenerationError(
                        f"Conditions too restrictive: acceptance rate {acceptance_rate:.6f} "
                        f"is below minimum {min_acceptance_rate}. "
                        f"Generated {total_generated} samples, accepted {total_accepted}. "
                        "Consider relaxing the conditions or using a different approach."
                    )

            # Adaptive batch sizing based on acceptance rate
            if total_accepted > 0:
                estimated_rate = total_accepted / total_generated
                needed = n_samples - total_collected
                # Estimate batch size needed, with safety margin
                batch_size = min(int(needed / max(estimated_rate, 0.01) * 1.5), 100000)
                batch_size = max(batch_size, 100)  # Minimum batch

        if not collected:
            raise GenerationError(
                f"Could not generate any samples satisfying conditions after "
                f"{max_iterations} iterations. Conditions may be impossible to satisfy."
            )

        result = pd.concat(collected, ignore_index=True)
        return result.head(n_samples).reset_index(drop=True)
