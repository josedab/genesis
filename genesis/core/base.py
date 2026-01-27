"""Base classes for synthetic data generators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.constraints import BaseConstraint, ConstraintSet
from genesis.core.exceptions import NotFittedError, ValidationError
from genesis.core.mixins import GeneratorExtensionsMixin
from genesis.core.types import (
    ColumnList,
    DataSchema,
    FittingResult,
    GeneratorMethod,
    ProgressCallback,
)

if TYPE_CHECKING:
    from genesis.core.config import GenesisConfig


class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators.

    This class defines the interface that all generators must implement.
    It follows the sklearn-style fit/generate pattern for ease of use.
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize the generator.

        Args:
            config: Generator configuration
            privacy: Privacy configuration
        """
        self.config = config or GeneratorConfig()
        self.privacy = privacy or PrivacyConfig()
        self._is_fitted = False
        self._schema: Optional[DataSchema] = None
        self._constraints: ConstraintSet = ConstraintSet()
        self._discrete_columns: List[str] = []
        self._original_data: Optional[pd.DataFrame] = None

    @property
    def is_fitted(self) -> bool:
        """Check if the generator has been fitted."""
        return self._is_fitted

    @property
    def schema(self) -> Optional[DataSchema]:
        """Get the data schema learned during fitting."""
        return self._schema

    @abstractmethod
    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[ColumnList] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Implementation of the fitting logic.

        Args:
            data: Training data
            discrete_columns: List of discrete/categorical columns
            progress_callback: Optional callback for progress updates

        Returns:
            FittingResult with training details
        """
        pass

    @abstractmethod
    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Implementation of the generation logic.

        Args:
            n_samples: Number of samples to generate
            conditions: Optional conditions for conditional generation
            progress_callback: Optional callback for progress updates

        Returns:
            Generated DataFrame
        """
        pass

    def fit(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[ColumnList] = None,
        constraints: Optional[List[BaseConstraint]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "BaseGenerator":
        """Fit the generator to the training data.

        Args:
            data: Training DataFrame
            discrete_columns: List of discrete/categorical column names
            constraints: List of constraints to enforce
            progress_callback: Optional callback for progress updates

        Returns:
            Self for method chaining

        Raises:
            ValidationError: If data validation fails
        """
        self._validate_input_data(data)

        # Store discrete columns
        self._discrete_columns = discrete_columns or []

        # Store constraints
        if constraints:
            self._constraints = ConstraintSet(constraints)

        # Store reference to original data for evaluation
        self._original_data = data.copy()

        # Call implementation
        result = self._fit_impl(data, discrete_columns, progress_callback)

        if result.success:
            self._is_fitted = True

        return self

    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        apply_constraints: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data.

        Args:
            n_samples: Number of samples to generate
            conditions: Optional conditions for conditional generation
            apply_constraints: Whether to apply constraints post-generation
            progress_callback: Optional callback for progress updates

        Returns:
            Generated DataFrame

        Raises:
            NotFittedError: If generator has not been fitted
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate data
        synthetic_data = self._generate_impl(n_samples, conditions, progress_callback)

        # Apply constraints if requested
        if apply_constraints and len(self._constraints) > 0:
            synthetic_data, _ = self._constraints.validate_and_transform(synthetic_data)

        return synthetic_data

    def fit_generate(
        self,
        data: pd.DataFrame,
        n_samples: int,
        discrete_columns: Optional[ColumnList] = None,
        constraints: Optional[List[BaseConstraint]] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Fit the generator and generate synthetic data in one call.

        Args:
            data: Training DataFrame
            n_samples: Number of samples to generate
            discrete_columns: List of discrete/categorical column names
            constraints: List of constraints to enforce
            conditions: Optional conditions for conditional generation

        Returns:
            Generated DataFrame
        """
        self.fit(data, discrete_columns, constraints)
        return self.generate(n_samples, conditions)

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data.

        Args:
            data: DataFrame to validate

        Raises:
            ValidationError: If validation fails
        """
        if data is None:
            raise ValidationError("Input data cannot be None")

        if not isinstance(data, pd.DataFrame):
            raise ValidationError(f"Expected pandas DataFrame, got {type(data)}")

        if len(data) == 0:
            raise ValidationError("Input data cannot be empty")

        if len(data.columns) == 0:
            raise ValidationError("Input data must have at least one column")

    def get_parameters(self) -> Dict[str, Any]:
        """Get the generator's learned parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "is_fitted": self._is_fitted,
            "config": self.config.to_dict(),
            "privacy": self.privacy.to_dict(),
            "discrete_columns": self._discrete_columns,
            "n_constraints": len(self._constraints),
        }

    def get_serialization_state(self) -> Dict[str, Any]:
        """Get the generator's state for serialization.

        Returns:
            Dictionary containing all state needed to reconstruct the generator.
        """
        return {
            "config": self.config.to_dict(),
            "privacy": self.privacy.to_dict(),
            "is_fitted": self._is_fitted,
            "schema": self._schema.to_dict() if self._schema else None,
            "discrete_columns": self._discrete_columns,
            "constraints": [
                {"type": c.__class__.__name__, "column": c.column, "name": c.name}
                for c in self._constraints.constraints
            ],
        }

    def set_serialization_state(self, state: Dict[str, Any]) -> None:
        """Restore the generator's state from serialization.

        Args:
            state: Dictionary from get_serialization_state()
        """
        self.config = GeneratorConfig.from_dict(state.get("config", {}))
        self.privacy = PrivacyConfig.from_dict(state.get("privacy", {}))
        self._is_fitted = state.get("is_fitted", False)
        self._discrete_columns = state.get("discrete_columns", [])
        # Note: schema and constraints require full reconstruction in subclasses

    def save(self, path: str, use_safe_serialization: bool = True) -> None:
        """Save the generator to a file.

        Args:
            path: Path to save the generator
            use_safe_serialization: If True (default), use safe JSON+numpy format.
                                    If False, use pickle (not recommended for
                                    untrusted environments).

        Note:
            Safe serialization is recommended for security. Pickle can execute
            arbitrary code when loading untrusted files.
        """
        if use_safe_serialization:
            from genesis.utils.serialization import save_model

            save_model(self, path)
        else:
            import pickle

            import warnings

            warnings.warn(
                "Pickle serialization is not recommended for security reasons. "
                "Use safe_serialization=True (default) instead.",
                UserWarning,
                stacklevel=2,
            )
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str, use_safe_serialization: bool = True) -> "BaseGenerator":
        """Load a generator from a file.

        Args:
            path: Path to the saved generator
            use_safe_serialization: If True (default), use safe JSON+numpy format.
                                    If False, use pickle (SECURITY WARNING: only
                                    load files you trust completely).

        Returns:
            Loaded generator instance
        """
        if use_safe_serialization:
            from genesis.utils.serialization import load_model

            return load_model(path, cls)
        else:
            import pickle

            import warnings

            warnings.warn(
                "Loading pickle files can execute arbitrary code. "
                "Only load files from trusted sources.",
                UserWarning,
                stacklevel=2,
            )
            with open(path, "rb") as f:
                return pickle.load(f)

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"


class SyntheticGenerator(GeneratorExtensionsMixin, BaseGenerator):
    """High-level synthetic data generator with automatic method selection.

    This is the main entry point for most users. It automatically selects
    the best generation method based on data characteristics.

    Includes capabilities from mixins:
    - generate_conditional(): Rejection-sampled conditional generation
    - upsample(): Class imbalance correction
    - generate_scenarios(): Batch scenario generation
    - quality_report(): Automatic quality evaluation

    Example:
        >>> generator = SyntheticGenerator(method='auto')
        >>> generator.fit(real_data)
        >>> synthetic_data = generator.generate(n_samples=10000)
    """

    def __init__(
        self,
        method: str = "auto",
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
    ) -> None:
        """Initialize the synthetic generator.

        Args:
            method: Generation method ('auto', 'ctgan', 'tvae', 'gaussian_copula')
            config: Generator configuration
            privacy: Privacy configuration
        """
        config = config or GeneratorConfig()
        # Only override config.method if explicit method was passed
        if method != "auto" or config.method == GeneratorMethod.AUTO:
            config.method = (
                method if isinstance(method, GeneratorMethod) else GeneratorMethod(method)
            )
        super().__init__(config, privacy)

        self._inner_generator: Optional[BaseGenerator] = None

    @classmethod
    def from_config(cls, genesis_config: "GenesisConfig") -> "SyntheticGenerator":
        """Create a generator from a unified GenesisConfig.

        Args:
            genesis_config: Unified configuration object

        Returns:
            Configured SyntheticGenerator instance

        Example:
            >>> config = GenesisConfig.production(privacy_level="high")
            >>> generator = SyntheticGenerator.from_config(config)
        """
        from genesis.core.config import GenesisConfig

        if not isinstance(genesis_config, GenesisConfig):
            raise TypeError(f"Expected GenesisConfig, got {type(genesis_config)}")

        return cls(
            config=genesis_config.training,
            privacy=genesis_config.privacy,
        )

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[ColumnList] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit implementation using selected method."""
        from genesis.analyzers.schema import SchemaAnalyzer
        from genesis.generators.auto import select_generator

        # Analyze schema
        analyzer = SchemaAnalyzer()
        self._schema = analyzer.analyze(data)

        # Select and initialize the appropriate generator
        self._inner_generator = select_generator(
            data=data,
            method=self.config.method,
            config=self.config,
            privacy=self.privacy,
            schema=self._schema,
        )

        # Fit the inner generator
        return self._inner_generator._fit_impl(data, discrete_columns, progress_callback)

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate implementation using selected method."""
        if self._inner_generator is None:
            raise NotFittedError(self.__class__.__name__)

        return self._inner_generator._generate_impl(n_samples, conditions, progress_callback)
