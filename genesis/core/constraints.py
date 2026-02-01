"""Constraint definitions for synthetic data generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from genesis.core.exceptions import ConstraintViolationError

# =============================================================================
# CONSTANTS
# =============================================================================

# Delta adjustments for constraint enforcement
RELATIVE_DELTA_FACTOR = 0.01  # 1% relative adjustment
ABSOLUTE_DELTA_EPSILON = 1e-6  # Small epsilon to ensure strict inequality

# Constraint iteration limits
DEFAULT_MAX_ITERATIONS = 10


class BaseConstraint(ABC):
    """Abstract base class for data constraints."""

    def __init__(self, column: str, name: Optional[str] = None) -> None:
        self.column = column
        self.name = name or self.__class__.__name__

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Check if data satisfies the constraint."""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data to satisfy the constraint."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(column='{self.column}')"


class PositiveConstraint(BaseConstraint):
    """Constraint that values must be positive (> 0)."""

    def __init__(self, column: str, strict: bool = True) -> None:
        super().__init__(column, "positive")
        self.strict = strict

    def validate(self, data: pd.DataFrame) -> bool:
        if self.column not in data.columns:
            return True
        values = data[self.column].dropna()
        if self.strict:
            return (values > 0).all()
        return (values >= 0).all()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            return data
        result = data.copy()
        if self.strict:
            mask = result[self.column] <= 0
            if mask.any():
                new_values = np.abs(result.loc[mask, self.column].astype(float)) + 1e-6
                result[self.column] = result[self.column].astype(float)
                result.loc[mask, self.column] = new_values
        else:
            mask = result[self.column] < 0
            if mask.any():
                new_values = np.abs(result.loc[mask, self.column].astype(float))
                result[self.column] = result[self.column].astype(float)
                result.loc[mask, self.column] = new_values
        return result


class NonNegativeConstraint(BaseConstraint):
    """Constraint that values must be non-negative (>= 0)."""

    def __init__(self, column: str) -> None:
        super().__init__(column, "non_negative")

    def validate(self, data: pd.DataFrame) -> bool:
        if self.column not in data.columns:
            return True
        values = data[self.column].dropna()
        return (values >= 0).all()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            return data
        result = data.copy()
        result.loc[result[self.column] < 0, self.column] = 0
        return result


class RangeConstraint(BaseConstraint):
    """Constraint that values must be within a specified range."""

    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        inclusive: bool = True,
    ) -> None:
        super().__init__(column, "range")
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(self, data: pd.DataFrame) -> bool:
        if self.column not in data.columns:
            return True
        values = data[self.column].dropna()
        if len(values) == 0:
            return True

        valid = np.ones(len(values), dtype=bool)
        if self.min_value is not None:
            if self.inclusive:
                valid &= values >= self.min_value
            else:
                valid &= values > self.min_value
        if self.max_value is not None:
            if self.inclusive:
                valid &= values <= self.max_value
            else:
                valid &= values < self.max_value
        return valid.all()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            return data
        result = data.copy()
        if self.min_value is not None:
            result.loc[result[self.column] < self.min_value, self.column] = self.min_value
        if self.max_value is not None:
            result.loc[result[self.column] > self.max_value, self.column] = self.max_value
        return result

    def __repr__(self) -> str:
        return (
            f"RangeConstraint(column='{self.column}', min={self.min_value}, max={self.max_value})"
        )


class UniqueConstraint(BaseConstraint):
    """Constraint that values must be unique."""

    def __init__(self, column: str) -> None:
        super().__init__(column, "unique")

    def validate(self, data: pd.DataFrame) -> bool:
        if self.column not in data.columns:
            return True
        values = data[self.column].dropna()
        return len(values) == len(values.unique())

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            return data
        result = data.copy()
        seen: Set[Any] = set()
        unique_values = []

        for val in result[self.column]:
            if pd.isna(val):
                unique_values.append(val)
            elif val in seen:
                # Generate a unique value
                if isinstance(val, (int, np.integer)):
                    new_val = val
                    while new_val in seen:
                        new_val = new_val + 1
                    unique_values.append(new_val)
                    seen.add(new_val)
                elif isinstance(val, str):
                    counter = 1
                    new_val = f"{val}_{counter}"
                    while new_val in seen:
                        counter += 1
                        new_val = f"{val}_{counter}"
                    unique_values.append(new_val)
                    seen.add(new_val)
                else:
                    unique_values.append(val)
                    seen.add(val)
            else:
                unique_values.append(val)
                seen.add(val)

        result[self.column] = unique_values
        return result


class CategoricalConstraint(BaseConstraint):
    """Constraint that values must be from a specified set of categories."""

    def __init__(self, column: str, categories: List[Any]) -> None:
        super().__init__(column, "categorical")
        self.categories = set(categories)

    def validate(self, data: pd.DataFrame) -> bool:
        if self.column not in data.columns:
            return True
        values = data[self.column].dropna()
        return all(v in self.categories for v in values)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            return data
        result = data.copy()
        # Map invalid values to the most common category
        categories_list = list(self.categories)
        mask = ~result[self.column].isin(self.categories) & result[self.column].notna()
        if mask.any():
            # Use the first category as default
            result.loc[mask, self.column] = categories_list[0]
        return result


class CustomConstraint(BaseConstraint):
    """Custom constraint with user-defined validation and transformation functions."""

    def __init__(
        self,
        column: str,
        validate_fn: Callable[[pd.Series], bool],
        transform_fn: Callable[[pd.Series], pd.Series],
        name: str = "custom",
    ) -> None:
        super().__init__(column, name)
        self.validate_fn = validate_fn
        self.transform_fn = transform_fn

    def validate(self, data: pd.DataFrame) -> bool:
        if self.column not in data.columns:
            return True
        return self.validate_fn(data[self.column])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            return data
        result = data.copy()
        result[self.column] = self.transform_fn(result[self.column])
        return result


class GreaterThanConstraint(BaseConstraint):
    """Constraint that one column must be greater than another."""

    def __init__(
        self,
        column: str,
        other_column: str,
        or_equal: bool = False,
    ) -> None:
        super().__init__(column, "greater_than")
        self.other_column = other_column
        self.or_equal = or_equal

    def validate(self, data: pd.DataFrame) -> bool:
        if self.column not in data.columns or self.other_column not in data.columns:
            return True
        mask = data[self.column].notna() & data[self.other_column].notna()
        if self.or_equal:
            return (data.loc[mask, self.column] >= data.loc[mask, self.other_column]).all()
        return (data.loc[mask, self.column] > data.loc[mask, self.other_column]).all()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns or self.other_column not in data.columns:
            return data
        result = data.copy()
        mask = result[self.column].notna() & result[self.other_column].notna()

        if self.or_equal:
            violating = mask & (result[self.column] < result[self.other_column])
        else:
            violating = mask & (result[self.column] <= result[self.other_column])

        if violating.any():
            # Swap values or add small delta
            if self.or_equal:
                result.loc[violating, self.column] = result.loc[violating, self.other_column]
            else:
                delta = (
                    np.abs(result.loc[violating, self.other_column]) * RELATIVE_DELTA_FACTOR
                    + ABSOLUTE_DELTA_EPSILON
                )
                result.loc[violating, self.column] = (
                    result.loc[violating, self.other_column] + delta
                )

        return result

    def __repr__(self) -> str:
        op = ">=" if self.or_equal else ">"
        return f"GreaterThanConstraint({self.column} {op} {self.other_column})"


class Constraint:
    """Factory class for creating constraints.

    Example:
        >>> constraints = [
        ...     Constraint.positive('age'),
        ...     Constraint.range('age', 0, 120),
        ...     Constraint.unique('customer_id'),
        ...     Constraint.categorical('status', ['active', 'inactive']),
        ... ]
    """

    @staticmethod
    def positive(column: str, strict: bool = True) -> PositiveConstraint:
        """Create a positive value constraint.

        Args:
            column: Column name
            strict: If True, values must be > 0. If False, values must be >= 0.

        Returns:
            PositiveConstraint instance
        """
        return PositiveConstraint(column, strict=strict)

    @staticmethod
    def non_negative(column: str) -> NonNegativeConstraint:
        """Create a non-negative value constraint (>= 0).

        Args:
            column: Column name

        Returns:
            NonNegativeConstraint instance
        """
        return NonNegativeConstraint(column)

    @staticmethod
    def range(
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        inclusive: bool = True,
    ) -> RangeConstraint:
        """Create a range constraint.

        Args:
            column: Column name
            min_value: Minimum allowed value (None for no minimum)
            max_value: Maximum allowed value (None for no maximum)
            inclusive: Whether boundaries are inclusive

        Returns:
            RangeConstraint instance
        """
        return RangeConstraint(column, min_value, max_value, inclusive)

    @staticmethod
    def unique(column: str) -> UniqueConstraint:
        """Create a uniqueness constraint.

        Args:
            column: Column name

        Returns:
            UniqueConstraint instance
        """
        return UniqueConstraint(column)

    @staticmethod
    def categorical(column: str, categories: List[Any]) -> CategoricalConstraint:
        """Create a categorical constraint.

        Args:
            column: Column name
            categories: List of allowed categories

        Returns:
            CategoricalConstraint instance
        """
        return CategoricalConstraint(column, categories)

    @staticmethod
    def greater_than(
        column: str, other_column: str, or_equal: bool = False
    ) -> GreaterThanConstraint:
        """Create a constraint that one column must be greater than another.

        Args:
            column: Column that must be greater
            other_column: Column to compare against
            or_equal: If True, allows equal values

        Returns:
            GreaterThanConstraint instance
        """
        return GreaterThanConstraint(column, other_column, or_equal)

    @staticmethod
    def custom(
        column: str,
        validate_fn: Callable[[pd.Series], bool],
        transform_fn: Callable[[pd.Series], pd.Series],
        name: str = "custom",
    ) -> CustomConstraint:
        """Create a custom constraint.

        Args:
            column: Column name
            validate_fn: Function that returns True if constraint is satisfied
            transform_fn: Function that transforms values to satisfy constraint
            name: Name for the constraint

        Returns:
            CustomConstraint instance
        """
        return CustomConstraint(column, validate_fn, transform_fn, name)


@dataclass
class ConstraintSet:
    """Collection of constraints with validation and transformation methods."""

    constraints: List[BaseConstraint] = field(default_factory=list)

    def add(self, constraint: BaseConstraint) -> "ConstraintSet":
        """Add a constraint to the set."""
        self.constraints.append(constraint)
        return self

    def validate(self, data: pd.DataFrame, raise_on_error: bool = False) -> Dict[str, bool]:
        """Validate all constraints against the data.

        Args:
            data: DataFrame to validate
            raise_on_error: If True, raise exception on first violation

        Returns:
            Dictionary mapping constraint names to validation results
        """
        results = {}
        for constraint in self.constraints:
            key = f"{constraint.name}:{constraint.column}"
            is_valid = constraint.validate(data)
            results[key] = is_valid

            if raise_on_error and not is_valid:
                raise ConstraintViolationError(
                    f"Constraint '{constraint.name}' violated for column '{constraint.column}'",
                    constraint_name=constraint.name,
                    column=constraint.column,
                )
        return results

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all constraint transformations to the data.

        Args:
            data: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        for constraint in self.constraints:
            result = constraint.transform(result)
        return result

    def validate_and_transform(
        self, data: pd.DataFrame, max_iterations: int = DEFAULT_MAX_ITERATIONS
    ) -> Tuple[pd.DataFrame, Dict[str, bool]]:
        """Validate and iteratively transform until all constraints are satisfied.

        Args:
            data: DataFrame to process
            max_iterations: Maximum transformation iterations

        Returns:
            Tuple of (transformed DataFrame, final validation results)
        """
        result = data.copy()
        for _ in range(max_iterations):
            validation_results = self.validate(result)
            if all(validation_results.values()):
                return result, validation_results
            result = self.transform(result)

        return result, self.validate(result)

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self):
        return iter(self.constraints)
