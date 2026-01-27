"""Input validation utilities for Genesis."""

from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from genesis.core.exceptions import (
    ColumnNotFoundError,
    InsufficientDataError,
    ValidationError,
)


def validate_dataframe(
    data: Any,
    min_rows: int = 1,
    min_columns: int = 1,
    required_columns: Optional[List[str]] = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Validate that input is a valid DataFrame.

    Args:
        data: Input to validate
        min_rows: Minimum number of rows required
        min_columns: Minimum number of columns required
        required_columns: List of columns that must be present
        allow_empty: Whether to allow empty DataFrames

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If validation fails
    """
    if data is None:
        raise ValidationError("Data cannot be None")

    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            raise ValidationError(f"Cannot convert input to DataFrame: {e}") from e

    if not allow_empty:
        if len(data) == 0:
            raise ValidationError("DataFrame cannot be empty")

        if len(data.columns) == 0:
            raise ValidationError("DataFrame must have at least one column")

    if len(data) < min_rows:
        raise InsufficientDataError(required=min_rows, provided=len(data))

    if len(data.columns) < min_columns:
        raise ValidationError(
            f"DataFrame must have at least {min_columns} columns, got {len(data.columns)}"
        )

    if required_columns:
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ColumnNotFoundError(
                column=list(missing)[0],
                available_columns=list(data.columns),
            )

    return data


def validate_columns_exist(
    data: pd.DataFrame,
    columns: List[str],
    raise_on_missing: bool = True,
) -> Tuple[List[str], List[str]]:
    """Validate that specified columns exist in the DataFrame.

    Args:
        data: DataFrame to check
        columns: List of column names to verify
        raise_on_missing: Whether to raise exception on missing columns

    Returns:
        Tuple of (existing columns, missing columns)

    Raises:
        ColumnNotFoundError: If raise_on_missing=True and columns are missing
    """
    existing = [col for col in columns if col in data.columns]
    missing = [col for col in columns if col not in data.columns]

    if missing and raise_on_missing:
        raise ColumnNotFoundError(
            column=missing[0],
            available_columns=list(data.columns),
        )

    return existing, missing


def validate_numeric_columns(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> List[str]:
    """Get and validate numeric columns.

    Args:
        data: DataFrame to check
        columns: Specific columns to validate (None for all numeric)

    Returns:
        List of valid numeric column names

    Raises:
        ValidationError: If specified columns are not numeric
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if columns is None:
        return numeric_cols

    non_numeric = [col for col in columns if col not in numeric_cols]
    if non_numeric:
        raise ValidationError(
            f"Columns {non_numeric} are not numeric. " f"Numeric columns: {numeric_cols}"
        )

    return [col for col in columns if col in numeric_cols]


def validate_categorical_columns(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_cardinality: Optional[int] = None,
) -> List[str]:
    """Get and validate categorical columns.

    Args:
        data: DataFrame to check
        columns: Specific columns to validate (None for all categorical)
        max_cardinality: Maximum allowed number of unique values

    Returns:
        List of valid categorical column names

    Raises:
        ValidationError: If columns have too high cardinality
    """
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    if columns is None:
        columns = cat_cols
    else:
        # Verify specified columns exist
        validate_columns_exist(data, columns)

    if max_cardinality:
        high_cardinality = [col for col in columns if data[col].nunique() > max_cardinality]
        if high_cardinality:
            raise ValidationError(
                f"Columns {high_cardinality} have more than {max_cardinality} unique values"
            )

    return columns


def validate_positive(value: Any, name: str = "value") -> float:
    """Validate that a value is positive.

    Args:
        value: Value to validate
        name: Name of the parameter (for error message)

    Returns:
        Validated positive float

    Raises:
        ValidationError: If value is not positive
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number, got {type(value)}") from e

    if num_value <= 0:
        raise ValidationError(f"{name} must be positive, got {num_value}")

    return num_value


def validate_non_negative(value: Any, name: str = "value") -> float:
    """Validate that a value is non-negative.

    Args:
        value: Value to validate
        name: Name of the parameter (for error message)

    Returns:
        Validated non-negative float

    Raises:
        ValidationError: If value is negative
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number, got {type(value)}") from e

    if num_value < 0:
        raise ValidationError(f"{name} must be non-negative, got {num_value}")

    return num_value


def validate_range(
    value: Any,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
    inclusive: bool = True,
) -> float:
    """Validate that a value is within a range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (None for no minimum)
        max_val: Maximum allowed value (None for no maximum)
        name: Name of the parameter (for error message)
        inclusive: Whether bounds are inclusive

    Returns:
        Validated float within range

    Raises:
        ValidationError: If value is out of range
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be a number, got {type(value)}") from e

    if min_val is not None:
        if inclusive and num_value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {num_value}")
        if not inclusive and num_value <= min_val:
            raise ValidationError(f"{name} must be > {min_val}, got {num_value}")

    if max_val is not None:
        if inclusive and num_value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {num_value}")
        if not inclusive and num_value >= max_val:
            raise ValidationError(f"{name} must be < {max_val}, got {num_value}")

    return num_value


def validate_probability(value: Any, name: str = "probability") -> float:
    """Validate that a value is a valid probability [0, 1].

    Args:
        value: Value to validate
        name: Name of the parameter (for error message)

    Returns:
        Validated probability

    Raises:
        ValidationError: If value is not in [0, 1]
    """
    return validate_range(value, 0.0, 1.0, name)


def validate_integer(
    value: Any,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    name: str = "value",
) -> int:
    """Validate that a value is an integer within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter (for error message)

    Returns:
        Validated integer

    Raises:
        ValidationError: If validation fails
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be an integer, got {type(value)}") from e

    if min_val is not None and int_value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {int_value}")

    if max_val is not None and int_value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {int_value}")

    return int_value


def validate_choice(
    value: Any,
    choices: List[Any],
    name: str = "value",
) -> Any:
    """Validate that a value is one of the allowed choices.

    Args:
        value: Value to validate
        choices: List of allowed values
        name: Name of the parameter (for error message)

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not in choices
    """
    if value not in choices:
        raise ValidationError(f"{name} must be one of {choices}, got {value}")
    return value


def validate_type(
    value: Any,
    expected_type: Union[Type, Tuple[Type, ...]],
    name: str = "value",
) -> Any:
    """Validate that a value is of the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        name: Name of the parameter (for error message)

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        type_names = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        raise ValidationError(f"{name} must be of type {type_names}, got {type(value).__name__}")
    return value


def validate_callable(value: Any, name: str = "value") -> Callable:
    """Validate that a value is callable.

    Args:
        value: Value to validate
        name: Name of the parameter (for error message)

    Returns:
        Validated callable

    Raises:
        ValidationError: If value is not callable
    """
    if not callable(value):
        raise ValidationError(f"{name} must be callable, got {type(value)}")
    return value
