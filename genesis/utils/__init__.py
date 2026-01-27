"""Utility functions for Genesis."""

from genesis.utils.logging import (
    ProgressLogger,
    TrainingLogger,
    console,
    create_progress,
    get_logger,
    set_log_level,
)
from genesis.utils.preprocessing import (
    balance_classes,
    clip_outliers,
    detect_outliers,
    handle_missing_values,
    normalize_column_names,
    remove_constant_columns,
    remove_duplicate_rows,
    split_data,
)
from genesis.utils.transformers import (
    BaseTransformer,
    CategoricalTransformer,
    DataTransformer,
    DatetimeTransformer,
    NumericalTransformer,
)
from genesis.utils.validation import (
    validate_callable,
    validate_categorical_columns,
    validate_choice,
    validate_columns_exist,
    validate_dataframe,
    validate_integer,
    validate_non_negative,
    validate_numeric_columns,
    validate_positive,
    validate_probability,
    validate_range,
    validate_type,
)

__all__ = [
    # Preprocessing
    "handle_missing_values",
    "detect_outliers",
    "clip_outliers",
    "normalize_column_names",
    "balance_classes",
    "split_data",
    "remove_constant_columns",
    "remove_duplicate_rows",
    # Transformers
    "BaseTransformer",
    "NumericalTransformer",
    "CategoricalTransformer",
    "DatetimeTransformer",
    "DataTransformer",
    # Logging
    "get_logger",
    "set_log_level",
    "create_progress",
    "ProgressLogger",
    "TrainingLogger",
    "console",
    # Validation
    "validate_dataframe",
    "validate_columns_exist",
    "validate_numeric_columns",
    "validate_categorical_columns",
    "validate_positive",
    "validate_non_negative",
    "validate_range",
    "validate_probability",
    "validate_integer",
    "validate_choice",
    "validate_type",
    "validate_callable",
]
