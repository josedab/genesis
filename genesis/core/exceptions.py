"""Custom exceptions for Genesis."""

from typing import Any, Optional


class GenesisError(Exception):
    """Base exception for all Genesis errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(GenesisError):
    """Error in configuration parameters."""

    pass


class ValidationError(GenesisError):
    """Error in input validation."""

    pass


class FittingError(GenesisError):
    """Error during model fitting."""

    pass


class GenerationError(GenesisError):
    """Error during synthetic data generation."""

    pass


class PrivacyError(GenesisError):
    """Error related to privacy constraints."""

    pass


class ConstraintViolationError(GenesisError):
    """Error when data constraints are violated."""

    def __init__(
        self,
        message: str,
        constraint_name: Optional[str] = None,
        column: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.constraint_name = constraint_name
        self.column = column


class BackendNotAvailableError(GenesisError):
    """Error when required backend (PyTorch/TensorFlow) is not available."""

    def __init__(self, backend: str, message: Optional[str] = None) -> None:
        default_msg = f"Backend '{backend}' is not available. Please install it with: pip install genesis-synth[{backend}]"
        super().__init__(message or default_msg)
        self.backend = backend


class DataError(GenesisError):
    """Error related to input data issues."""

    pass


class SchemaError(GenesisError):
    """Error in data schema inference or validation."""

    pass


class EvaluationError(GenesisError):
    """Error during quality evaluation."""

    pass


class NotFittedError(GenesisError):
    """Error when trying to generate before fitting."""

    def __init__(self, generator_name: str = "Generator") -> None:
        super().__init__(f"{generator_name} has not been fitted. Call fit() before generate().")
        self.generator_name = generator_name


class InsufficientDataError(DataError):
    """Error when there is not enough data for training."""

    def __init__(self, required: int, provided: int, message: Optional[str] = None) -> None:
        default_msg = f"Insufficient data: required at least {required} samples, but got {provided}"
        super().__init__(message or default_msg)
        self.required = required
        self.provided = provided


class ColumnNotFoundError(DataError):
    """Error when a specified column is not found in the data."""

    def __init__(self, column: str, available_columns: Optional[list[str]] = None) -> None:
        msg = f"Column '{column}' not found in data"
        if available_columns:
            msg += f". Available columns: {available_columns}"
        super().__init__(msg)
        self.column = column
        self.available_columns = available_columns


class UnsupportedDataTypeError(DataError):
    """Error when data type is not supported."""

    def __init__(self, dtype: str, column: Optional[str] = None) -> None:
        msg = f"Unsupported data type: {dtype}"
        if column:
            msg += f" in column '{column}'"
        super().__init__(msg)
        self.dtype = dtype
        self.column = column
