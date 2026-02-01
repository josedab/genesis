"""Type definitions for Genesis."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Type aliases
ArrayLike = Union[np.ndarray, pd.Series, List[Any]]
DataFrameLike = Union[pd.DataFrame, Dict[str, ArrayLike]]
NumericType = Union[int, float, np.number]
ColumnName = str
ColumnList = List[ColumnName]


class ColumnType(Enum):
    """Enumeration of column data types."""

    NUMERIC_CONTINUOUS = auto()
    NUMERIC_DISCRETE = auto()
    CATEGORICAL = auto()
    BOOLEAN = auto()
    DATETIME = auto()
    TEXT = auto()
    IDENTIFIER = auto()
    UNKNOWN = auto()


class GeneratorMethod(Enum):
    """Enumeration of available generator methods."""

    AUTO = "auto"
    CTGAN = "ctgan"
    TVAE = "tvae"
    GAUSSIAN_COPULA = "gaussian_copula"
    TIMEGAN = "timegan"
    STATISTICAL = "statistical"
    LLM = "llm"


class BackendType(Enum):
    """Enumeration of deep learning backends."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    AUTO = "auto"


class PrivacyLevel(Enum):
    """Enumeration of privacy protection levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""

    name: str
    dtype: ColumnType
    nullable: bool = True
    unique: bool = False
    cardinality: Optional[int] = None
    min_value: Optional[NumericType] = None
    max_value: Optional[NumericType] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    categories: Optional[List[Any]] = None
    missing_rate: float = 0.0
    is_quasi_identifier: bool = False
    is_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype.name,
            "nullable": self.nullable,
            "unique": self.unique,
            "cardinality": self.cardinality,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": self.mean,
            "std": self.std,
            "categories": self.categories,
            "missing_rate": self.missing_rate,
            "is_quasi_identifier": self.is_quasi_identifier,
            "is_sensitive": self.is_sensitive,
        }


@dataclass
class DataSchema:
    """Schema information for a dataset."""

    columns: Dict[str, ColumnMetadata] = field(default_factory=dict)
    n_rows: int = 0
    n_columns: int = 0
    primary_key: Optional[ColumnName] = None
    foreign_keys: Dict[ColumnName, Tuple[str, ColumnName]] = field(default_factory=dict)

    def get_column_names(self, dtype: Optional[ColumnType] = None) -> List[str]:
        """Get column names, optionally filtered by type."""
        if dtype is None:
            return list(self.columns.keys())
        return [name for name, meta in self.columns.items() if meta.dtype == dtype]

    def get_numeric_columns(self) -> List[str]:
        """Get names of numeric columns."""
        return [
            name
            for name, meta in self.columns.items()
            if meta.dtype in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE)
        ]

    def get_categorical_columns(self) -> List[str]:
        """Get names of categorical columns."""
        return [
            name
            for name, meta in self.columns.items()
            if meta.dtype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "columns": {name: meta.to_dict() for name, meta in self.columns.items()},
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "primary_key": self.primary_key,
            "foreign_keys": self.foreign_keys,
        }


@dataclass
class GenerationResult:
    """Result of synthetic data generation."""

    data: pd.DataFrame
    n_samples: int
    method: str
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_samples == 0:
            self.n_samples = len(self.data)


@dataclass
class FittingResult:
    """Result of model fitting."""

    success: bool
    fitting_time: float
    n_epochs: Optional[int] = None
    final_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetric:
    """A single evaluation metric result."""

    name: str
    value: float
    description: str = ""
    column: Optional[str] = None
    threshold: Optional[float] = None
    passed: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.threshold is not None and self.passed is None:
            self.passed = self.value >= self.threshold


@dataclass
class PrivacyMetrics:
    """Collection of privacy-related metrics."""

    reidentification_risk: float = 0.0
    attribute_disclosure_risk: float = 0.0
    membership_inference_risk: float = 0.0
    distance_to_closest_record: float = 0.0
    k_anonymity_level: Optional[int] = None
    l_diversity_level: Optional[int] = None
    epsilon_spent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reidentification_risk": self.reidentification_risk,
            "attribute_disclosure_risk": self.attribute_disclosure_risk,
            "membership_inference_risk": self.membership_inference_risk,
            "distance_to_closest_record": self.distance_to_closest_record,
            "k_anonymity_level": self.k_anonymity_level,
            "l_diversity_level": self.l_diversity_level,
            "epsilon_spent": self.epsilon_spent,
        }


# Callback types
ProgressCallback = Callable[[int, int, Dict[str, Any]], None]
EarlyStoppingCallback = Callable[[int, float], bool]


@dataclass
class TrainingEvent:
    """Event data passed to training callbacks.

    Attributes:
        step: Current training step (epoch or batch)
        total_steps: Total number of steps (if known)
        phase: Training phase (e.g., 'fitting', 'generating', 'evaluating')
        metrics: Dictionary of metric values
        message: Optional human-readable message
    """

    step: int
    total_steps: Optional[int] = None
    phase: str = "training"
    metrics: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None

    @property
    def progress(self) -> float:
        """Get progress as fraction (0.0-1.0)."""
        if self.total_steps and self.total_steps > 0:
            return min(self.step / self.total_steps, 1.0)
        return 0.0


class TrainingCallback:
    """Protocol-like base class for training callbacks.

    Implement any subset of these methods to receive training events.
    All methods are optional - unimplemented methods are no-ops.

    Example:
        >>> class MyCallback(TrainingCallback):
        ...     def on_epoch_end(self, event: TrainingEvent) -> None:
        ...         print(f"Epoch {event.step}: loss={event.metrics.get('loss')}")
        ...
        >>> generator.fit(data, callbacks=[MyCallback()])
    """

    def on_fit_start(self, event: TrainingEvent) -> None:
        """Called when fitting begins."""
        pass

    def on_fit_end(self, event: TrainingEvent) -> None:
        """Called when fitting completes."""
        pass

    def on_epoch_start(self, event: TrainingEvent) -> None:
        """Called at the start of each training epoch."""
        pass

    def on_epoch_end(self, event: TrainingEvent) -> None:
        """Called at the end of each training epoch."""
        pass

    def on_batch_start(self, event: TrainingEvent) -> None:
        """Called at the start of each training batch."""
        pass

    def on_batch_end(self, event: TrainingEvent) -> None:
        """Called at the end of each training batch."""
        pass

    def on_generate_start(self, event: TrainingEvent) -> None:
        """Called when generation begins."""
        pass

    def on_generate_end(self, event: TrainingEvent) -> None:
        """Called when generation completes."""
        pass

    def on_error(self, event: TrainingEvent, error: Exception) -> None:
        """Called when an error occurs during training."""
        pass


class ProgressLoggerCallback(TrainingCallback):
    """Simple callback that logs progress to the console.

    Example:
        >>> generator.fit(data, callbacks=[ProgressLoggerCallback()])
    """

    def __init__(self, log_frequency: int = 10) -> None:
        """Initialize the callback.

        Args:
            log_frequency: Log every N epochs
        """
        self.log_frequency = log_frequency

    def on_epoch_end(self, event: TrainingEvent) -> None:
        """Log epoch progress."""
        if event.step % self.log_frequency == 0:
            loss = event.metrics.get("loss", "N/A")
            print(f"Epoch {event.step}/{event.total_steps or '?'}: loss={loss}")

    def on_fit_end(self, event: TrainingEvent) -> None:
        """Log completion."""
        print(f"Training completed in {event.step} epochs")
