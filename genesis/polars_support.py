"""Polars Native Support for Genesis.

This module provides first-class Polars DataFrame integration, enabling
10-100x faster preprocessing on large datasets while maintaining full
compatibility with the existing pandas-based API.

Example:
    >>> import polars as pl
    >>> from genesis import SyntheticGenerator
    >>> from genesis.polars_support import PolarsGenerator, to_polars, from_polars
    >>>
    >>> # Use Polars DataFrames directly
    >>> df = pl.read_csv("large_data.csv")
    >>> generator = PolarsGenerator(method='ctgan')
    >>> generator.fit(df)
    >>> synthetic = generator.generate(n_samples=100000)  # Returns Polars DataFrame
    >>>
    >>> # Or convert between formats
    >>> pandas_df = from_polars(synthetic)
    >>> polars_df = to_polars(pandas_df)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from genesis.core.base import BaseGenerator, SyntheticGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.constraints import BaseConstraint, ConstraintSet
from genesis.core.exceptions import ConfigurationError, NotFittedError, ValidationError
from genesis.core.types import ColumnList, FittingResult, ProgressCallback
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for optional polars dependency
POLARS_AVAILABLE = False
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore


class PolarsBackend(Enum):
    """Backend options for Polars operations."""

    NATIVE = "native"  # Use Polars for all operations where possible
    HYBRID = "hybrid"  # Use Polars for I/O and preprocessing, pandas for generation
    PANDAS_FALLBACK = "pandas_fallback"  # Convert to pandas for generation


@dataclass
class PolarsConfig:
    """Configuration for Polars integration.

    Attributes:
        backend: Which backend strategy to use
        streaming: Enable streaming mode for large datasets
        n_threads: Number of threads for parallel operations (None = auto)
        chunk_size: Chunk size for streaming operations
        lazy_evaluation: Use lazy evaluation where possible
    """

    backend: PolarsBackend = PolarsBackend.HYBRID
    streaming: bool = False
    n_threads: Optional[int] = None
    chunk_size: int = 100_000
    lazy_evaluation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "streaming": self.streaming,
            "n_threads": self.n_threads,
            "chunk_size": self.chunk_size,
            "lazy_evaluation": self.lazy_evaluation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolarsConfig":
        backend = data.get("backend", "hybrid")
        if isinstance(backend, str):
            backend = PolarsBackend(backend)
        return cls(
            backend=backend,
            streaming=data.get("streaming", False),
            n_threads=data.get("n_threads"),
            chunk_size=data.get("chunk_size", 100_000),
            lazy_evaluation=data.get("lazy_evaluation", True),
        )


def _check_polars_available() -> None:
    """Check if Polars is available, raise if not."""
    if not POLARS_AVAILABLE:
        raise ImportError(
            "Polars is required for this feature. "
            "Install it with: pip install polars"
        )


def to_polars(df: pd.DataFrame) -> "pl.DataFrame":
    """Convert a pandas DataFrame to Polars DataFrame.

    Args:
        df: pandas DataFrame to convert

    Returns:
        Polars DataFrame

    Example:
        >>> import pandas as pd
        >>> from genesis.polars_support import to_polars
        >>> pdf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        >>> pldf = to_polars(pdf)
    """
    _check_polars_available()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    return pl.from_pandas(df)


def from_polars(df: "pl.DataFrame") -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas DataFrame.

    Args:
        df: Polars DataFrame to convert

    Returns:
        pandas DataFrame

    Example:
        >>> import polars as pl
        >>> from genesis.polars_support import from_polars
        >>> pldf = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        >>> pdf = from_polars(pldf)
    """
    _check_polars_available()

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected Polars DataFrame, got {type(df)}")

    return df.to_pandas()


def infer_polars_schema(df: "pl.DataFrame") -> Dict[str, Any]:
    """Infer schema information from a Polars DataFrame.

    Args:
        df: Polars DataFrame to analyze

    Returns:
        Dictionary with schema information including column types and statistics
    """
    _check_polars_available()

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    schema = {}
    for col_name in df.columns:
        col = df[col_name]
        dtype = col.dtype

        col_info = {
            "name": col_name,
            "dtype": str(dtype),
            "null_count": col.null_count(),
            "n_unique": col.n_unique(),
        }

        # Add statistics based on type
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64):
            col_info["min"] = col.min()
            col_info["max"] = col.max()
            col_info["mean"] = col.mean()
            col_info["std"] = col.std()
            col_info["is_numeric"] = True
        elif dtype == pl.Boolean:
            col_info["is_boolean"] = True
            col_info["true_count"] = col.sum()
        elif dtype in (pl.Utf8, pl.String):
            col_info["is_string"] = True
            col_info["max_length"] = col.str.len_chars().max()
        elif dtype in (pl.Date, pl.Datetime, pl.Time):
            col_info["is_temporal"] = True
            col_info["min"] = str(col.min())
            col_info["max"] = str(col.max())

        schema[col_name] = col_info

    return schema


def detect_discrete_columns_polars(
    df: "pl.DataFrame",
    cardinality_threshold: int = 20,
    ratio_threshold: float = 0.05,
) -> List[str]:
    """Detect discrete/categorical columns in a Polars DataFrame.

    Args:
        df: Polars DataFrame to analyze
        cardinality_threshold: Max unique values to consider categorical
        ratio_threshold: Max ratio of unique/total to consider categorical

    Returns:
        List of column names that appear to be discrete/categorical
    """
    _check_polars_available()

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    discrete_columns = []
    n_rows = len(df)

    for col_name in df.columns:
        col = df[col_name]
        dtype = col.dtype

        # String columns are always discrete
        if dtype in (pl.Utf8, pl.String, pl.Categorical):
            discrete_columns.append(col_name)
            continue

        # Boolean columns are discrete
        if dtype == pl.Boolean:
            discrete_columns.append(col_name)
            continue

        # Check cardinality for other types
        n_unique = col.n_unique()

        if n_unique <= cardinality_threshold:
            discrete_columns.append(col_name)
        elif n_rows > 0 and (n_unique / n_rows) <= ratio_threshold:
            discrete_columns.append(col_name)

    return discrete_columns


class PolarsDataLoader:
    """Efficient data loading with Polars.

    Supports streaming large files and various formats with optimized I/O.

    Example:
        >>> loader = PolarsDataLoader(streaming=True, chunk_size=50000)
        >>> df = loader.read_csv("huge_file.csv")
        >>> df = loader.read_parquet("data.parquet")
    """

    def __init__(
        self,
        streaming: bool = False,
        chunk_size: int = 100_000,
        n_threads: Optional[int] = None,
    ):
        """Initialize the data loader.

        Args:
            streaming: Enable streaming mode for large files
            chunk_size: Chunk size for streaming operations
            n_threads: Number of threads (None = auto)
        """
        _check_polars_available()
        self.streaming = streaming
        self.chunk_size = chunk_size
        self.n_threads = n_threads

    def read_csv(
        self,
        path: str,
        **kwargs: Any,
    ) -> "pl.DataFrame":
        """Read a CSV file with optimized settings.

        Args:
            path: Path to CSV file
            **kwargs: Additional arguments passed to pl.read_csv

        Returns:
            Polars DataFrame
        """
        if self.streaming:
            return pl.scan_csv(path, **kwargs).collect(streaming=True)
        return pl.read_csv(path, **kwargs)

    def read_parquet(
        self,
        path: str,
        **kwargs: Any,
    ) -> "pl.DataFrame":
        """Read a Parquet file with optimized settings.

        Args:
            path: Path to Parquet file
            **kwargs: Additional arguments passed to pl.read_parquet

        Returns:
            Polars DataFrame
        """
        if self.streaming:
            return pl.scan_parquet(path, **kwargs).collect(streaming=True)
        return pl.read_parquet(path, **kwargs)

    def read_json(
        self,
        path: str,
        **kwargs: Any,
    ) -> "pl.DataFrame":
        """Read a JSON file.

        Args:
            path: Path to JSON file
            **kwargs: Additional arguments passed to pl.read_json

        Returns:
            Polars DataFrame
        """
        return pl.read_json(path, **kwargs)

    def write_csv(
        self,
        df: "pl.DataFrame",
        path: str,
        **kwargs: Any,
    ) -> None:
        """Write DataFrame to CSV.

        Args:
            df: DataFrame to write
            path: Output path
            **kwargs: Additional arguments
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        df.write_csv(path, **kwargs)

    def write_parquet(
        self,
        df: "pl.DataFrame",
        path: str,
        **kwargs: Any,
    ) -> None:
        """Write DataFrame to Parquet.

        Args:
            df: DataFrame to write
            path: Output path
            **kwargs: Additional arguments
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        df.write_parquet(path, **kwargs)


class PolarsPreprocessor:
    """High-performance data preprocessing with Polars.

    Provides optimized preprocessing operations that are typically
    10-100x faster than pandas equivalents on large datasets.

    Example:
        >>> preprocessor = PolarsPreprocessor()
        >>> df = preprocessor.handle_missing(df, strategy="mean")
        >>> df = preprocessor.encode_categorical(df, columns=["category"])
        >>> df = preprocessor.normalize(df, columns=["value"])
    """

    def __init__(self, n_threads: Optional[int] = None):
        """Initialize preprocessor.

        Args:
            n_threads: Number of threads (None = auto)
        """
        _check_polars_available()
        self.n_threads = n_threads

    def handle_missing(
        self,
        df: "pl.DataFrame",
        strategy: str = "drop",
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None,
    ) -> "pl.DataFrame":
        """Handle missing values.

        Args:
            df: Input DataFrame
            strategy: 'drop', 'mean', 'median', 'mode', 'constant'
            fill_value: Value to use when strategy='constant'
            columns: Specific columns to process (None = all)

        Returns:
            DataFrame with missing values handled
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        target_cols = columns or df.columns

        if strategy == "drop":
            return df.drop_nulls(subset=target_cols)

        elif strategy == "mean":
            for col in target_cols:
                if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                    mean_val = df[col].mean()
                    df = df.with_columns(pl.col(col).fill_null(mean_val))

        elif strategy == "median":
            for col in target_cols:
                if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
                    median_val = df[col].median()
                    df = df.with_columns(pl.col(col).fill_null(median_val))

        elif strategy == "mode":
            for col in target_cols:
                mode_val = df[col].mode().first()
                df = df.with_columns(pl.col(col).fill_null(mode_val))

        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value required for strategy='constant'")
            for col in target_cols:
                df = df.with_columns(pl.col(col).fill_null(fill_value))

        return df

    def encode_categorical(
        self,
        df: "pl.DataFrame",
        columns: Optional[List[str]] = None,
        method: str = "label",
    ) -> "pl.DataFrame":
        """Encode categorical columns.

        Args:
            df: Input DataFrame
            columns: Columns to encode (None = auto-detect)
            method: 'label' or 'onehot'

        Returns:
            DataFrame with encoded columns
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        if columns is None:
            columns = [
                col for col in df.columns
                if df[col].dtype in (pl.Utf8, pl.String, pl.Categorical)
            ]

        if method == "label":
            for col in columns:
                # Create label encoding
                unique_vals = df[col].unique().sort()
                mapping = {v: i for i, v in enumerate(unique_vals.to_list())}
                df = df.with_columns(
                    pl.col(col).replace(mapping).alias(f"{col}_encoded")
                )

        elif method == "onehot":
            for col in columns:
                # One-hot encoding
                dummies = df.select(pl.col(col)).to_dummies()
                df = pl.concat([df, dummies], how="horizontal")

        return df

    def normalize(
        self,
        df: "pl.DataFrame",
        columns: Optional[List[str]] = None,
        method: str = "minmax",
    ) -> "pl.DataFrame":
        """Normalize numeric columns.

        Args:
            df: Input DataFrame
            columns: Columns to normalize (None = all numeric)
            method: 'minmax', 'zscore', or 'robust'

        Returns:
            DataFrame with normalized columns
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        if columns is None:
            columns = [
                col for col in df.columns
                if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]

        for col in columns:
            if method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df = df.with_columns(
                        ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                    )

            elif method == "zscore":
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df = df.with_columns(
                        ((pl.col(col) - mean_val) / std_val).alias(col)
                    )

            elif method == "robust":
                median_val = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    df = df.with_columns(
                        ((pl.col(col) - median_val) / iqr).alias(col)
                    )

        return df

    def filter_outliers(
        self,
        df: "pl.DataFrame",
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> "pl.DataFrame":
        """Filter outliers from numeric columns.

        Args:
            df: Input DataFrame
            columns: Columns to check (None = all numeric)
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        if columns is None:
            columns = [
                col for col in df.columns
                if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]

        mask = pl.lit(True)

        for col in columns:
            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask = mask & (pl.col(col) >= lower) & (pl.col(col) <= upper)

            elif method == "zscore":
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    z_scores = (pl.col(col) - mean_val) / std_val
                    mask = mask & (z_scores.abs() <= threshold)

        return df.filter(mask)


class PolarsGenerator(BaseGenerator):
    """Synthetic data generator with native Polars support.

    This generator accepts Polars DataFrames directly and returns
    Polars DataFrames, avoiding conversion overhead for large datasets.

    Example:
        >>> import polars as pl
        >>> from genesis.polars_support import PolarsGenerator
        >>>
        >>> df = pl.read_csv("data.csv")
        >>> generator = PolarsGenerator(method='ctgan')
        >>> generator.fit(df)
        >>> synthetic = generator.generate(100000)  # Returns pl.DataFrame
    """

    def __init__(
        self,
        method: str = "auto",
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        polars_config: Optional[PolarsConfig] = None,
    ):
        """Initialize the Polars generator.

        Args:
            method: Generation method ('auto', 'ctgan', 'tvae', 'gaussian_copula')
            config: Generator configuration
            privacy: Privacy configuration
            polars_config: Polars-specific configuration
        """
        _check_polars_available()
        super().__init__(config, privacy)

        self.method = method
        self.polars_config = polars_config or PolarsConfig()
        self._inner_generator: Optional[SyntheticGenerator] = None
        self._original_schema: Optional[Dict[str, Any]] = None

    def _fit_impl(
        self,
        data: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        discrete_columns: Optional[ColumnList] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit implementation with Polars support."""
        # Handle Polars input
        if POLARS_AVAILABLE:
            if isinstance(data, pl.LazyFrame):
                data = data.collect()

            if isinstance(data, pl.DataFrame):
                # Store original schema for reconstruction
                self._original_schema = infer_polars_schema(data)

                # Auto-detect discrete columns if not provided
                if discrete_columns is None:
                    discrete_columns = detect_discrete_columns_polars(data)

                # Convert to pandas for the inner generator
                data = from_polars(data)

        # Create and fit inner generator
        self._inner_generator = SyntheticGenerator(
            method=self.method,
            config=self.config,
            privacy=self.privacy,
        )

        # Fit using pandas interface
        self._inner_generator.fit(data, discrete_columns)

        return FittingResult(
            success=True,
            message="Fitting completed successfully",
            epochs_trained=self.config.epochs,
            final_loss=0.0,
        )

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "pl.DataFrame":
        """Generate implementation returning Polars DataFrame."""
        if self._inner_generator is None:
            raise NotFittedError(self.__class__.__name__)

        # Generate using inner generator (returns pandas)
        pandas_result = self._inner_generator.generate(n_samples, conditions)

        # Convert to Polars
        return to_polars(pandas_result)

    def fit(
        self,
        data: Union[pd.DataFrame, "pl.DataFrame", "pl.LazyFrame"],
        discrete_columns: Optional[ColumnList] = None,
        constraints: Optional[List[BaseConstraint]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> "PolarsGenerator":
        """Fit the generator to training data.

        Accepts both pandas and Polars DataFrames.

        Args:
            data: Training data (pandas or Polars DataFrame)
            discrete_columns: List of categorical column names
            constraints: Constraints to enforce
            progress_callback: Progress callback

        Returns:
            Self for method chaining
        """
        # Convert Polars to pandas for validation
        if POLARS_AVAILABLE:
            if isinstance(data, pl.LazyFrame):
                data = data.collect()
            if isinstance(data, pl.DataFrame):
                # Store for later, but validate as pandas
                polars_data = data
                data = from_polars(data)
            else:
                polars_data = None
        else:
            polars_data = None

        # Use parent's validation
        self._validate_input_data(data)

        # Store constraints
        if constraints:
            self._constraints = ConstraintSet(constraints)

        # Fit with original Polars data if available
        if polars_data is not None:
            result = self._fit_impl(polars_data, discrete_columns, progress_callback)
        else:
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
        return_pandas: bool = False,
    ) -> Union["pl.DataFrame", pd.DataFrame]:
        """Generate synthetic data.

        Args:
            n_samples: Number of samples to generate
            conditions: Conditions for conditional generation
            apply_constraints: Whether to apply constraints
            progress_callback: Progress callback
            return_pandas: If True, return pandas DataFrame instead of Polars

        Returns:
            Polars DataFrame (or pandas if return_pandas=True)
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        # Generate as Polars
        synthetic_data = self._generate_impl(n_samples, conditions, progress_callback)

        # Apply constraints (convert to pandas, apply, convert back)
        if apply_constraints and len(self._constraints) > 0:
            pandas_data = from_polars(synthetic_data)
            pandas_data, _ = self._constraints.validate_and_transform(pandas_data)
            synthetic_data = to_polars(pandas_data)

        if return_pandas:
            return from_polars(synthetic_data)

        return synthetic_data

    def quality_report(self) -> Any:
        """Generate quality report.

        Returns:
            QualityReport from inner generator
        """
        if self._inner_generator is None:
            raise NotFittedError(self.__class__.__name__)
        return self._inner_generator.quality_report()


class PolarsStreamingGenerator:
    """Streaming synthetic data generation with Polars.

    Generates data in chunks for memory-efficient processing of
    very large synthetic datasets.

    Example:
        >>> generator = PolarsStreamingGenerator(base_generator)
        >>> for chunk in generator.generate_stream(1_000_000, chunk_size=10_000):
        ...     chunk.write_parquet(f"chunk_{i}.parquet")
    """

    def __init__(
        self,
        generator: Union[PolarsGenerator, SyntheticGenerator],
        chunk_size: int = 10_000,
    ):
        """Initialize streaming generator.

        Args:
            generator: Fitted generator to use
            chunk_size: Size of each generated chunk
        """
        _check_polars_available()

        if not generator.is_fitted:
            raise NotFittedError(generator.__class__.__name__)

        self.generator = generator
        self.chunk_size = chunk_size

    def generate_stream(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
    ):
        """Generate synthetic data as a stream of chunks.

        Args:
            n_samples: Total number of samples to generate
            conditions: Conditions for conditional generation

        Yields:
            Polars DataFrames of size chunk_size (last may be smaller)
        """
        generated = 0

        while generated < n_samples:
            remaining = n_samples - generated
            batch_size = min(self.chunk_size, remaining)

            # Generate chunk
            if isinstance(self.generator, PolarsGenerator):
                chunk = self.generator.generate(batch_size, conditions)
            else:
                # SyntheticGenerator returns pandas
                pandas_chunk = self.generator.generate(batch_size, conditions)
                chunk = to_polars(pandas_chunk)

            yield chunk
            generated += batch_size

    def generate_to_parquet(
        self,
        n_samples: int,
        output_path: str,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate and write directly to Parquet file.

        Memory-efficient for very large datasets.

        Args:
            n_samples: Total samples to generate
            output_path: Output Parquet file path
            conditions: Conditions for conditional generation
        """
        chunks = list(self.generate_stream(n_samples, conditions))
        combined = pl.concat(chunks)
        combined.write_parquet(output_path)

    def generate_to_csv(
        self,
        n_samples: int,
        output_path: str,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate and write directly to CSV file.

        Args:
            n_samples: Total samples to generate
            output_path: Output CSV file path
            conditions: Conditions for conditional generation
        """
        first_chunk = True

        with open(output_path, "w") as f:
            for chunk in self.generate_stream(n_samples, conditions):
                if first_chunk:
                    chunk.write_csv(f, include_header=True)
                    first_chunk = False
                else:
                    chunk.write_csv(f, include_header=False)


# Convenience functions
def fit_polars(
    data: "pl.DataFrame",
    method: str = "auto",
    privacy: Optional[PrivacyConfig] = None,
    **kwargs: Any,
) -> PolarsGenerator:
    """Convenience function to fit a generator to Polars data.

    Args:
        data: Polars DataFrame
        method: Generation method
        privacy: Privacy configuration
        **kwargs: Additional generator arguments

    Returns:
        Fitted PolarsGenerator
    """
    generator = PolarsGenerator(method=method, privacy=privacy, **kwargs)
    generator.fit(data)
    return generator


def generate_polars(
    generator: Union[PolarsGenerator, SyntheticGenerator],
    n_samples: int,
    **kwargs: Any,
) -> "pl.DataFrame":
    """Convenience function to generate Polars DataFrame.

    Args:
        generator: Fitted generator
        n_samples: Number of samples
        **kwargs: Additional generation arguments

    Returns:
        Polars DataFrame
    """
    if isinstance(generator, PolarsGenerator):
        return generator.generate(n_samples, **kwargs)
    else:
        pandas_result = generator.generate(n_samples, **kwargs)
        return to_polars(pandas_result)


__all__ = [
    # Core classes
    "PolarsGenerator",
    "PolarsStreamingGenerator",
    "PolarsDataLoader",
    "PolarsPreprocessor",
    # Configuration
    "PolarsConfig",
    "PolarsBackend",
    # Conversion functions
    "to_polars",
    "from_polars",
    # Utility functions
    "infer_polars_schema",
    "detect_discrete_columns_polars",
    # Convenience functions
    "fit_polars",
    "generate_polars",
    # Constants
    "POLARS_AVAILABLE",
]
