"""Schema inference for data analysis."""

from typing import Dict, List, Optional, Set

import pandas as pd

from genesis.core.types import ColumnMetadata, ColumnType, DataSchema


class SchemaAnalyzer:
    """Analyzer for inferring data schema from a DataFrame.

    This class automatically detects column types, cardinality, nullable status,
    and other metadata useful for synthetic data generation.
    """

    def __init__(
        self,
        categorical_threshold: int = 50,
        text_length_threshold: int = 100,
        unique_ratio_threshold: float = 0.95,
        sample_size: Optional[int] = None,
    ) -> None:
        """Initialize the schema analyzer.

        Args:
            categorical_threshold: Max unique values to consider a column categorical
            text_length_threshold: Min average length to consider text (not categorical)
            unique_ratio_threshold: Ratio above which column is considered unique/identifier
            sample_size: Number of rows to sample for analysis (None for all)
        """
        self.categorical_threshold = categorical_threshold
        self.text_length_threshold = text_length_threshold
        self.unique_ratio_threshold = unique_ratio_threshold
        self.sample_size = sample_size

    def analyze(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
    ) -> DataSchema:
        """Analyze a DataFrame and infer its schema.

        Args:
            data: DataFrame to analyze
            discrete_columns: Optional list of columns to treat as discrete/categorical

        Returns:
            DataSchema containing metadata for all columns
        """
        discrete_columns = set(discrete_columns or [])

        # Sample if needed
        if self.sample_size and len(data) > self.sample_size:
            sample = data.sample(n=self.sample_size, random_state=42)
        else:
            sample = data

        schema = DataSchema(
            n_rows=len(data),
            n_columns=len(data.columns),
        )

        for col in data.columns:
            col_data = sample[col]
            full_col_data = data[col]

            # Detect column type
            col_type = self._detect_column_type(col_data, col, discrete_columns)

            # Compute metadata based on type
            metadata = self._compute_column_metadata(col, full_col_data, col_type, discrete_columns)
            schema.columns[col] = metadata

        # Detect primary key
        schema.primary_key = self._detect_primary_key(data, schema)

        return schema

    def _detect_column_type(
        self,
        col_data: pd.Series,
        col_name: str,
        discrete_columns: Set[str],
    ) -> ColumnType:
        """Detect the type of a column.

        Args:
            col_data: Column data
            col_name: Column name
            discrete_columns: Set of columns explicitly marked as discrete

        Returns:
            Detected ColumnType
        """
        # Check if explicitly marked as discrete
        if col_name in discrete_columns:
            return ColumnType.CATEGORICAL

        dtype = col_data.dtype
        non_null = col_data.dropna()

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME

        # Check boolean
        if pd.api.types.is_bool_dtype(dtype) or col_data.dropna().isin([True, False, 0, 1]).all():
            unique_vals = col_data.dropna().unique()
            if len(unique_vals) <= 2:
                return ColumnType.BOOLEAN

        # Check numeric
        if pd.api.types.is_numeric_dtype(dtype):
            n_unique = col_data.nunique()
            n_total = len(col_data.dropna())

            # Check if likely identifier
            if n_total > 0 and n_unique / n_total >= self.unique_ratio_threshold:
                return ColumnType.IDENTIFIER

            # Check if discrete (integer with few unique values)
            if pd.api.types.is_integer_dtype(dtype):
                if n_unique <= self.categorical_threshold:
                    return ColumnType.NUMERIC_DISCRETE

            return ColumnType.NUMERIC_CONTINUOUS

        # Check object/string type
        if pd.api.types.is_object_dtype(dtype) or str(dtype).startswith("str"):
            n_unique = col_data.nunique()
            n_total = len(col_data.dropna())

            # Check if identifier
            if n_total > 0 and n_unique / n_total >= self.unique_ratio_threshold:
                return ColumnType.IDENTIFIER

            # Check if text (long strings) vs categorical
            if len(non_null) > 0:
                avg_length = non_null.astype(str).str.len().mean()
                if avg_length > self.text_length_threshold:
                    return ColumnType.TEXT

            # Check cardinality for categorical
            if n_unique <= self.categorical_threshold:
                return ColumnType.CATEGORICAL

            return ColumnType.TEXT

        # Check category dtype
        if str(dtype) == "category":
            return ColumnType.CATEGORICAL

        return ColumnType.UNKNOWN

    def _compute_column_metadata(
        self,
        col_name: str,
        col_data: pd.Series,
        col_type: ColumnType,
        discrete_columns: Set[str],
    ) -> ColumnMetadata:
        """Compute detailed metadata for a column.

        Args:
            col_name: Column name
            col_data: Full column data
            col_type: Detected column type
            discrete_columns: Set of discrete columns

        Returns:
            ColumnMetadata with computed statistics
        """
        non_null = col_data.dropna()
        n_total = len(col_data)
        n_missing = col_data.isna().sum()

        metadata = ColumnMetadata(
            name=col_name,
            dtype=col_type,
            nullable=n_missing > 0,
            unique=col_data.nunique() == n_total,
            cardinality=col_data.nunique(),
            missing_rate=n_missing / n_total if n_total > 0 else 0.0,
        )

        # Add type-specific metadata
        if col_type in (ColumnType.NUMERIC_CONTINUOUS, ColumnType.NUMERIC_DISCRETE):
            if len(non_null) > 0:
                metadata.min_value = float(non_null.min())
                metadata.max_value = float(non_null.max())
                metadata.mean = float(non_null.mean())
                metadata.std = float(non_null.std())

        elif col_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            metadata.categories = col_data.dropna().unique().tolist()

        return metadata

    def _detect_primary_key(
        self,
        data: pd.DataFrame,
        schema: DataSchema,
    ) -> Optional[str]:
        """Detect the most likely primary key column.

        Args:
            data: DataFrame
            schema: Schema with column metadata

        Returns:
            Name of the likely primary key column, or None
        """
        candidates = []

        for col_name, meta in schema.columns.items():
            if meta.dtype == ColumnType.IDENTIFIER:
                candidates.append(col_name)
            elif meta.unique and meta.cardinality == len(data):
                candidates.append(col_name)

        if not candidates:
            return None

        # Prefer columns with "id" in the name
        for candidate in candidates:
            if "id" in candidate.lower():
                return candidate

        return candidates[0]

    def get_column_types_summary(self, schema: DataSchema) -> Dict[str, List[str]]:
        """Get a summary of columns grouped by type.

        Args:
            schema: Data schema

        Returns:
            Dictionary mapping type names to lists of column names
        """
        summary: Dict[str, List[str]] = {}

        for col_name, meta in schema.columns.items():
            type_name = meta.dtype.name
            if type_name not in summary:
                summary[type_name] = []
            summary[type_name].append(col_name)

        return summary


def infer_schema(
    data: pd.DataFrame,
    discrete_columns: Optional[List[str]] = None,
) -> DataSchema:
    """Convenience function to infer schema from a DataFrame.

    Args:
        data: DataFrame to analyze
        discrete_columns: Optional list of discrete columns

    Returns:
        Inferred DataSchema
    """
    analyzer = SchemaAnalyzer()
    return analyzer.analyze(data, discrete_columns)
