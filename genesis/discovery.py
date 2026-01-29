"""Automated schema discovery from databases and data sources.

This module provides automatic schema detection and synthetic database
generation from various sources including databases, CSV files, and APIs.

Example:
    >>> from genesis.discovery import SchemaDiscovery
    >>>
    >>> # From database
    >>> discovery = SchemaDiscovery.from_database("postgresql://user:pass@host/db")
    >>> schema = discovery.discover()
    >>>
    >>> # Generate synthetic replica
    >>> generator = discovery.create_generator()
    >>> synthetic_tables = generator.generate_all(n_samples=10000)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnSchema:
    """Schema information for a single column."""

    name: str
    dtype: str
    nullable: bool = True
    unique: bool = False
    primary_key: bool = False
    foreign_key: Optional[str] = None  # "table.column" format

    # Statistics
    n_unique: Optional[int] = None
    null_ratio: Optional[float] = None
    sample_values: Optional[List[Any]] = None

    # For numeric columns
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None

    # For string columns
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Detected regex pattern

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "unique": self.unique,
            "primary_key": self.primary_key,
            "foreign_key": self.foreign_key,
            "n_unique": self.n_unique,
            "null_ratio": self.null_ratio,
            "sample_values": self.sample_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "max_length": self.max_length,
            "pattern": self.pattern,
        }


@dataclass
class TableSchema:
    """Schema information for a table."""

    name: str
    columns: List[ColumnSchema] = field(default_factory=list)
    n_rows: Optional[int] = None
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, str] = field(
        default_factory=dict
    )  # local_col -> remote_table.remote_col

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "n_rows": self.n_rows,
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
        }

    @property
    def column_names(self) -> List[str]:
        """Get list of column names."""
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None


@dataclass
class DatabaseSchema:
    """Complete database schema."""

    name: str
    tables: List[TableSchema] = field(default_factory=list)
    source_type: str = "unknown"  # postgresql, mysql, sqlite, csv, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "tables": [t.to_dict() for t in self.tables],
            "source_type": self.source_type,
        }

    def get_table(self, name: str) -> Optional[TableSchema]:
        """Get table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def get_generation_order(self) -> List[str]:
        """Get tables in order for generation (parents before children)."""
        # Build dependency graph
        deps: Dict[str, List[str]] = {t.name: [] for t in self.tables}
        for table in self.tables:
            for fk_target in table.foreign_keys.values():
                parent_table = fk_target.split(".")[0]
                if parent_table in deps:
                    deps[table.name].append(parent_table)

        # Topological sort
        result = []
        visited = set()
        temp_visited = set()

        def visit(name: str) -> None:
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            temp_visited.add(name)
            for dep in deps.get(name, []):
                visit(dep)
            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        for table_name in deps:
            if table_name not in visited:
                visit(table_name)

        return result


class SchemaDiscovery:
    """Discover and analyze database schemas for synthetic data generation."""

    def __init__(
        self,
        connection: Optional[Any] = None,
        engine: Optional[Any] = None,
    ) -> None:
        """Initialize schema discovery.

        Args:
            connection: Database connection or connection string
            engine: SQLAlchemy engine
        """
        self._connection = connection
        self._engine = engine
        self._schema: Optional[DatabaseSchema] = None
        self._sample_data: Dict[str, pd.DataFrame] = {}

    @classmethod
    def from_database(
        cls,
        connection_string: str,
    ) -> "SchemaDiscovery":
        """Create discovery from database connection string.

        Args:
            connection_string: SQLAlchemy connection string

        Returns:
            SchemaDiscovery instance
        """
        try:
            from sqlalchemy import create_engine

            engine = create_engine(connection_string)
            return cls(engine=engine)
        except ImportError as e:
            raise ImportError(
                "SQLAlchemy is required for database discovery. "
                "Install with: pip install sqlalchemy"
            ) from e

    @classmethod
    def from_csv_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "*.csv",
    ) -> "SchemaDiscovery":
        """Create discovery from directory of CSV files.

        Args:
            directory: Path to directory containing CSV files
            pattern: Glob pattern for CSV files

        Returns:
            SchemaDiscovery instance
        """
        instance = cls()
        directory = Path(directory)

        tables = []
        for csv_path in directory.glob(pattern):
            df = pd.read_csv(csv_path)
            table_name = csv_path.stem
            instance._sample_data[table_name] = df
            tables.append(instance._analyze_dataframe(df, table_name))

        instance._schema = DatabaseSchema(
            name=directory.name,
            tables=tables,
            source_type="csv",
        )

        return instance

    @classmethod
    def from_dataframes(
        cls,
        dataframes: Dict[str, pd.DataFrame],
        name: str = "dataset",
    ) -> "SchemaDiscovery":
        """Create discovery from dictionary of DataFrames.

        Args:
            dataframes: Dict mapping table names to DataFrames
            name: Name for the dataset

        Returns:
            SchemaDiscovery instance
        """
        instance = cls()
        instance._sample_data = dataframes.copy()

        tables = []
        for table_name, df in dataframes.items():
            tables.append(instance._analyze_dataframe(df, table_name))

        instance._schema = DatabaseSchema(
            name=name,
            tables=tables,
            source_type="dataframe",
        )

        return instance

    def discover(
        self,
        tables: Optional[List[str]] = None,
        sample_size: int = 10000,
        infer_relationships: bool = True,
    ) -> DatabaseSchema:
        """Discover database schema.

        Args:
            tables: Specific tables to discover (None = all)
            sample_size: Number of rows to sample for analysis
            infer_relationships: Whether to infer FK relationships

        Returns:
            DatabaseSchema with discovered information
        """
        if self._schema is not None:
            return self._schema

        if self._engine is None:
            raise ValueError("No database connection available")

        from sqlalchemy import MetaData, inspect

        inspector = inspect(self._engine)
        metadata = MetaData()
        metadata.reflect(bind=self._engine)

        # Get table names
        table_names = tables or inspector.get_table_names()

        discovered_tables = []
        for table_name in table_names:
            logger.info(f"Analyzing table: {table_name}")

            # Get column info
            columns = []
            pk_columns = [
                pk["name"]
                for pk in inspector.get_pk_constraint(table_name).get("constrained_columns", [])
            ]
            fk_info = {}
            for fk in inspector.get_foreign_keys(table_name):
                for i, col in enumerate(fk.get("constrained_columns", [])):
                    ref_table = fk.get("referred_table", "")
                    ref_cols = fk.get("referred_columns", [])
                    if ref_cols:
                        fk_info[col] = f"{ref_table}.{ref_cols[i]}"

            for col_info in inspector.get_columns(table_name):
                col_name = col_info["name"]
                col_schema = ColumnSchema(
                    name=col_name,
                    dtype=str(col_info["type"]),
                    nullable=col_info.get("nullable", True),
                    primary_key=col_name in pk_columns,
                    foreign_key=fk_info.get(col_name),
                )
                columns.append(col_schema)

            # Sample data for statistics
            query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
            try:
                sample_df = pd.read_sql(query, self._engine)
                self._sample_data[table_name] = sample_df

                # Enrich columns with statistics
                for col_schema in columns:
                    if col_schema.name in sample_df.columns:
                        self._enrich_column_stats(col_schema, sample_df[col_schema.name])

                n_rows = len(sample_df)
            except Exception as e:
                logger.warning(f"Could not sample {table_name}: {e}")
                n_rows = None

            table_schema = TableSchema(
                name=table_name,
                columns=columns,
                n_rows=n_rows,
                primary_keys=pk_columns,
                foreign_keys=fk_info,
            )
            discovered_tables.append(table_schema)

        # Infer additional relationships if requested
        if infer_relationships:
            self._infer_relationships(discovered_tables)

        # Determine source type
        source_type = str(self._engine.url.get_backend_name()) if self._engine else "unknown"

        self._schema = DatabaseSchema(
            name=str(self._engine.url.database) if self._engine else "database",
            tables=discovered_tables,
            source_type=source_type,
        )

        return self._schema

    def _analyze_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> TableSchema:
        """Analyze a DataFrame to create table schema."""
        columns = []

        for col_name in df.columns:
            series = df[col_name]

            # Determine dtype
            if pd.api.types.is_numeric_dtype(series):
                dtype = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                dtype = "datetime"
            elif pd.api.types.is_bool_dtype(series):
                dtype = "boolean"
            else:
                dtype = "string"

            col_schema = ColumnSchema(
                name=col_name,
                dtype=dtype,
                nullable=series.isna().any(),
                unique=series.nunique() == len(series.dropna()),
            )

            self._enrich_column_stats(col_schema, series)
            columns.append(col_schema)

        # Detect primary key candidates
        pk_candidates = [c.name for c in columns if c.unique and not c.nullable]

        return TableSchema(
            name=table_name,
            columns=columns,
            n_rows=len(df),
            primary_keys=pk_candidates[:1],  # Take first candidate
        )

    def _enrich_column_stats(
        self,
        col_schema: ColumnSchema,
        series: pd.Series,
    ) -> None:
        """Enrich column schema with statistics from data."""
        col_schema.n_unique = int(series.nunique())
        col_schema.null_ratio = float(series.isna().mean())

        # Sample values
        non_null = series.dropna()
        if len(non_null) > 0:
            col_schema.sample_values = non_null.head(5).tolist()

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            col_schema.min_value = float(non_null.min()) if len(non_null) > 0 else None
            col_schema.max_value = float(non_null.max()) if len(non_null) > 0 else None
            col_schema.mean_value = float(non_null.mean()) if len(non_null) > 0 else None

        # String stats
        if series.dtype == object:
            str_lens = non_null.astype(str).str.len()
            if len(str_lens) > 0:
                col_schema.max_length = int(str_lens.max())

    def _infer_relationships(
        self,
        tables: List[TableSchema],
    ) -> None:
        """Infer foreign key relationships from column names and data."""
        for table in tables:
            for col in table.columns:
                if col.foreign_key:
                    continue  # Already has FK

                # Check naming patterns: table_id, tableId, etc.
                for other_table in tables:
                    if other_table.name == table.name:
                        continue

                    # Check if column name matches pattern
                    patterns = [
                        f"{other_table.name}_id",
                        f"{other_table.name}Id",
                        f"{other_table.name.lower()}_id",
                    ]

                    if col.name.lower() in [p.lower() for p in patterns]:
                        # Check if values match
                        if (
                            other_table.name in self._sample_data
                            and table.name in self._sample_data
                        ):

                            source_vals = set(self._sample_data[table.name][col.name].dropna())
                            pk_col = (
                                other_table.primary_keys[0] if other_table.primary_keys else None
                            )

                            if pk_col:
                                target_vals = set(
                                    self._sample_data[other_table.name][pk_col].dropna()
                                )

                                # If most source values are in target, likely a FK
                                if len(source_vals) > 0:
                                    overlap = len(source_vals & target_vals) / len(source_vals)
                                    if overlap > 0.8:
                                        col.foreign_key = f"{other_table.name}.{pk_col}"
                                        table.foreign_keys[col.name] = col.foreign_key
                                        logger.info(
                                            f"Inferred FK: {table.name}.{col.name} -> {col.foreign_key}"
                                        )

    def create_generator(
        self,
        method: str = "auto",
    ) -> "MultiTableSynthesizer":
        """Create a multi-table generator from discovered schema.

        Args:
            method: Generator method to use

        Returns:
            Configured MultiTableSynthesizer
        """
        if self._schema is None:
            raise ValueError("Must call discover() first")

        from genesis.multitable import MultiTableGenerator, RelationalSchema

        # Build relational schema
        rel_schema = RelationalSchema(name=self._schema.name)

        for table in self._schema.tables:
            rel_schema.add_table(
                name=table.name,
                primary_key=table.primary_keys[0] if table.primary_keys else None,
            )

        for table in self._schema.tables:
            for local_col, target in table.foreign_keys.items():
                parent_table, parent_col = target.split(".")
                rel_schema.add_relationship(
                    parent_table=parent_table,
                    child_table=table.name,
                    parent_key=parent_col,
                    child_key=local_col,
                )

        return MultiTableGenerator(schema=rel_schema, method=method)

    def get_sample_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get sampled data for a table."""
        return self._sample_data.get(table_name)

    @property
    def schema(self) -> Optional[DatabaseSchema]:
        """Get discovered schema."""
        return self._schema


class MultiTableSynthesizer:
    """Synthesize multiple related tables while preserving relationships."""

    def __init__(
        self,
        schema: DatabaseSchema,
        method: str = "auto",
    ) -> None:
        """Initialize synthesizer.

        Args:
            schema: Database schema to synthesize
            method: Generator method
        """
        self.schema = schema
        self.method = method
        self._generators: Dict[str, Any] = {}
        self._fitted = False

    def fit(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> "MultiTableSynthesizer":
        """Fit generators for all tables.

        Args:
            data: Dict mapping table names to DataFrames

        Returns:
            Self for chaining
        """
        from genesis import SyntheticGenerator

        for table_name in self.schema.get_generation_order():
            if table_name not in data:
                logger.warning(f"No data for table {table_name}")
                continue

            table_schema = self.schema.get_table(table_name)
            df = data[table_name]

            # Identify discrete columns
            discrete_cols = []
            if table_schema:
                for col in table_schema.columns:
                    if col.dtype in ("string", "categorical") or col.n_unique and col.n_unique < 50:
                        discrete_cols.append(col.name)

            generator = SyntheticGenerator(method=self.method)
            generator.fit(df, discrete_columns=discrete_cols)
            self._generators[table_name] = generator

        self._fitted = True
        return self

    def generate(
        self,
        n_samples: Union[int, Dict[str, int]] = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for all tables.

        Args:
            n_samples: Number of samples per table (int or dict)

        Returns:
            Dict mapping table names to synthetic DataFrames
        """
        if not self._fitted:
            raise ValueError("Must fit before generating")

        result = {}

        # Handle uniform vs per-table sample sizes
        if isinstance(n_samples, int):
            n_samples = dict.fromkeys(self._generators, n_samples)

        for table_name in self.schema.get_generation_order():
            if table_name not in self._generators:
                continue

            n = n_samples.get(table_name, 1000)
            generator = self._generators[table_name]

            synthetic = generator.generate(n)
            result[table_name] = synthetic

            logger.info(f"Generated {len(synthetic)} rows for {table_name}")

        return result
