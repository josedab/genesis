"""Input/Output utilities for Genesis."""

from genesis.io.csv_io import detect_csv_schema, read_csv, write_csv
from genesis.io.database import get_table_schema, list_tables, read_sql, write_sql
from genesis.io.pandas_io import (
    concat_dataframes,
    load_dataframe,
    sample_dataframe,
    save_dataframe,
    validate_schema_match,
)

__all__ = [
    # Pandas
    "load_dataframe",
    "save_dataframe",
    "sample_dataframe",
    "concat_dataframes",
    "validate_schema_match",
    # CSV
    "read_csv",
    "write_csv",
    "detect_csv_schema",
    # Database
    "read_sql",
    "write_sql",
    "list_tables",
    "get_table_schema",
]
