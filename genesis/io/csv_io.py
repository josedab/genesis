"""CSV file I/O utilities."""

from typing import Any, Dict, List, Optional

import pandas as pd


def read_csv(
    path: str,
    encoding: str = "utf-8",
    sep: str = ",",
    header: int = 0,
    dtype: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[List[str]] = None,
    na_values: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read CSV file with common defaults.

    Args:
        path: Path to CSV file
        encoding: File encoding
        sep: Column separator
        header: Row number for header
        dtype: Column data types
        parse_dates: Columns to parse as dates
        na_values: Values to treat as NA
        **kwargs: Additional pandas arguments

    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(
        path,
        encoding=encoding,
        sep=sep,
        header=header,
        dtype=dtype,
        parse_dates=parse_dates,
        na_values=na_values,
        **kwargs,
    )


def write_csv(
    df: pd.DataFrame,
    path: str,
    encoding: str = "utf-8",
    sep: str = ",",
    index: bool = False,
    date_format: Optional[str] = None,
    **kwargs,
) -> None:
    """Write DataFrame to CSV file.

    Args:
        df: DataFrame to write
        path: Output path
        encoding: File encoding
        sep: Column separator
        index: Whether to write row index
        date_format: Format for datetime columns
        **kwargs: Additional pandas arguments
    """
    df.to_csv(
        path,
        encoding=encoding,
        sep=sep,
        index=index,
        date_format=date_format,
        **kwargs,
    )


def detect_csv_schema(
    path: str,
    n_rows: int = 1000,
) -> Dict[str, Any]:
    """Detect CSV schema by sampling rows.

    Args:
        path: Path to CSV file
        n_rows: Number of rows to sample

    Returns:
        Dictionary with detected schema
    """
    # Read sample
    df = pd.read_csv(path, nrows=n_rows)

    schema = {
        "columns": list(df.columns),
        "n_rows_sampled": len(df),
        "column_types": {},
    }

    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(),
            "n_missing": df[col].isna().sum(),
            "sample_values": df[col].dropna().head(3).tolist(),
        }
        schema["column_types"][col] = col_info

    return schema
