"""Pandas DataFrame I/O utilities."""

from typing import Any, Dict, List, Optional, Union

import pandas as pd


def load_dataframe(
    source: Union[str, pd.DataFrame, Dict[str, Any]],
    **kwargs,
) -> pd.DataFrame:
    """Load DataFrame from various sources.

    Args:
        source: File path, DataFrame, or dictionary
        **kwargs: Additional arguments for pandas read functions

    Returns:
        Loaded DataFrame
    """
    if isinstance(source, pd.DataFrame):
        return source

    if isinstance(source, dict):
        return pd.DataFrame(source)

    if isinstance(source, str):
        if source.endswith(".csv"):
            return pd.read_csv(source, **kwargs)
        elif source.endswith(".parquet"):
            return pd.read_parquet(source, **kwargs)
        elif source.endswith((".xls", ".xlsx")):
            return pd.read_excel(source, **kwargs)
        elif source.endswith(".json"):
            return pd.read_json(source, **kwargs)
        elif source.endswith((".pkl", ".pickle")):
            return pd.read_pickle(source, **kwargs)
        else:
            # Try to infer format
            try:
                return pd.read_csv(source, **kwargs)
            except Exception:
                return pd.read_parquet(source, **kwargs)

    raise ValueError(f"Cannot load DataFrame from {type(source)}")


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    **kwargs,
) -> None:
    """Save DataFrame to file.

    Args:
        df: DataFrame to save
        path: Output file path
        **kwargs: Additional arguments for pandas write functions
    """
    if path.endswith(".csv"):
        df.to_csv(path, index=False, **kwargs)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False, **kwargs)
    elif path.endswith((".xls", ".xlsx")):
        df.to_excel(path, index=False, **kwargs)
    elif path.endswith(".json"):
        df.to_json(path, **kwargs)
    elif path.endswith((".pkl", ".pickle")):
        df.to_pickle(path, **kwargs)
    else:
        # Default to CSV
        df.to_csv(path, index=False, **kwargs)


def sample_dataframe(
    df: pd.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Sample rows from DataFrame.

    Args:
        df: DataFrame to sample
        n: Number of rows to sample
        frac: Fraction of rows to sample
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    if n is not None:
        return df.sample(n=min(n, len(df)), random_state=random_state)
    elif frac is not None:
        return df.sample(frac=frac, random_state=random_state)
    else:
        return df


def concat_dataframes(
    dfs: List[pd.DataFrame],
    ignore_index: bool = True,
) -> pd.DataFrame:
    """Concatenate multiple DataFrames.

    Args:
        dfs: List of DataFrames
        ignore_index: Whether to reset index

    Returns:
        Concatenated DataFrame
    """
    return pd.concat(dfs, ignore_index=ignore_index)


def validate_schema_match(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> Dict[str, Any]:
    """Check if two DataFrames have matching schemas.

    Args:
        df1: First DataFrame
        df2: Second DataFrame

    Returns:
        Dictionary with match results
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    return {
        "columns_match": cols1 == cols2,
        "missing_in_df2": list(cols1 - cols2),
        "extra_in_df2": list(cols2 - cols1),
        "common_columns": list(cols1 & cols2),
        "dtype_matches": {col: df1[col].dtype == df2[col].dtype for col in cols1 & cols2},
    }
