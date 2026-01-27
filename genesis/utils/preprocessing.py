"""Data preprocessing utilities for Genesis."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def handle_missing_values(
    data: pd.DataFrame,
    strategy: str = "auto",
    fill_value: Optional[Any] = None,
    numeric_strategy: str = "mean",
    categorical_strategy: str = "mode",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Handle missing values in the data.

    Args:
        data: Input DataFrame
        strategy: Strategy for handling missing values ('auto', 'drop', 'fill')
        fill_value: Value to use for filling (when strategy='fill')
        numeric_strategy: Strategy for numeric columns ('mean', 'median', 'mode', 'zero')
        categorical_strategy: Strategy for categorical columns ('mode', 'unknown')

    Returns:
        Tuple of (processed DataFrame, dict of fill values used)
    """
    result = data.copy()
    fill_values: Dict[str, Any] = {}

    for col in result.columns:
        if result[col].isna().sum() == 0:
            continue

        if strategy == "drop":
            result = result.dropna(subset=[col])
            continue

        if pd.api.types.is_numeric_dtype(result[col]):
            if numeric_strategy == "mean":
                fill_val = result[col].mean()
            elif numeric_strategy == "median":
                fill_val = result[col].median()
            elif numeric_strategy == "mode":
                fill_val = result[col].mode().iloc[0] if len(result[col].mode()) > 0 else 0
            elif numeric_strategy == "zero":
                fill_val = 0
            else:
                fill_val = fill_value if fill_value is not None else result[col].mean()
        else:
            if categorical_strategy == "mode":
                mode_vals = result[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
            elif categorical_strategy == "unknown":
                fill_val = "unknown"
            else:
                fill_val = fill_value if fill_value is not None else "unknown"

        result[col] = result[col].fillna(fill_val)
        fill_values[col] = fill_val

    return result, fill_values


def detect_outliers(
    data: pd.DataFrame,
    method: str = "iqr",
    columns: Optional[List[str]] = None,
    threshold: float = 1.5,
) -> Dict[str, np.ndarray]:
    """Detect outliers in numeric columns.

    Args:
        data: Input DataFrame
        method: Detection method ('iqr', 'zscore', 'mad')
        columns: Columns to check (None for all numeric)
        threshold: Threshold for outlier detection

    Returns:
        Dictionary mapping column names to boolean arrays (True = outlier)
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    outliers: Dict[str, np.ndarray] = {}

    for col in columns:
        if col not in data.columns:
            continue

        values = data[col].dropna()
        if len(values) == 0:
            outliers[col] = np.zeros(len(data), dtype=bool)
            continue

        if method == "iqr":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask = (data[col] < lower_bound) | (data[col] > upper_bound)

        elif method == "zscore":
            z_full = np.abs((data[col] - values.mean()) / values.std())
            mask = z_full > threshold

        elif method == "mad":
            median = values.median()
            mad = np.median(np.abs(values - median))
            if mad == 0:
                mask = np.zeros(len(data), dtype=bool)
            else:
                modified_z = 0.6745 * (data[col] - median) / mad
                mask = np.abs(modified_z) > threshold

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        outliers[col] = mask.values

    return outliers


def clip_outliers(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower_percentile: float = 1,
    upper_percentile: float = 99,
) -> pd.DataFrame:
    """Clip outliers to specified percentiles.

    Args:
        data: Input DataFrame
        columns: Columns to clip (None for all numeric)
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping

    Returns:
        DataFrame with clipped values
    """
    result = data.copy()

    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue
        lower = np.percentile(result[col].dropna(), lower_percentile)
        upper = np.percentile(result[col].dropna(), upper_percentile)
        result[col] = result[col].clip(lower, upper)

    return result


def normalize_column_names(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Normalize column names to be valid identifiers.

    Args:
        data: Input DataFrame

    Returns:
        Tuple of (DataFrame with normalized names, mapping of old to new names)
    """
    import re

    result = data.copy()
    name_mapping: Dict[str, str] = {}

    new_columns = []
    for col in result.columns:
        # Convert to string and clean
        new_col = str(col)
        new_col = re.sub(r"[^\w\s]", "_", new_col)
        new_col = re.sub(r"\s+", "_", new_col)
        new_col = re.sub(r"_+", "_", new_col)
        new_col = new_col.strip("_").lower()

        if not new_col or new_col[0].isdigit():
            new_col = f"col_{new_col}"

        # Handle duplicates
        base_col = new_col
        counter = 1
        while new_col in new_columns:
            new_col = f"{base_col}_{counter}"
            counter += 1

        new_columns.append(new_col)
        if col != new_col:
            name_mapping[col] = new_col

    result.columns = new_columns
    return result, name_mapping


def balance_classes(
    data: pd.DataFrame,
    target_column: str,
    strategy: str = "oversample",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Balance class distribution in the data.

    Args:
        data: Input DataFrame
        target_column: Column containing class labels
        strategy: Balancing strategy ('oversample', 'undersample', 'smote')
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    rng = np.random.RandomState(random_state)
    class_counts = data[target_column].value_counts()

    if strategy == "oversample":
        max_count = class_counts.max()
        balanced_dfs = []

        for class_val in class_counts.index:
            class_data = data[data[target_column] == class_val]
            n_samples = max_count - len(class_data)

            if n_samples > 0:
                oversampled = class_data.sample(n=n_samples, replace=True, random_state=rng)
                balanced_dfs.append(pd.concat([class_data, oversampled]))
            else:
                balanced_dfs.append(class_data)

        return pd.concat(balanced_dfs).reset_index(drop=True)

    elif strategy == "undersample":
        min_count = class_counts.min()
        balanced_dfs = []

        for class_val in class_counts.index:
            class_data = data[data[target_column] == class_val]
            undersampled = class_data.sample(n=min_count, random_state=rng)
            balanced_dfs.append(undersampled)

        return pd.concat(balanced_dfs).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    stratify_column: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.

    Args:
        data: Input DataFrame
        test_size: Proportion of data for test set
        stratify_column: Column for stratified splitting
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    from sklearn.model_selection import train_test_split

    stratify = data[stratify_column] if stratify_column else None
    train_data, test_data = train_test_split(
        data, test_size=test_size, stratify=stratify, random_state=random_state
    )

    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


def remove_constant_columns(data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns with constant values.

    Args:
        data: Input DataFrame

    Returns:
        Tuple of (DataFrame without constant columns, list of removed column names)
    """
    constant_cols = []

    for col in data.columns:
        if data[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)

    result = data.drop(columns=constant_cols)
    return result, constant_cols


def remove_duplicate_rows(
    data: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first"
) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows.

    Args:
        data: Input DataFrame
        subset: Columns to consider for duplicates
        keep: Which duplicates to keep ('first', 'last', False)

    Returns:
        Tuple of (DataFrame without duplicates, number of removed rows)
    """
    n_original = len(data)
    result = data.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    n_removed = n_original - len(result)
    return result, n_removed
