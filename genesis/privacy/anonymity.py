"""K-anonymity and L-diversity implementations."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def check_k_anonymity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int = 5,
) -> Dict[str, Any]:
    """Check if data satisfies k-anonymity.

    K-anonymity requires that each combination of quasi-identifiers
    appears at least k times.

    Args:
        data: DataFrame to check
        quasi_identifiers: List of quasi-identifier columns
        k: Minimum anonymity level

    Returns:
        Dictionary with k-anonymity analysis
    """
    # Filter to existing columns
    qi_cols = [c for c in quasi_identifiers if c in data.columns]

    if not qi_cols:
        return {"error": "No valid quasi-identifier columns"}

    # Group by quasi-identifiers
    group_sizes = data.groupby(qi_cols).size()

    # Check violations
    violations = group_sizes[group_sizes < k]
    n_violating_groups = len(violations)
    n_violating_records = violations.sum()

    # Actual k achieved
    actual_k = int(group_sizes.min()) if len(group_sizes) > 0 else 0

    return {
        "satisfies_k": actual_k >= k,
        "requested_k": k,
        "achieved_k": actual_k,
        "n_groups": len(group_sizes),
        "n_violating_groups": n_violating_groups,
        "n_violating_records": int(n_violating_records),
        "violation_rate": n_violating_records / len(data) if len(data) > 0 else 0,
        "group_size_distribution": {
            "min": int(group_sizes.min()) if len(group_sizes) > 0 else 0,
            "max": int(group_sizes.max()) if len(group_sizes) > 0 else 0,
            "mean": float(group_sizes.mean()) if len(group_sizes) > 0 else 0,
            "median": float(group_sizes.median()) if len(group_sizes) > 0 else 0,
        },
    }


def enforce_k_anonymity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int = 5,
    method: str = "suppress",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enforce k-anonymity on data.

    Args:
        data: DataFrame to process
        quasi_identifiers: List of quasi-identifier columns
        k: Minimum anonymity level
        method: Method to use ('suppress', 'generalize')

    Returns:
        Tuple of (processed DataFrame, statistics)
    """
    qi_cols = [c for c in quasi_identifiers if c in data.columns]

    if not qi_cols:
        return data, {"error": "No valid quasi-identifier columns"}

    if method == "suppress":
        return _suppress_for_k_anonymity(data, qi_cols, k)
    elif method == "generalize":
        return _generalize_for_k_anonymity(data, qi_cols, k)
    else:
        raise ValueError(f"Unknown method: {method}")


def _suppress_for_k_anonymity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Suppress records that violate k-anonymity."""
    # Group by quasi-identifiers
    group_sizes = data.groupby(quasi_identifiers).size()

    # Find violating groups
    violating_groups = set(group_sizes[group_sizes < k].index)

    if not violating_groups:
        return data.copy(), {"n_suppressed": 0, "method": "suppress"}

    # Create mask for records to keep
    if len(quasi_identifiers) == 1:
        # Single QI: index values are scalars, not tuples
        qi = quasi_identifiers[0]
        violating_mask = data[qi].isin(violating_groups)
    else:
        # Multiple QIs: index values are tuples
        def is_in_violating_group(row):
            key = tuple(row[qi] for qi in quasi_identifiers)
            return key in violating_groups

        # Mark violating records
        violating_mask = data.apply(is_in_violating_group, axis=1)

    # Keep non-violating records
    result = data[~violating_mask].copy()

    return result, {
        "n_suppressed": int(violating_mask.sum()),
        "suppression_rate": float(violating_mask.mean()),
        "method": "suppress",
    }


def _generalize_for_k_anonymity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generalize values to achieve k-anonymity."""
    result = data.copy()
    generalizations = {}

    for qi in quasi_identifiers:
        col = result[qi]

        if pd.api.types.is_numeric_dtype(col):
            # Numeric: bin into ranges
            try:
                n_bins = max(1, len(col.unique()) // k)
                result[qi] = pd.cut(col, bins=n_bins).astype(str)
                generalizations[qi] = "binning"
            except Exception:
                generalizations[qi] = "unchanged"
        else:
            # Categorical: group rare categories
            value_counts = col.value_counts()
            rare_values = value_counts[value_counts < k].index.tolist()

            if rare_values:
                rv = rare_values  # Capture for lambda
                result[qi] = result[qi].apply(lambda x, rv=rv: "OTHER" if x in rv else x)
                generalizations[qi] = f"grouped {len(rare_values)} rare values"
            else:
                generalizations[qi] = "unchanged"

    return result, {
        "generalizations": generalizations,
        "method": "generalize",
    }


def check_l_diversity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_column: str,
    l_value: int = 2,
) -> Dict[str, Any]:
    """Check if data satisfies l-diversity.

    L-diversity requires that each equivalence class (defined by QIs)
    has at least l "well-represented" values for the sensitive attribute.

    Args:
        data: DataFrame to check
        quasi_identifiers: List of quasi-identifier columns
        sensitive_column: Sensitive attribute column
        l_value: Minimum diversity level

    Returns:
        Dictionary with l-diversity analysis
    """
    qi_cols = [c for c in quasi_identifiers if c in data.columns]

    if not qi_cols:
        return {"error": "No valid quasi-identifier columns"}

    if sensitive_column not in data.columns:
        return {"error": f"Sensitive column '{sensitive_column}' not found"}

    # Group by quasi-identifiers
    diversity_per_group = data.groupby(qi_cols)[sensitive_column].nunique()

    # Check violations
    violations = diversity_per_group[diversity_per_group < l_value]
    n_violating_groups = len(violations)

    # Actual l achieved
    actual_l = int(diversity_per_group.min()) if len(diversity_per_group) > 0 else 0

    return {
        "satisfies_l": actual_l >= l_value,
        "requested_l": l_value,
        "achieved_l": actual_l,
        "n_groups": len(diversity_per_group),
        "n_violating_groups": n_violating_groups,
        "diversity_distribution": {
            "min": int(diversity_per_group.min()) if len(diversity_per_group) > 0 else 0,
            "max": int(diversity_per_group.max()) if len(diversity_per_group) > 0 else 0,
            "mean": float(diversity_per_group.mean()) if len(diversity_per_group) > 0 else 0,
        },
    }


def enforce_l_diversity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_column: str,
    l_value: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enforce l-diversity on data by suppressing violating records.

    Args:
        data: DataFrame to process
        quasi_identifiers: List of quasi-identifier columns
        sensitive_column: Sensitive attribute column
        l_value: Minimum diversity level

    Returns:
        Tuple of (processed DataFrame, statistics)
    """
    qi_cols = [c for c in quasi_identifiers if c in data.columns]

    if not qi_cols or sensitive_column not in data.columns:
        return data, {"error": "Invalid columns"}

    # Find groups with insufficient diversity
    diversity = data.groupby(qi_cols)[sensitive_column].nunique()
    violating_groups = set(diversity[diversity < l_value].index)

    if not violating_groups:
        return data.copy(), {"n_suppressed": 0}

    # Mark violating records
    def is_in_violating_group(row):
        key = tuple(row[qi] for qi in qi_cols)
        return key in violating_groups

    violating_mask = data.apply(is_in_violating_group, axis=1)
    result = data[~violating_mask].copy()

    return result, {
        "n_suppressed": int(violating_mask.sum()),
        "suppression_rate": float(violating_mask.mean()),
    }


def suppress_rare_categories(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.01,
    replacement: str = "OTHER",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Suppress rare categories in categorical columns.

    Args:
        data: DataFrame to process
        columns: Columns to process (None for all categorical)
        threshold: Frequency threshold for suppression
        replacement: Value to replace rare categories with

    Returns:
        Tuple of (processed DataFrame, count of suppressed per column)
    """
    result = data.copy()

    if columns is None:
        columns = result.select_dtypes(include=["object", "category"]).columns.tolist()

    suppressed_counts = {}

    for col in columns:
        if col not in result.columns:
            continue

        value_counts = result[col].value_counts(normalize=True)
        rare_values = value_counts[value_counts < threshold].index.tolist()

        if rare_values:
            mask = result[col].isin(rare_values)
            result.loc[mask, col] = replacement
            suppressed_counts[col] = len(rare_values)

    return result, suppressed_counts
