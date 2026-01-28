"""Multi-table referential integrity constraints."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from genesis.multitable.schema import ForeignKey, RelationalSchema


def validate_referential_integrity(
    tables: Dict[str, pd.DataFrame],
    schema: RelationalSchema,
) -> Dict[str, List[str]]:
    """Validate referential integrity across tables.

    Args:
        tables: Dictionary of table DataFrames
        schema: Relational schema

    Returns:
        Dictionary of violations per table
    """
    violations: Dict[str, List[str]] = {}

    for table_name, table_schema in schema.tables.items():
        if table_name not in tables:
            violations[table_name] = [f"Table '{table_name}' not found"]
            continue

        table_violations = []
        df = tables[table_name]

        for fk in table_schema.foreign_keys:
            if fk.parent_table not in tables:
                table_violations.append(f"Parent table '{fk.parent_table}' not found")
                continue

            parent_df = tables[fk.parent_table]

            if fk.child_column not in df.columns:
                table_violations.append(f"Child column '{fk.child_column}' not found")
                continue

            if fk.parent_column not in parent_df.columns:
                table_violations.append(f"Parent column '{fk.parent_column}' not found")
                continue

            # Check that all child values exist in parent
            child_values = set(df[fk.child_column].dropna().unique())
            parent_values = set(parent_df[fk.parent_column].dropna().unique())

            orphan_values = child_values - parent_values
            if orphan_values:
                n_orphans = len(orphan_values)
                sample = list(orphan_values)[:3]
                table_violations.append(
                    f"FK violation: {n_orphans} orphan values in "
                    f"{fk.child_column} -> {fk.parent_table}.{fk.parent_column} "
                    f"(e.g., {sample})"
                )

        if table_violations:
            violations[table_name] = table_violations

    return violations


def enforce_referential_integrity(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    fk: ForeignKey,
    strategy: str = "filter",
) -> pd.DataFrame:
    """Enforce referential integrity for a foreign key.

    Args:
        child_df: Child table DataFrame
        parent_df: Parent table DataFrame
        fk: Foreign key definition
        strategy: Enforcement strategy ('filter', 'sample')

    Returns:
        Child DataFrame with integrity enforced
    """
    if fk.child_column not in child_df.columns:
        return child_df

    if fk.parent_column not in parent_df.columns:
        return child_df

    parent_values = set(parent_df[fk.parent_column].dropna().unique())

    if strategy == "filter":
        # Remove rows with invalid references
        valid_mask = (
            child_df[fk.child_column].isin(parent_values) | child_df[fk.child_column].isna()
        )
        return child_df[valid_mask].copy()

    elif strategy == "sample":
        # Replace invalid references with valid ones
        result = child_df.copy()
        invalid_mask = (
            ~result[fk.child_column].isin(parent_values) & result[fk.child_column].notna()
        )

        if invalid_mask.any():
            valid_values = list(parent_values)
            n_invalid = invalid_mask.sum()
            replacements = np.random.choice(valid_values, size=n_invalid)
            result.loc[invalid_mask, fk.child_column] = replacements

        return result

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compute_cardinality_distribution(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    fk: ForeignKey,
) -> Dict[str, Any]:
    """Compute cardinality distribution for a relationship.

    Args:
        parent_df: Parent table DataFrame
        child_df: Child table DataFrame
        fk: Foreign key definition

    Returns:
        Dictionary with cardinality statistics
    """
    if fk.child_column not in child_df.columns or fk.parent_column not in parent_df.columns:
        return {"error": "Invalid columns"}

    # Count children per parent
    child_counts = child_df.groupby(fk.child_column).size()
    parent_keys = set(parent_df[fk.parent_column].dropna().unique())

    # Include parents with no children
    counts = []
    for pk in parent_keys:
        counts.append(child_counts.get(pk, 0))

    counts = np.array(counts)

    return {
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "median": float(np.median(counts)),
        "n_with_zero_children": int(np.sum(counts == 0)),
        "cardinality_type": _determine_cardinality_type(counts),
    }


def _determine_cardinality_type(counts: np.ndarray) -> str:
    """Determine the cardinality type from child counts."""
    if len(counts) == 0:
        return "unknown"

    max_count = np.max(counts)
    min_count = np.min(counts)

    if max_count <= 1:
        return "1:1"
    elif min_count >= 1:
        return "1:N (mandatory)"
    else:
        return "1:N (optional)"


def sample_child_counts(
    cardinality_dist: Dict[str, Any],
    n_parents: int,
    method: str = "empirical",
) -> np.ndarray:
    """Sample child counts for parent records.

    Args:
        cardinality_dist: Cardinality distribution statistics
        n_parents: Number of parent records
        method: Sampling method ('empirical', 'poisson')

    Returns:
        Array of child counts per parent
    """
    mean = cardinality_dist.get("mean", 1.0)
    std = cardinality_dist.get("std", 1.0)
    min_val = cardinality_dist.get("min", 0)
    max_val = cardinality_dist.get("max", 10)

    if method == "poisson":
        # Use Poisson distribution
        counts = np.random.poisson(mean, n_parents)
    else:
        # Use truncated normal
        counts = np.random.normal(mean, std, n_parents)
        counts = np.clip(counts, min_val, max_val)
        counts = np.round(counts).astype(int)

    return counts
