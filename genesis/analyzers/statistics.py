"""Statistical analysis utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class UnivariateStats:
    """Univariate statistics for a column."""

    column: str
    count: int
    missing_count: int
    missing_rate: float
    unique_count: int
    dtype: str

    # Numeric stats (None for non-numeric)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    # Categorical stats (None for numeric)
    mode: Optional[Any] = None
    mode_frequency: Optional[float] = None
    categories: Optional[List[Any]] = None
    category_frequencies: Optional[Dict[Any, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "column": self.column,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_rate": self.missing_rate,
            "unique_count": self.unique_count,
            "dtype": self.dtype,
        }

        if self.mean is not None:
            result.update(
                {
                    "mean": self.mean,
                    "std": self.std,
                    "min": self.min_val,
                    "max": self.max_val,
                    "median": self.median,
                    "q1": self.q1,
                    "q3": self.q3,
                    "skewness": self.skewness,
                    "kurtosis": self.kurtosis,
                }
            )

        if self.categories is not None:
            result.update(
                {
                    "mode": self.mode,
                    "mode_frequency": self.mode_frequency,
                    "n_categories": len(self.categories),
                }
            )

        return result


@dataclass
class DatasetStats:
    """Statistical summary for an entire dataset."""

    n_rows: int
    n_columns: int
    n_numeric: int
    n_categorical: int
    total_missing: int
    total_missing_rate: float
    memory_usage_bytes: int
    column_stats: Dict[str, UnivariateStats] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "total_missing": self.total_missing,
            "total_missing_rate": self.total_missing_rate,
            "memory_usage_bytes": self.memory_usage_bytes,
            "columns": {k: v.to_dict() for k, v in self.column_stats.items()},
        }


class StatisticalAnalyzer:
    """Analyzer for computing comprehensive statistics on data."""

    def __init__(
        self,
        compute_correlations: bool = True,
        max_categories_display: int = 20,
    ) -> None:
        """Initialize the statistical analyzer.

        Args:
            compute_correlations: Whether to compute correlation matrix
            max_categories_display: Max categories to include in output
        """
        self.compute_correlations = compute_correlations
        self.max_categories_display = max_categories_display

    def analyze(self, data: pd.DataFrame) -> DatasetStats:
        """Compute comprehensive statistics for a DataFrame.

        Args:
            data: DataFrame to analyze

        Returns:
            DatasetStats with all computed statistics
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

        stats_result = DatasetStats(
            n_rows=len(data),
            n_columns=len(data.columns),
            n_numeric=len(numeric_cols),
            n_categorical=len(categorical_cols),
            total_missing=data.isna().sum().sum(),
            total_missing_rate=data.isna().sum().sum() / data.size if data.size > 0 else 0,
            memory_usage_bytes=data.memory_usage(deep=True).sum(),
        )

        # Compute column-wise statistics
        for col in data.columns:
            col_stats = self._compute_column_stats(data[col])
            stats_result.column_stats[col] = col_stats

        # Compute correlation matrix for numeric columns
        if self.compute_correlations and len(numeric_cols) > 1:
            stats_result.correlation_matrix = data[numeric_cols].corr()

        return stats_result

    def _compute_column_stats(self, col: pd.Series) -> UnivariateStats:
        """Compute statistics for a single column.

        Args:
            col: Column data

        Returns:
            UnivariateStats for the column
        """
        n_total = len(col)
        n_missing = col.isna().sum()
        non_null = col.dropna()

        base_stats = UnivariateStats(
            column=col.name,
            count=n_total,
            missing_count=n_missing,
            missing_rate=n_missing / n_total if n_total > 0 else 0,
            unique_count=col.nunique(),
            dtype=str(col.dtype),
        )

        if pd.api.types.is_numeric_dtype(col) and len(non_null) > 0:
            # Numeric statistics
            base_stats.mean = float(non_null.mean())
            base_stats.std = float(non_null.std())
            base_stats.min_val = float(non_null.min())
            base_stats.max_val = float(non_null.max())
            base_stats.median = float(non_null.median())
            base_stats.q1 = float(non_null.quantile(0.25))
            base_stats.q3 = float(non_null.quantile(0.75))

            if len(non_null) > 2:
                base_stats.skewness = float(non_null.skew())
                base_stats.kurtosis = float(non_null.kurtosis())
        else:
            # Categorical statistics
            value_counts = non_null.value_counts()
            if len(value_counts) > 0:
                base_stats.mode = value_counts.index[0]
                base_stats.mode_frequency = (
                    value_counts.iloc[0] / len(non_null) if len(non_null) > 0 else 0
                )

                # Store categories (limited)
                if len(value_counts) <= self.max_categories_display:
                    base_stats.categories = value_counts.index.tolist()
                    base_stats.category_frequencies = (value_counts / len(non_null)).to_dict()
                else:
                    base_stats.categories = value_counts.head(
                        self.max_categories_display
                    ).index.tolist()

        return base_stats

    def compare_distributions(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Compare distributions between real and synthetic data.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data

        Returns:
            Dictionary of comparison metrics per column
        """
        results = {}

        for col in real_data.columns:
            if col not in synthetic_data.columns:
                continue

            col_results = {}
            real_col = real_data[col].dropna()
            syn_col = synthetic_data[col].dropna()

            if len(real_col) == 0 or len(syn_col) == 0:
                continue

            if pd.api.types.is_numeric_dtype(real_col):
                # KS test for numeric
                ks_stat, ks_pval = stats.ks_2samp(real_col, syn_col)
                col_results["ks_statistic"] = ks_stat
                col_results["ks_pvalue"] = ks_pval

                # Wasserstein distance
                col_results["wasserstein"] = float(stats.wasserstein_distance(real_col, syn_col))

                # Mean/std comparison
                col_results["mean_diff"] = abs(real_col.mean() - syn_col.mean())
                col_results["std_diff"] = abs(real_col.std() - syn_col.std())

            else:
                # Chi-squared for categorical
                real_counts = real_col.value_counts()
                syn_counts = syn_col.value_counts()

                # Align categories
                all_categories = set(real_counts.index) | set(syn_counts.index)
                real_aligned = [real_counts.get(cat, 0) for cat in all_categories]
                syn_aligned = [syn_counts.get(cat, 0) for cat in all_categories]

                # Normalize to same scale
                real_total = sum(real_aligned)
                syn_total = sum(syn_aligned)
                if real_total > 0 and syn_total > 0:
                    real_freq = [c / real_total for c in real_aligned]
                    syn_freq = [c / syn_total for c in syn_aligned]

                    # Total variation distance
                    col_results["tv_distance"] = 0.5 * sum(
                        abs(r - s) for r, s in zip(real_freq, syn_freq)
                    )

                    # Jensen-Shannon divergence
                    col_results["js_divergence"] = self._js_divergence(
                        np.array(real_freq), np.array(syn_freq)
                    )

            results[col] = col_results

        return results

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        m = 0.5 * (p + q)
        return float(0.5 * (stats.entropy(p, m) + stats.entropy(q, m)))


def compute_correlation_matrix(
    data: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.

    Args:
        data: DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix as DataFrame
    """
    numeric_data = data.select_dtypes(include=[np.number])
    return numeric_data.corr(method=method)


def compare_correlation_matrices(
    real_corr: pd.DataFrame,
    synthetic_corr: pd.DataFrame,
) -> Dict[str, float]:
    """Compare two correlation matrices.

    Args:
        real_corr: Correlation matrix from real data
        synthetic_corr: Correlation matrix from synthetic data

    Returns:
        Dictionary of comparison metrics
    """
    # Align columns
    common_cols = list(set(real_corr.columns) & set(synthetic_corr.columns))
    if not common_cols:
        return {}

    real_aligned = real_corr.loc[common_cols, common_cols]
    syn_aligned = synthetic_corr.loc[common_cols, common_cols]

    # Flatten upper triangle
    mask = np.triu(np.ones_like(real_aligned, dtype=bool), k=1)
    real_vals = real_aligned.values[mask]
    syn_vals = syn_aligned.values[mask]

    if len(real_vals) == 0:
        return {}

    return {
        "correlation_mae": float(np.mean(np.abs(real_vals - syn_vals))),
        "correlation_rmse": float(np.sqrt(np.mean((real_vals - syn_vals) ** 2))),
        "correlation_max_diff": float(np.max(np.abs(real_vals - syn_vals))),
    }
