"""Statistical fidelity metrics for evaluation.

This module provides statistical tests and metrics for comparing
real and synthetic data distributions.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Statistical Fidelity Constants
# =============================================================================

# Number of bins for histogram-based comparisons (Jensen-Shannon divergence)
DEFAULT_HISTOGRAM_BINS: int = 50

# Normalization factor for chi-squared statistic to convert to a [0,1] score.
# Chi-squared values are divided by (n_categories * this factor).
CHI_SQUARED_NORMALIZATION_FACTOR: float = 100.0

# Weights for computing overall fidelity score from component scores.
# Column-wise mean score weight vs correlation score weight.
FIDELITY_COLUMN_WEIGHT: float = 0.7
FIDELITY_CORRELATION_WEIGHT: float = 0.3

# Normalization divisor for shape metrics (skewness + kurtosis difference).
SHAPE_METRICS_NORMALIZATION: float = 10.0


def kolmogorov_smirnov_test(
    real: pd.Series,
    synthetic: pd.Series,
) -> Dict[str, float]:
    """Perform Kolmogorov-Smirnov test for continuous columns.

    Args:
        real: Real data column
        synthetic: Synthetic data column

    Returns:
        Dictionary with statistic and p-value
    """
    real_vals = real.dropna().values
    syn_vals = synthetic.dropna().values

    if len(real_vals) == 0 or len(syn_vals) == 0:
        return {"statistic": 1.0, "pvalue": 0.0}

    statistic, pvalue = stats.ks_2samp(real_vals, syn_vals)

    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "score": 1.0 - float(statistic),  # Higher is better
    }


def chi_squared_test(
    real: pd.Series,
    synthetic: pd.Series,
) -> Dict[str, float]:
    """Perform Chi-squared test for categorical columns.

    Args:
        real: Real data column
        synthetic: Synthetic data column

    Returns:
        Dictionary with statistic and p-value
    """
    real_counts = real.value_counts()
    syn_counts = synthetic.value_counts()

    # Align categories
    all_categories = set(real_counts.index) | set(syn_counts.index)

    if len(all_categories) < 2:
        return {"statistic": 0.0, "pvalue": 1.0, "score": 1.0}

    real_aligned = np.array([real_counts.get(cat, 0) for cat in all_categories])
    syn_aligned = np.array([syn_counts.get(cat, 0) for cat in all_categories])

    # Normalize to same scale
    real_freq = real_aligned / real_aligned.sum() if real_aligned.sum() > 0 else real_aligned
    syn_freq = syn_aligned / syn_aligned.sum() if syn_aligned.sum() > 0 else syn_aligned

    # Use actual counts scaled to same total
    total = max(real_aligned.sum(), syn_aligned.sum())
    real_expected = real_freq * total
    syn_observed = syn_freq * total

    # Add small epsilon to avoid division by zero
    real_expected = np.maximum(real_expected, 1e-10)

    try:
        statistic, pvalue = stats.chisquare(syn_observed, real_expected)
        # Normalize statistic to [0,1] score
        normalized_stat = statistic / len(all_categories) if len(all_categories) > 0 else 0
        score = max(0, 1 - normalized_stat / CHI_SQUARED_NORMALIZATION_FACTOR)
    except Exception:
        statistic, pvalue, score = 0.0, 1.0, 1.0

    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "score": float(score),
    }


def wasserstein_distance(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Compute Wasserstein (Earth Mover's) distance.

    Args:
        real: Real data column
        synthetic: Synthetic data column

    Returns:
        Wasserstein distance (lower is better)
    """
    real_vals = real.dropna().values
    syn_vals = synthetic.dropna().values

    if len(real_vals) == 0 or len(syn_vals) == 0:
        return float("inf")

    return float(stats.wasserstein_distance(real_vals, syn_vals))


def jensen_shannon_divergence(
    real: pd.Series,
    synthetic: pd.Series,
    n_bins: int = DEFAULT_HISTOGRAM_BINS,
) -> float:
    """Compute Jensen-Shannon divergence between distributions.

    Args:
        real: Real data column
        synthetic: Synthetic data column
        n_bins: Number of bins for histogram

    Returns:
        JS divergence (0 to 1, lower is better)
    """
    real_vals = real.dropna().values
    syn_vals = synthetic.dropna().values

    if len(real_vals) == 0 or len(syn_vals) == 0:
        return 1.0

    # Create common bins
    all_vals = np.concatenate([real_vals, syn_vals])
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    # Compute histograms
    real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
    syn_hist, _ = np.histogram(syn_vals, bins=bins, density=True)

    # Add epsilon to avoid log(0)
    eps = 1e-10
    real_hist = real_hist + eps
    syn_hist = syn_hist + eps

    # Normalize
    real_hist = real_hist / real_hist.sum()
    syn_hist = syn_hist / syn_hist.sum()

    # Compute JS divergence
    m = 0.5 * (real_hist + syn_hist)
    js_div = 0.5 * (stats.entropy(real_hist, m) + stats.entropy(syn_hist, m))

    return float(js_div)


def total_variation_distance(
    real: pd.Series,
    synthetic: pd.Series,
) -> float:
    """Compute Total Variation distance for categorical columns.

    Args:
        real: Real data column
        synthetic: Synthetic data column

    Returns:
        TV distance (0 to 1, lower is better)
    """
    real_counts = real.value_counts(normalize=True)
    syn_counts = synthetic.value_counts(normalize=True)

    all_categories = set(real_counts.index) | set(syn_counts.index)

    tv = 0.0
    for cat in all_categories:
        real_p = real_counts.get(cat, 0)
        syn_p = syn_counts.get(cat, 0)
        tv += abs(real_p - syn_p)

    return float(tv / 2)


def correlation_comparison(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> Dict[str, float]:
    """Compare correlation matrices between real and synthetic data.

    Args:
        real: Real DataFrame
        synthetic: Synthetic DataFrame

    Returns:
        Dictionary with comparison metrics
    """
    # Get numeric columns
    numeric_cols = real.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_cols if c in synthetic.columns]

    if len(common_cols) < 2:
        return {"mae": 0.0, "rmse": 0.0, "score": 1.0}

    real_corr = real[common_cols].corr().values
    syn_corr = synthetic[common_cols].corr().values

    # Get upper triangle
    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    real_vals = real_corr[mask]
    syn_vals = syn_corr[mask]

    # Handle NaN
    valid = ~(np.isnan(real_vals) | np.isnan(syn_vals))
    if not valid.any():
        return {"mae": 0.0, "rmse": 0.0, "score": 1.0}

    real_vals = real_vals[valid]
    syn_vals = syn_vals[valid]

    mae = float(np.mean(np.abs(real_vals - syn_vals)))
    rmse = float(np.sqrt(np.mean((real_vals - syn_vals) ** 2)))
    score = max(0, 1 - mae)

    return {"mae": mae, "rmse": rmse, "score": score}


def distribution_shape_metrics(
    real: pd.Series,
    synthetic: pd.Series,
) -> Dict[str, float]:
    """Compare distribution shape metrics (skewness, kurtosis).

    Args:
        real: Real data column
        synthetic: Synthetic data column

    Returns:
        Dictionary with shape comparison metrics
    """
    real_vals = real.dropna().values
    syn_vals = synthetic.dropna().values

    if len(real_vals) < 3 or len(syn_vals) < 3:
        return {"skewness_diff": 0.0, "kurtosis_diff": 0.0, "score": 1.0}

    # Convert boolean arrays to float to avoid scipy issues
    if real_vals.dtype == bool:
        real_vals = real_vals.astype(float)
    if syn_vals.dtype == bool:
        syn_vals = syn_vals.astype(float)

    real_skew = float(stats.skew(real_vals))
    syn_skew = float(stats.skew(syn_vals))
    real_kurt = float(stats.kurtosis(real_vals))
    syn_kurt = float(stats.kurtosis(syn_vals))

    skew_diff = abs(real_skew - syn_skew)
    kurt_diff = abs(real_kurt - syn_kurt)

    # Normalize score using shape metrics normalization constant
    score = max(0, 1 - (skew_diff + kurt_diff) / SHAPE_METRICS_NORMALIZATION)

    return {
        "real_skewness": real_skew,
        "synthetic_skewness": syn_skew,
        "skewness_diff": skew_diff,
        "real_kurtosis": real_kurt,
        "synthetic_kurtosis": syn_kurt,
        "kurtosis_diff": kurt_diff,
        "score": score,
    }


def compute_statistical_fidelity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    discrete_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive statistical fidelity metrics.

    Args:
        real: Real DataFrame
        synthetic: Synthetic DataFrame
        discrete_columns: List of discrete/categorical columns

    Returns:
        Dictionary with all statistical fidelity metrics
    """
    discrete_columns = set(discrete_columns or [])
    results = {"column_metrics": {}, "overall": {}}

    column_scores = []

    for col in real.columns:
        if col not in synthetic.columns:
            continue

        col_results = {}

        if col in discrete_columns or real[col].dtype == object:
            # Categorical tests
            chi2_result = chi_squared_test(real[col], synthetic[col])
            tv_dist = total_variation_distance(real[col], synthetic[col])

            col_results["chi_squared"] = chi2_result
            col_results["tv_distance"] = tv_dist
            col_results["score"] = chi2_result["score"]
        else:
            # Continuous tests
            ks_result = kolmogorov_smirnov_test(real[col], synthetic[col])
            w_dist = wasserstein_distance(real[col], synthetic[col])
            js_div = jensen_shannon_divergence(real[col], synthetic[col])
            shape = distribution_shape_metrics(real[col], synthetic[col])

            col_results["ks_test"] = ks_result
            col_results["wasserstein"] = w_dist
            col_results["js_divergence"] = js_div
            col_results["shape"] = shape
            col_results["score"] = ks_result["score"]

        results["column_metrics"][col] = col_results
        column_scores.append(col_results["score"])

    # Correlation comparison
    corr_comparison = correlation_comparison(real, synthetic)
    results["correlation"] = corr_comparison

    # Overall score
    if column_scores:
        results["overall"]["mean_score"] = float(np.mean(column_scores))
        results["overall"]["min_score"] = float(np.min(column_scores))
        results["overall"]["correlation_score"] = corr_comparison["score"]

        # Weighted overall using defined constants
        results["overall"]["fidelity_score"] = (
            FIDELITY_COLUMN_WEIGHT * results["overall"]["mean_score"]
            + FIDELITY_CORRELATION_WEIGHT * results["overall"]["correlation_score"]
        )
    else:
        results["overall"]["fidelity_score"] = 0.0

    return results
