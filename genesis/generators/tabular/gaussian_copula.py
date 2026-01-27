"""Gaussian Copula generator for tabular data."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.types import ColumnList, FittingResult, ProgressCallback
from genesis.generators.tabular.base import BaseTabularGenerator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class GaussianCopulaGenerator(BaseTabularGenerator):
    """Gaussian Copula generator for synthetic tabular data.

    This generator uses a Gaussian copula to model dependencies between
    columns while fitting marginal distributions separately. It's faster
    than deep learning methods and works well for smaller datasets.

    Example:
        >>> generator = GaussianCopulaGenerator()
        >>> generator.fit(real_data, discrete_columns=['gender'])
        >>> synthetic_data = generator.generate(n_samples=1000)
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize Gaussian Copula generator.

        Args:
            config: Generator configuration
            privacy: Privacy configuration
            verbose: Whether to print progress
        """
        super().__init__(config, privacy)
        self.verbose = verbose

        # Learned parameters
        self._correlation_matrix: Optional[np.ndarray] = None
        self._marginal_params: Dict[str, Dict[str, Any]] = {}
        self._column_names: List[str] = []
        self._discrete_columns: List[str] = []
        self._n_samples_trained: int = 0

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[ColumnList] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit Gaussian Copula to the training data."""
        start_time = time.time()

        self._column_names = list(data.columns)
        self._discrete_columns = discrete_columns or []
        self._n_samples_trained = len(data)

        if self.verbose:
            logger.info(
                f"Fitting Gaussian Copula on {len(data)} samples, {len(data.columns)} columns"
            )

        # Step 1: Fit marginal distributions
        self._fit_marginals(data)

        # Step 2: Transform to uniform [0, 1] using CDFs
        uniform_data = self._to_uniform(data)

        # Step 3: Transform to normal using inverse CDF
        normal_data = stats.norm.ppf(np.clip(uniform_data, 0.001, 0.999))

        # Step 4: Estimate correlation matrix
        self._correlation_matrix = np.corrcoef(normal_data, rowvar=False)

        # Handle potential numerical issues
        self._correlation_matrix = self._make_positive_definite(self._correlation_matrix)

        fitting_time = time.time() - start_time

        if self.verbose:
            logger.info(f"Fitting completed in {fitting_time:.2f}s")

        return FittingResult(
            success=True,
            fitting_time=fitting_time,
            metadata={
                "n_columns": len(self._column_names),
                "n_samples": self._n_samples_trained,
            },
        )

    def _fit_marginals(self, data: pd.DataFrame) -> None:
        """Fit marginal distributions for each column."""
        for col in self._column_names:
            col_data = data[col].dropna()

            if col in self._discrete_columns or data[col].dtype == object:
                # Categorical: store value counts
                value_counts = col_data.value_counts(normalize=True)
                self._marginal_params[col] = {
                    "type": "categorical",
                    "categories": value_counts.index.tolist(),
                    "probabilities": value_counts.values.tolist(),
                }
            else:
                # Continuous: fit distribution
                params = self._fit_continuous_distribution(col_data.values)
                self._marginal_params[col] = params

    def _fit_continuous_distribution(self, values: np.ndarray) -> Dict[str, Any]:
        """Fit a continuous distribution to the data.

        Tries multiple distributions and selects the best fit.
        """
        # Try different distributions
        distributions = [
            ("norm", stats.norm),
            ("beta", stats.beta),
            ("gamma", stats.gamma),
            ("lognorm", stats.lognorm),
        ]

        best_dist = None
        best_params = None
        best_ks = float("inf")

        for name, dist in distributions:
            try:
                params = dist.fit(values)
                ks_stat, _ = stats.kstest(values, name, params)

                if ks_stat < best_ks:
                    best_ks = ks_stat
                    best_dist = name
                    best_params = params
            except Exception:
                continue

        if best_dist is None:
            # Fallback to normal
            best_dist = "norm"
            best_params = (np.mean(values), np.std(values))

        return {
            "type": "continuous",
            "distribution": best_dist,
            "params": best_params,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def _to_uniform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data to uniform [0, 1] using fitted CDFs."""
        uniform = np.zeros((len(data), len(self._column_names)))

        for i, col in enumerate(self._column_names):
            col_data = data[col].values
            params = self._marginal_params[col]

            if params["type"] == "categorical":
                # Map categories to cumulative probabilities
                categories = params["categories"]
                probs = params["probabilities"]
                cum_probs = np.cumsum(probs)

                for j, val in enumerate(col_data):
                    if pd.isna(val):
                        uniform[j, i] = 0.5
                    elif val in categories:
                        idx = categories.index(val)
                        # Use midpoint of CDF interval
                        lower = cum_probs[idx - 1] if idx > 0 else 0
                        upper = cum_probs[idx]
                        uniform[j, i] = (lower + upper) / 2
                    else:
                        uniform[j, i] = 0.5
            else:
                # Use fitted distribution CDF
                dist = getattr(stats, params["distribution"])
                uniform[:, i] = dist.cdf(col_data, *params["params"])

        return uniform

    def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite."""
        # Handle NaN values
        matrix = np.nan_to_num(matrix, nan=0.0)

        # Ensure diagonal is 1
        np.fill_diagonal(matrix, 1.0)

        # Make symmetric
        matrix = (matrix + matrix.T) / 2

        # Eigenvalue decomposition to fix non-positive definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Clip small/negative eigenvalues
        eigenvalues = np.clip(eigenvalues, 1e-10, None)

        # Reconstruct
        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Normalize to correlation matrix
        d = np.sqrt(np.diag(matrix))
        matrix = matrix / np.outer(d, d)

        return matrix

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data using Gaussian Copula."""
        if self._correlation_matrix is None:
            raise RuntimeError("Model not fitted")

        # Step 1: Sample from multivariate normal with learned correlation
        normal_samples = np.random.multivariate_normal(
            mean=np.zeros(len(self._column_names)),
            cov=self._correlation_matrix,
            size=n_samples,
        )

        # Step 2: Transform to uniform using standard normal CDF
        uniform_samples = stats.norm.cdf(normal_samples)

        # Step 3: Transform to original space using inverse CDFs
        result = {}

        for i, col in enumerate(self._column_names):
            params = self._marginal_params[col]
            uniform_col = uniform_samples[:, i]

            if params["type"] == "categorical":
                result[col] = self._inverse_categorical(uniform_col, params)
            else:
                result[col] = self._inverse_continuous(uniform_col, params)

        return pd.DataFrame(result)

    def _inverse_categorical(
        self,
        uniform_values: np.ndarray,
        params: Dict[str, Any],
    ) -> pd.Series:
        """Inverse transform for categorical columns."""
        categories = params["categories"]
        probs = params["probabilities"]
        cum_probs = np.cumsum(probs)

        result = []
        for u in uniform_values:
            idx = np.searchsorted(cum_probs, u)
            idx = min(idx, len(categories) - 1)
            result.append(categories[idx])

        return pd.Series(result)

    def _inverse_continuous(
        self,
        uniform_values: np.ndarray,
        params: Dict[str, Any],
    ) -> pd.Series:
        """Inverse transform for continuous columns."""
        dist = getattr(stats, params["distribution"])
        values = dist.ppf(uniform_values, *params["params"])

        # Clip to original range
        values = np.clip(values, params["min"], params["max"])

        return pd.Series(values)

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get the learned correlation matrix.

        Returns:
            Correlation matrix as DataFrame, or None if not fitted
        """
        if self._correlation_matrix is None:
            return None

        return pd.DataFrame(
            self._correlation_matrix,
            index=self._column_names,
            columns=self._column_names,
        )
