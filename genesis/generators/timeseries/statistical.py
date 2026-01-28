"""Statistical time series generator using ARIMA and decomposition."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genesis.core.config import GeneratorConfig, PrivacyConfig, TimeSeriesConfig
from genesis.core.types import FittingResult, ProgressCallback
from genesis.generators.timeseries.base import BaseTimeSeriesGenerator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class StatisticalTimeSeriesGenerator(BaseTimeSeriesGenerator):
    """Statistical time series generator using ARIMA and decomposition.

    This generator uses traditional statistical methods to model and generate
    time series data. It's faster than deep learning methods and works well
    for simpler time series patterns.

    Example:
        >>> generator = StatisticalTimeSeriesGenerator()
        >>> generator.fit(time_series_data)
        >>> synthetic_data = generator.generate(n_samples=1000)
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        ts_config: Optional[TimeSeriesConfig] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(config, privacy, ts_config)
        self.verbose = verbose

        self._model_params: Dict[str, Dict[str, Any]] = {}
        self._residual_params: Dict[str, Dict[str, float]] = {}

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Fit statistical models to time series data."""
        start_time = time.time()

        self._feature_names = list(data.columns)
        self._n_features = len(self._feature_names)

        if self.verbose:
            logger.info(
                f"Fitting statistical time series model on {len(data)} timesteps, {self._n_features} features"
            )

        # Store raw statistics for each feature
        for col in self._feature_names:
            col_data = data[col].dropna().values

            # Fit statistics
            self._model_params[col] = {
                "mean": float(np.mean(col_data)),
                "std": float(np.std(col_data)),
                "min": float(np.min(col_data)),
                "max": float(np.max(col_data)),
            }

            # Fit AR(1) model (simple autoregressive)
            if len(col_data) > 2:
                # Compute lag-1 autocorrelation
                autocorr = np.corrcoef(col_data[:-1], col_data[1:])[0, 1]
                self._model_params[col]["ar1_coef"] = (
                    float(autocorr) if not np.isnan(autocorr) else 0.0
                )
            else:
                self._model_params[col]["ar1_coef"] = 0.0

            # Compute first differences for residuals
            if len(col_data) > 1:
                diffs = np.diff(col_data)
                self._residual_params[col] = {
                    "mean": float(np.mean(diffs)),
                    "std": float(np.std(diffs)),
                }
            else:
                self._residual_params[col] = {"mean": 0.0, "std": 1.0}

        # Compute cross-correlations
        self._correlation_matrix = data.corr().values

        fitting_time = time.time() - start_time

        if self.verbose:
            logger.info(f"Fitting completed in {fitting_time:.2f}s")

        return FittingResult(
            success=True,
            fitting_time=fitting_time,
            metadata={"n_features": self._n_features},
        )

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Generate synthetic time series using fitted models."""
        result = {}

        for col in self._feature_names:
            params = self._model_params[col]
            residual_params = self._residual_params[col]

            # Initialize with random starting point
            series = np.zeros(n_samples)
            series[0] = np.random.normal(params["mean"], params["std"])

            # Generate using AR(1) process
            ar_coef = params["ar1_coef"]
            noise_std = residual_params["std"]

            for t in range(1, n_samples):
                noise = np.random.normal(0, noise_std)
                series[t] = (1 - ar_coef) * params["mean"] + ar_coef * series[t - 1] + noise

            # Clip to original range
            series = np.clip(series, params["min"], params["max"])
            result[col] = series

        return pd.DataFrame(result)
