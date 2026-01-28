"""MLflow integration for Genesis synthetic data generation.

This module provides integration with MLflow for experiment tracking,
logging generator parameters, quality metrics, and generated data artifacts.

Example:
    >>> import mlflow
    >>> from genesis import SyntheticGenerator
    >>> from genesis.integrations import MLflowCallback, log_generator_to_mlflow
    >>>
    >>> with mlflow.start_run():
    ...     generator = SyntheticGenerator(method='ctgan')
    ...     generator.fit(data)
    ...     synthetic = generator.generate(1000)
    ...     log_generator_to_mlflow(generator, synthetic)
"""

import json
import tempfile
from typing import Any, Dict, Optional

import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


def _check_mlflow_available() -> None:
    """Check if MLflow is installed."""
    try:
        import mlflow  # noqa: F401
    except ImportError as e:
        raise ImportError("MLflow is not installed. Install with: pip install mlflow") from e


class MLflowCallback:
    """Callback for logging training progress to MLflow.

    This callback can be passed to generator.fit() to log training
    metrics to MLflow in real-time.

    Example:
        >>> callback = MLflowCallback()
        >>> generator.fit(data, progress_callback=callback)
    """

    def __init__(
        self,
        log_frequency: int = 10,
        prefix: str = "train",
    ) -> None:
        """Initialize the callback.

        Args:
            log_frequency: How often to log metrics (every N steps)
            prefix: Prefix for metric names
        """
        _check_mlflow_available()
        self.log_frequency = log_frequency
        self.prefix = prefix
        self._step = 0

    def __call__(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log training metrics to MLflow.

        Args:
            epoch: Current epoch
            step: Current step within epoch
            metrics: Dictionary of metrics to log
        """
        import mlflow

        self._step += 1

        if self._step % self.log_frequency != 0:
            return

        # Log metrics with prefix
        for name, value in metrics.items():
            metric_name = f"{self.prefix}/{name}" if self.prefix else name
            mlflow.log_metric(metric_name, value, step=self._step)

        # Always log epoch
        mlflow.log_metric(f"{self.prefix}/epoch", epoch, step=self._step)


def log_generator_to_mlflow(
    generator: Any,
    synthetic_data: Optional[pd.DataFrame] = None,
    real_data: Optional[pd.DataFrame] = None,
    run_name: Optional[str] = None,
    log_model: bool = True,
    log_data_sample: bool = True,
    sample_size: int = 100,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Log a trained generator and its outputs to MLflow.

    Args:
        generator: Fitted generator instance
        synthetic_data: Generated synthetic data
        real_data: Original training data (for quality metrics)
        run_name: Optional run name
        log_model: Whether to log the pickled model
        log_data_sample: Whether to log a sample of synthetic data
        sample_size: Size of data sample to log
        extra_params: Additional parameters to log
        extra_metrics: Additional metrics to log
        tags: Tags for the run

    Returns:
        MLflow run ID

    Example:
        >>> run_id = log_generator_to_mlflow(
        ...     generator,
        ...     synthetic_data=synthetic,
        ...     tags={"project": "customer-data"}
        ... )
    """
    import mlflow

    _check_mlflow_available()

    # Start or get current run
    active_run = mlflow.active_run()
    if active_run is None:
        run = mlflow.start_run(run_name=run_name)
        should_end_run = True
    else:
        run = active_run
        should_end_run = False

    try:
        run_id = run.info.run_id

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log generator parameters
        params = generator.get_parameters()
        config = params.get("config", {})

        mlflow.log_params(
            {
                "generator_type": generator.__class__.__name__,
                "method": str(config.get("method", "unknown")),
                "is_fitted": params.get("is_fitted", False),
                "n_constraints": params.get("n_constraints", 0),
            }
        )

        # Log config parameters
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"config.{key}", value)

        # Log privacy parameters
        privacy = params.get("privacy", {})
        for key, value in privacy.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"privacy.{key}", value)

        # Log extra params
        if extra_params:
            mlflow.log_params(extra_params)

        # Log quality metrics if both datasets available
        if synthetic_data is not None and real_data is not None:
            _log_quality_metrics(real_data, synthetic_data)

        # Log extra metrics
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)

        # Log data sample
        if log_data_sample and synthetic_data is not None:
            sample = synthetic_data.head(sample_size)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                sample.to_csv(f.name, index=False)
                mlflow.log_artifact(f.name, "synthetic_data")

        # Log model
        if log_model:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                generator.save(f.name)
                mlflow.log_artifact(f.name, "model")

        logger.info(f"Logged generator to MLflow run: {run_id}")
        return run_id

    finally:
        if should_end_run:
            mlflow.end_run()


def _log_quality_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> None:
    """Log quality metrics comparing real and synthetic data."""
    import mlflow

    try:
        from genesis.evaluation.evaluator import QualityEvaluator

        evaluator = QualityEvaluator(
            real_data=real_data,
            synthetic_data=synthetic_data,
        )
        report = evaluator.evaluate()

        # Log summary metrics
        metrics = report.to_dict()

        if "statistical" in metrics:
            for key, value in metrics["statistical"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"quality/statistical/{key}", value)

        if "ml_utility" in metrics:
            for key, value in metrics["ml_utility"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"quality/ml_utility/{key}", value)

        if "privacy" in metrics:
            for key, value in metrics["privacy"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"quality/privacy/{key}", value)

        if "overall_score" in metrics:
            mlflow.log_metric("quality/overall_score", metrics["overall_score"])

    except Exception as e:
        logger.warning(f"Could not compute quality metrics: {e}")


def log_quality_report_to_mlflow(
    report: Any,
    artifact_path: str = "quality_report",
) -> None:
    """Log a quality report to MLflow.

    Args:
        report: QualityReport instance
        artifact_path: Path in artifacts to store report
    """
    import mlflow

    _check_mlflow_available()

    # Log metrics
    metrics = report.to_dict()

    def log_nested_metrics(d: Dict, prefix: str = "") -> None:
        for key, value in d.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)
            elif isinstance(value, dict):
                log_nested_metrics(value, metric_name)

    log_nested_metrics(metrics, "quality")

    # Log full report as JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(metrics, f, indent=2, default=str)
        mlflow.log_artifact(f.name, artifact_path)

    logger.info("Logged quality report to MLflow")


class MLflowExperimentTracker:
    """Context manager for tracking synthetic data generation experiments.

    Example:
        >>> with MLflowExperimentTracker("my_experiment") as tracker:
        ...     generator = SyntheticGenerator()
        ...     generator.fit(data, progress_callback=tracker.callback)
        ...     synthetic = generator.generate(1000)
        ...     tracker.log_generator(generator, synthetic)
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional run name
            tags: Optional tags
        """
        _check_mlflow_available()
        import mlflow

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self._run = None
        self.callback = MLflowCallback()

        # Set experiment
        mlflow.set_experiment(experiment_name)

    def __enter__(self) -> "MLflowExperimentTracker":
        """Start the MLflow run."""
        import mlflow

        self._run = mlflow.start_run(run_name=self.run_name)
        if self.tags:
            mlflow.set_tags(self.tags)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the MLflow run."""
        import mlflow

        if exc_type is not None:
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()

    def log_generator(
        self,
        generator: Any,
        synthetic_data: Optional[pd.DataFrame] = None,
        real_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """Log generator to current run."""
        log_generator_to_mlflow(
            generator,
            synthetic_data=synthetic_data,
            real_data=real_data,
            log_model=True,
            **kwargs,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        import mlflow

        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics."""
        import mlflow

        mlflow.log_metrics(metrics)

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._run.info.run_id if self._run else None
