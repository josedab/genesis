"""Weights & Biases (wandb) integration for Genesis synthetic data generation.

This module provides integration with Weights & Biases for experiment tracking,
logging generator parameters, quality metrics, and generated data artifacts.

Example:
    >>> import wandb
    >>> from genesis import SyntheticGenerator
    >>> from genesis.integrations import WandbCallback, log_generator_to_wandb
    >>>
    >>> wandb.init(project="synthetic-data")
    >>> generator = SyntheticGenerator(method='ctgan')
    >>> generator.fit(data, progress_callback=WandbCallback())
    >>> synthetic = generator.generate(1000)
    >>> log_generator_to_wandb(generator, synthetic)
    >>> wandb.finish()
"""

import json
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


def _check_wandb_available() -> None:
    """Check if wandb is installed."""
    try:
        import wandb  # noqa: F401
    except ImportError as e:
        raise ImportError("Weights & Biases is not installed. Install with: pip install wandb") from e


class WandbCallback:
    """Callback for logging training progress to Weights & Biases.

    This callback can be passed to generator.fit() to log training
    metrics to W&B in real-time.

    Example:
        >>> callback = WandbCallback()
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
        _check_wandb_available()
        self.log_frequency = log_frequency
        self.prefix = prefix
        self._step = 0

    def __call__(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log training metrics to W&B.

        Args:
            epoch: Current epoch
            step: Current step within epoch
            metrics: Dictionary of metrics to log
        """
        import wandb

        self._step += 1

        if self._step % self.log_frequency != 0:
            return

        # Prepare metrics with prefix
        log_dict = {}
        for name, value in metrics.items():
            metric_name = f"{self.prefix}/{name}" if self.prefix else name
            log_dict[metric_name] = value

        log_dict[f"{self.prefix}/epoch"] = epoch
        log_dict[f"{self.prefix}/step"] = self._step

        wandb.log(log_dict)


def log_generator_to_wandb(
    generator: Any,
    synthetic_data: Optional[pd.DataFrame] = None,
    real_data: Optional[pd.DataFrame] = None,
    log_model: bool = True,
    log_data_sample: bool = True,
    sample_size: int = 100,
    extra_config: Optional[Dict[str, Any]] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Log a trained generator and its outputs to Weights & Biases.

    Args:
        generator: Fitted generator instance
        synthetic_data: Generated synthetic data
        real_data: Original training data (for quality metrics)
        log_model: Whether to log the pickled model as artifact
        log_data_sample: Whether to log a sample of synthetic data
        sample_size: Size of data sample to log
        extra_config: Additional config to log
        extra_metrics: Additional metrics to log

    Example:
        >>> log_generator_to_wandb(
        ...     generator,
        ...     synthetic_data=synthetic,
        ...     extra_metrics={"custom_score": 0.95}
        ... )
    """
    import wandb

    _check_wandb_available()

    if wandb.run is None:
        logger.warning("No active W&B run. Call wandb.init() first.")
        return

    # Log generator parameters to config
    params = generator.get_parameters()
    config = params.get("config", {})
    privacy = params.get("privacy", {})

    wandb.config.update(
        {
            "generator_type": generator.__class__.__name__,
            "method": str(config.get("method", "unknown")),
            "is_fitted": params.get("is_fitted", False),
            "n_constraints": params.get("n_constraints", 0),
            **{
                f"config_{k}": v
                for k, v in config.items()
                if isinstance(v, (str, int, float, bool))
            },
            **{
                f"privacy_{k}": v
                for k, v in privacy.items()
                if isinstance(v, (str, int, float, bool))
            },
        }
    )

    if extra_config:
        wandb.config.update(extra_config)

    # Log quality metrics
    if synthetic_data is not None and real_data is not None:
        _log_quality_metrics_wandb(real_data, synthetic_data)

    # Log extra metrics
    if extra_metrics:
        wandb.log(extra_metrics)

    # Log data sample as table
    if log_data_sample and synthetic_data is not None:
        sample = synthetic_data.head(sample_size)
        wandb.log({"synthetic_data_sample": wandb.Table(dataframe=sample)})

    # Log model as artifact
    if log_model:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            generator.save(f.name)
            artifact = wandb.Artifact(
                name=f"generator-{wandb.run.id}",
                type="model",
                description="Trained synthetic data generator",
            )
            artifact.add_file(f.name, "generator.pkl")
            wandb.log_artifact(artifact)

    logger.info("Logged generator to W&B")


def _log_quality_metrics_wandb(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> None:
    """Log quality metrics comparing real and synthetic data."""
    import wandb

    try:
        from genesis.evaluation.evaluator import QualityEvaluator

        evaluator = QualityEvaluator(
            real_data=real_data,
            synthetic_data=synthetic_data,
        )
        report = evaluator.evaluate()

        # Log summary metrics
        metrics = report.to_dict()

        log_dict = {}

        if "statistical" in metrics:
            for key, value in metrics["statistical"].items():
                if isinstance(value, (int, float)):
                    log_dict[f"quality/statistical/{key}"] = value

        if "ml_utility" in metrics:
            for key, value in metrics["ml_utility"].items():
                if isinstance(value, (int, float)):
                    log_dict[f"quality/ml_utility/{key}"] = value

        if "privacy" in metrics:
            for key, value in metrics["privacy"].items():
                if isinstance(value, (int, float)):
                    log_dict[f"quality/privacy/{key}"] = value

        if "overall_score" in metrics:
            log_dict["quality/overall_score"] = metrics["overall_score"]

        wandb.log(log_dict)

        # Also log as summary
        wandb.run.summary.update(log_dict)

    except Exception as e:
        logger.warning(f"Could not compute quality metrics: {e}")


def log_quality_report_to_wandb(
    report: Any,
) -> None:
    """Log a quality report to Weights & Biases.

    Args:
        report: QualityReport instance
    """
    import wandb

    _check_wandb_available()

    if wandb.run is None:
        logger.warning("No active W&B run. Call wandb.init() first.")
        return

    # Log metrics
    metrics = report.to_dict()

    def flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
        result = {}
        for key, value in d.items():
            metric_name = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)):
                result[metric_name] = value
            elif isinstance(value, dict):
                result.update(flatten_dict(value, metric_name))
        return result

    flat_metrics = flatten_dict(metrics, "quality")
    wandb.log(flat_metrics)
    wandb.run.summary.update(flat_metrics)

    # Log full report as JSON artifact
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(metrics, f, indent=2, default=str)
        artifact = wandb.Artifact(
            name=f"quality-report-{wandb.run.id}",
            type="report",
        )
        artifact.add_file(f.name, "quality_report.json")
        wandb.log_artifact(artifact)

    logger.info("Logged quality report to W&B")


class WandbExperimentTracker:
    """Context manager for tracking synthetic data generation experiments with W&B.

    Example:
        >>> with WandbExperimentTracker(project="synth-data") as tracker:
        ...     generator = SyntheticGenerator()
        ...     generator.fit(data, progress_callback=tracker.callback)
        ...     synthetic = generator.generate(1000)
        ...     tracker.log_generator(generator, synthetic)
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            project: W&B project name
            name: Optional run name
            config: Initial config dict
            tags: Optional tags
            notes: Optional notes
        """
        _check_wandb_available()

        self.project = project
        self.name = name
        self.config = config or {}
        self.tags = tags
        self.notes = notes
        self.callback = WandbCallback()

    def __enter__(self) -> "WandbExperimentTracker":
        """Start the W&B run."""
        import wandb

        wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the W&B run."""
        import wandb

        if exc_type is not None:
            wandb.run.summary["error"] = str(exc_val)
        wandb.finish()

    def log_generator(
        self,
        generator: Any,
        synthetic_data: Optional[pd.DataFrame] = None,
        real_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """Log generator to current run."""
        log_generator_to_wandb(
            generator,
            synthetic_data=synthetic_data,
            real_data=real_data,
            log_model=True,
            **kwargs,
        )

    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics."""
        import wandb

        wandb.log(metrics)

    def log_table(self, name: str, df: pd.DataFrame) -> None:
        """Log a DataFrame as a table."""
        import wandb

        wandb.log({name: wandb.Table(dataframe=df)})

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        import wandb

        return wandb.run.id if wandb.run else None


def create_wandb_sweep_config(
    method_options: List[str] = None,
    epoch_range: tuple = (100, 500),
    batch_size_options: List[int] = None,
) -> Dict[str, Any]:
    """Create a W&B sweep configuration for hyperparameter tuning.

    Args:
        method_options: Generator methods to try
        epoch_range: Range of epochs (min, max)
        batch_size_options: Batch sizes to try

    Returns:
        Sweep configuration dict

    Example:
        >>> sweep_config = create_wandb_sweep_config()
        >>> sweep_id = wandb.sweep(sweep_config, project="synth-data")
        >>> wandb.agent(sweep_id, train_fn, count=10)
    """
    method_options = method_options or ["ctgan", "tvae", "gaussian_copula"]
    batch_size_options = batch_size_options or [100, 250, 500]

    return {
        "method": "bayes",
        "metric": {
            "name": "quality/overall_score",
            "goal": "maximize",
        },
        "parameters": {
            "method": {"values": method_options},
            "epochs": {
                "distribution": "int_uniform",
                "min": epoch_range[0],
                "max": epoch_range[1],
            },
            "batch_size": {"values": batch_size_options},
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
        },
    }
