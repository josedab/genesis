"""Logging utilities for Genesis."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Default console for rich output
console = Console()

# Logger instances cache
_loggers: dict = {}


def get_logger(
    name: str = "genesis",
    level: int = logging.INFO,
    use_rich: bool = True,
) -> logging.Logger:
    """Get or create a logger with the specified configuration.

    Args:
        name: Logger name
        level: Logging level
        use_rich: Whether to use rich formatting

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    if use_rich:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    logger.addHandler(handler)
    _loggers[name] = logger

    return logger


def set_log_level(level: int, name: str = "genesis") -> None:
    """Set the log level for a logger.

    Args:
        level: New logging level
        name: Logger name
    """
    logger = get_logger(name)
    logger.setLevel(level)


def create_progress(
    description: str = "Processing",
    total: Optional[int] = None,
    transient: bool = False,
) -> Progress:
    """Create a rich progress bar.

    Args:
        description: Description for the progress bar
        total: Total number of steps (None for indeterminate)
        transient: Whether to remove the progress bar when done

    Returns:
        Progress instance
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    return Progress(*columns, console=console, transient=transient)


class ProgressLogger:
    """Progress logger that combines logging and progress bars."""

    def __init__(
        self,
        total: int,
        description: str = "Training",
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10,
    ) -> None:
        """Initialize progress logger.

        Args:
            total: Total number of steps
            description: Description for the progress
            logger: Logger instance to use
            log_interval: Interval for logging updates (in percentage)
        """
        self.total = total
        self.description = description
        self.logger = logger or get_logger()
        self.log_interval = log_interval

        self._current = 0
        self._last_logged_percent = 0
        self._progress: Optional[Progress] = None
        self._task_id = None

    def __enter__(self) -> "ProgressLogger":
        """Enter context manager."""
        self._progress = create_progress(self.description, self.total)
        self._progress.start()
        self._task_id = self._progress.add_task(self.description, total=self.total)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if self._progress:
            self._progress.stop()

    def update(self, n: int = 1, **kwargs) -> None:
        """Update progress.

        Args:
            n: Number of steps to advance
            **kwargs: Additional data to log
        """
        self._current += n

        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=n)

        # Log at intervals
        percent = int(100 * self._current / self.total) if self.total > 0 else 0
        if percent >= self._last_logged_percent + self.log_interval:
            self._last_logged_percent = percent
            msg = f"{self.description}: {percent}%"
            if kwargs:
                extras = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in kwargs.items()
                )
                msg += f" ({extras})"
            self.logger.info(msg)

    def set_description(self, description: str) -> None:
        """Update the description."""
        self.description = description
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, description=description)


class TrainingLogger:
    """Logger specifically for training loops."""

    def __init__(
        self,
        n_epochs: int,
        logger: Optional[logging.Logger] = None,
        log_every_n_epochs: int = 10,
    ) -> None:
        """Initialize training logger.

        Args:
            n_epochs: Total number of epochs
            logger: Logger instance
            log_every_n_epochs: Log every N epochs
        """
        self.n_epochs = n_epochs
        self.logger = logger or get_logger()
        self.log_every_n_epochs = log_every_n_epochs

        self._metrics_history: dict = {}

    def log_epoch(
        self,
        epoch: int,
        metrics: dict,
        force: bool = False,
    ) -> None:
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
            force: Force logging regardless of interval
        """
        # Store metrics
        for key, value in metrics.items():
            if key not in self._metrics_history:
                self._metrics_history[key] = []
            self._metrics_history[key].append(value)

        # Log if at interval or forced
        if force or epoch % self.log_every_n_epochs == 0 or epoch == self.n_epochs - 1:
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()
            )
            self.logger.info(f"Epoch {epoch + 1}/{self.n_epochs} | {metrics_str}")

    def get_history(self) -> dict:
        """Get the metrics history."""
        return self._metrics_history.copy()

    def log_summary(self) -> None:
        """Log a summary of training."""
        if not self._metrics_history:
            return

        self.logger.info("=" * 50)
        self.logger.info("Training Summary")
        self.logger.info("=" * 50)

        for key, values in self._metrics_history.items():
            if values:
                final_val = values[-1]
                best_val = min(values) if "loss" in key.lower() else max(values)
                self.logger.info(f"{key}: final={final_val:.4f}, best={best_val:.4f}")
