"""Configuration classes for streaming generation."""

from dataclasses import dataclass, field
from typing import List, Optional

import time


@dataclass
class StreamingConfig:
    """Configuration for streaming generation.

    Attributes:
        batch_size: Number of samples per batch.
        buffer_size: Size of the internal buffer for async generation.
        max_batches: Maximum number of batches to generate (None for infinite).
        delay_seconds: Delay between batch generations.
        auto_update: Whether to automatically update the model with new data.
        update_frequency: Number of samples before triggering an update.
    """

    batch_size: int = 100
    buffer_size: int = 10
    max_batches: Optional[int] = None
    delay_seconds: float = 0.0
    auto_update: bool = False
    update_frequency: int = 1000  # Update after N samples


@dataclass
class StreamingStats:
    """Statistics for streaming generation.

    Attributes:
        batches_generated: Total number of batches generated.
        samples_generated: Total number of samples generated.
        updates_applied: Number of incremental model updates applied.
        start_time: Timestamp when streaming started.
        errors: List of error messages encountered.
    """

    batches_generated: int = 0
    samples_generated: int = 0
    updates_applied: int = 0
    start_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    @property
    def samples_per_second(self) -> float:
        """Calculate samples per second throughput."""
        if self.start_time is None or self.samples_generated == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.samples_generated / elapsed if elapsed > 0 else 0.0

    def reset(self) -> None:
        """Reset all statistics."""
        self.batches_generated = 0
        self.samples_generated = 0
        self.updates_applied = 0
        self.start_time = None
        self.errors.clear()
