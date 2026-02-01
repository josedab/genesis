"""Core streaming generator implementation."""

import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, Optional

import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.exceptions import NotFittedError
from genesis.streaming.config import StreamingConfig, StreamingStats
from genesis.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Constants for incremental learning
DEFAULT_PARTIAL_FIT_WEIGHT = 0.1
MIN_BUFFER_SIZE_FOR_UPDATE = 5


class StreamingGenerator:
    """Generator that supports streaming and incremental updates.

    This generator is optimized for:
    - Continuous data generation
    - Incremental model updates
    - Memory-efficient batch generation

    Supports dependency injection for the underlying generator,
    improving testability.

    Example:
        >>> stream = StreamingGenerator(method='gaussian_copula')
        >>> stream.fit(initial_data)
        >>> for batch in stream.generate_stream(n_batches=10):
        ...     process(batch)
    """

    def __init__(
        self,
        method: str = "gaussian_copula",
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        streaming_config: Optional[StreamingConfig] = None,
        generator: Optional[BaseGenerator] = None,
    ) -> None:
        """Initialize the streaming generator.

        Args:
            method: Generator method ('gaussian_copula' recommended for streaming)
            config: Generator configuration
            privacy: Privacy configuration
            streaming_config: Streaming-specific configuration
            generator: Optional pre-configured generator (for dependency injection)
        """
        self.method = method
        self.config = config or GeneratorConfig()
        self.privacy = privacy or PrivacyConfig()
        self.streaming_config = streaming_config or StreamingConfig()

        self._generator: Optional[BaseGenerator] = generator
        self._is_fitted = generator is not None and getattr(generator, "_is_fitted", False)
        self._stats = StreamingStats()
        self._buffer: queue.Queue = queue.Queue(maxsize=self.streaming_config.buffer_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # For incremental updates
        self._data_buffer: List[pd.DataFrame] = []
        self._samples_since_update = 0
        self._discrete_columns: List[str] = []

    def fit(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
    ) -> "StreamingGenerator":
        """Fit the generator to initial data.

        Args:
            data: Training DataFrame
            discrete_columns: List of categorical column names

        Returns:
            Self for method chaining
        """
        from genesis import SyntheticGenerator

        if self._generator is None:
            self._generator = SyntheticGenerator(
                method=self.method,
                config=self.config,
                privacy=self.privacy,
            )

        self._generator.fit(data, discrete_columns=discrete_columns)
        self._is_fitted = True
        self._discrete_columns = discrete_columns or []

        logger.info(f"Fitted streaming generator with {len(data)} samples")
        return self

    def partial_fit(
        self,
        new_data: pd.DataFrame,
        weight: float = DEFAULT_PARTIAL_FIT_WEIGHT,
    ) -> "StreamingGenerator":
        """Incrementally update the model with new data.

        For Gaussian Copula, this updates the learned distributions
        using a weighted combination of old and new statistics.

        Args:
            new_data: New data to incorporate
            weight: Weight for new data (0-1, higher = more influence)

        Returns:
            Self for method chaining
        """
        if not self._is_fitted:
            raise NotFittedError("StreamingGenerator")

        self._data_buffer.append(new_data)

        # Trigger update if we have enough data
        if len(self._data_buffer) >= MIN_BUFFER_SIZE_FOR_UPDATE:
            combined = pd.concat(self._data_buffer, ignore_index=True)

            # Refit with weighted sampling
            if (
                hasattr(self._generator, "_original_data")
                and self._generator._original_data is not None
            ):
                original = self._generator._original_data

                # Sample to balance old and new
                n_original = int(len(original) * (1 - weight))
                n_new = int(len(combined) * weight)

                sampled_original = original.sample(n=min(n_original, len(original)), replace=False)
                sampled_new = combined.sample(n=min(n_new, len(combined)), replace=False)

                refit_data = pd.concat([sampled_original, sampled_new], ignore_index=True)
                self._generator.fit(refit_data, discrete_columns=self._discrete_columns)

            self._data_buffer = []
            self._stats.updates_applied += 1
            logger.info(f"Applied incremental update #{self._stats.updates_applied}")

        return self

    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate a batch of synthetic data.

        Args:
            n_samples: Number of samples
            conditions: Optional generation conditions

        Returns:
            Generated DataFrame
        """
        if not self._is_fitted:
            raise NotFittedError("StreamingGenerator")

        result = self._generator.generate(n_samples, conditions=conditions)
        self._stats.samples_generated += len(result)
        self._stats.batches_generated += 1

        return result

    def generate_stream(
        self,
        n_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Generate synthetic data as a stream of batches.

        Args:
            n_batches: Number of batches (None = infinite)
            batch_size: Size of each batch
            conditions: Optional generation conditions

        Yields:
            DataFrames of synthetic data
        """
        if not self._is_fitted:
            raise NotFittedError("StreamingGenerator")

        batch_size = batch_size or self.streaming_config.batch_size
        n_batches = n_batches or self.streaming_config.max_batches

        self._stats.start_time = time.time()
        batch_count = 0

        while n_batches is None or batch_count < n_batches:
            try:
                batch = self.generate(batch_size, conditions)
                yield batch
                batch_count += 1

                if self.streaming_config.delay_seconds > 0:
                    time.sleep(self.streaming_config.delay_seconds)

            except Exception as e:
                self._stats.errors.append(str(e))
                logger.error(f"Error in stream generation: {e}")
                raise

    def generate_async(
        self,
        callback: Callable[[pd.DataFrame], None],
        n_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Generate data asynchronously in a background thread.

        Args:
            callback: Function to call with each batch
            n_batches: Number of batches to generate
            batch_size: Size of each batch
        """
        if not self._is_fitted:
            raise NotFittedError("StreamingGenerator")

        self._stop_event.clear()

        def worker() -> None:
            try:
                for batch in self.generate_stream(n_batches, batch_size):
                    if self._stop_event.is_set():
                        break
                    callback(batch)
            except Exception as e:
                logger.error(f"Async generation error: {e}")

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
        logger.info("Started async generation worker")

    def stop_async(self) -> None:
        """Stop async generation."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        logger.info("Stopped async generation")

    def generate_to_queue(
        self,
        output_queue: queue.Queue,
        n_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> threading.Thread:
        """Generate data and put batches into a queue.

        Args:
            output_queue: Queue to put batches into
            n_batches: Number of batches
            batch_size: Size of each batch

        Returns:
            Worker thread
        """

        def worker() -> None:
            for batch in self.generate_stream(n_batches, batch_size):
                if self._stop_event.is_set():
                    break
                output_queue.put(batch)
            output_queue.put(None)  # Signal completion

        self._stop_event.clear()
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread

    @property
    def stats(self) -> StreamingStats:
        """Get streaming statistics."""
        return self._stats

    @property
    def is_fitted(self) -> bool:
        """Check if generator is fitted."""
        return self._is_fitted

    def reset(self) -> "StreamingGenerator":
        """Reset the generator state while keeping configuration.

        Clears fitted model, buffers, and statistics.

        Returns:
            Self for method chaining
        """
        self.stop()
        self._generator = None
        self._is_fitted = False
        self._stats = StreamingStats()
        self._buffer = queue.Queue(maxsize=self.streaming_config.buffer_size)
        self._data_buffer = []
        self._samples_since_update = 0
        self._discrete_columns = []
        logger.info("Streaming generator reset")
        return self


class DataStreamProcessor:
    """Process incoming data streams and generate synthetic data.

    Useful for real-time applications where data arrives continuously
    and synthetic data needs to be generated in response.

    Example:
        >>> processor = DataStreamProcessor(generator, window_size=1000)
        >>> for incoming in data_stream:
        ...     synthetic = processor.process(incoming)
    """

    # Default configuration constants
    DEFAULT_WINDOW_SIZE = 1000
    DEFAULT_UPDATE_THRESHOLD = 100

    def __init__(
        self,
        generator: StreamingGenerator,
        window_size: int = DEFAULT_WINDOW_SIZE,
        update_threshold: int = DEFAULT_UPDATE_THRESHOLD,
    ) -> None:
        """Initialize the processor.

        Args:
            generator: StreamingGenerator to use
            window_size: Size of sliding window for model updates
            update_threshold: Number of new samples before triggering update
        """
        self.generator = generator
        self.window_size = window_size
        self.update_threshold = update_threshold

        self._input_buffer: List[pd.DataFrame] = []
        self._input_count = 0

    def process(
        self,
        incoming_data: pd.DataFrame,
        generate_ratio: float = 1.0,
    ) -> pd.DataFrame:
        """Process incoming data and generate synthetic data.

        Args:
            incoming_data: New data that arrived
            generate_ratio: Ratio of synthetic to real data to generate

        Returns:
            Generated synthetic data
        """
        # Add to buffer
        self._input_buffer.append(incoming_data)
        self._input_count += len(incoming_data)

        # Check if we should update the model
        if self._input_count >= self.update_threshold:
            combined = pd.concat(self._input_buffer, ignore_index=True)

            # Keep only recent data within window
            if len(combined) > self.window_size:
                combined = combined.tail(self.window_size)

            self.generator.partial_fit(combined)

            # Reset buffer but keep some overlap
            self._input_buffer = [combined.tail(self.window_size // 2)]
            self._input_count = len(self._input_buffer[0])

        # Generate synthetic data
        n_generate = int(len(incoming_data) * generate_ratio)
        if n_generate > 0:
            return self.generator.generate(n_generate)

        return pd.DataFrame()

    def reset(self) -> None:
        """Reset the processor state."""
        self._input_buffer.clear()
        self._input_count = 0


class BatchIterator:
    """Iterator for efficient batch generation.

    Provides a clean iterator interface for generating
    a fixed number of samples in batches.

    Example:
        >>> for batch in BatchIterator(generator, total_samples=10000, batch_size=100):
        ...     process(batch)
    """

    def __init__(
        self,
        generator: StreamingGenerator,
        total_samples: int,
        batch_size: int = 1000,
    ) -> None:
        """Initialize the iterator.

        Args:
            generator: Generator to use
            total_samples: Total samples to generate
            batch_size: Size of each batch
        """
        self.generator = generator
        self.total_samples = total_samples
        self.batch_size = batch_size
        self._generated = 0

    def __iter__(self) -> Iterator[pd.DataFrame]:
        return self

    def __next__(self) -> pd.DataFrame:
        if self._generated >= self.total_samples:
            raise StopIteration

        remaining = self.total_samples - self._generated
        n = min(self.batch_size, remaining)

        batch = self.generator.generate(n)
        self._generated += len(batch)

        return batch

    def __len__(self) -> int:
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def generate_streaming(
    data: pd.DataFrame,
    n_samples: int,
    batch_size: int = 1000,
    method: str = "gaussian_copula",
    discrete_columns: Optional[List[str]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Convenience function for streaming generation.

    Args:
        data: Training data
        n_samples: Total samples to generate
        batch_size: Size of each batch
        method: Generator method
        discrete_columns: Categorical columns

    Yields:
        DataFrames of synthetic data
    """
    generator = StreamingGenerator(method=method)
    generator.fit(data, discrete_columns=discrete_columns)

    n_batches = (n_samples + batch_size - 1) // batch_size
    remaining = n_samples

    for batch in generator.generate_stream(n_batches=n_batches, batch_size=batch_size):
        if len(batch) > remaining:
            yield batch.head(remaining)
            break
        remaining -= len(batch)
        yield batch
