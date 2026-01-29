"""Streaming and incremental synthetic data generation.

This module provides capabilities for:
- Streaming generation (generate data continuously)
- Incremental model updates (update models without full retraining)
- Online learning for synthetic data

Example:
    >>> from genesis.streaming import StreamingGenerator
    >>>
    >>> # Create streaming generator
    >>> stream = StreamingGenerator(method='gaussian_copula')
    >>> stream.fit(initial_data)
    >>>
    >>> # Generate data in batches
    >>> for batch in stream.generate_stream(n_batches=100, batch_size=100):
    ...     process(batch)
    >>>
    >>> # Update model incrementally
    >>> stream.partial_fit(new_data)
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional

import pandas as pd

from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.exceptions import NotFittedError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming generation."""

    batch_size: int = 100
    buffer_size: int = 10
    max_batches: Optional[int] = None
    delay_seconds: float = 0.0
    auto_update: bool = False
    update_frequency: int = 1000  # Update after N samples


@dataclass
class StreamingStats:
    """Statistics for streaming generation."""

    batches_generated: int = 0
    samples_generated: int = 0
    updates_applied: int = 0
    start_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)

    @property
    def samples_per_second(self) -> float:
        """Calculate samples per second."""
        if self.start_time is None or self.samples_generated == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.samples_generated / elapsed if elapsed > 0 else 0.0


class StreamingGenerator:
    """Generator that supports streaming and incremental updates.

    This generator is optimized for:
    - Continuous data generation
    - Incremental model updates
    - Memory-efficient batch generation
    """

    def __init__(
        self,
        method: str = "gaussian_copula",
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        streaming_config: Optional[StreamingConfig] = None,
    ) -> None:
        """Initialize the streaming generator.

        Args:
            method: Generator method ('gaussian_copula' recommended for streaming)
            config: Generator configuration
            privacy: Privacy configuration
            streaming_config: Streaming-specific configuration
        """
        self.method = method
        self.config = config or GeneratorConfig()
        self.privacy = privacy or PrivacyConfig()
        self.streaming_config = streaming_config or StreamingConfig()

        self._generator: Optional[Any] = None
        self._is_fitted = False
        self._stats = StreamingStats()
        self._buffer: queue.Queue = queue.Queue(maxsize=self.streaming_config.buffer_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # For incremental updates
        self._data_buffer: List[pd.DataFrame] = []
        self._samples_since_update = 0

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
        weight: float = 0.1,
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

        # For now, re-fit with combined data (true incremental would be method-specific)
        # This is a simplified approach; a production implementation would
        # update statistics incrementally for Gaussian Copula

        self._data_buffer.append(new_data)

        # Trigger update if we have enough data
        if len(self._data_buffer) >= 5:
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

        def worker():
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

        def worker():
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


class DataStreamProcessor:
    """Process incoming data streams and generate synthetic data.

    Useful for real-time applications where data arrives continuously
    and synthetic data needs to be generated in response.
    """

    def __init__(
        self,
        generator: StreamingGenerator,
        window_size: int = 1000,
        update_threshold: int = 100,
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


class BatchIterator:
    """Iterator for efficient batch generation."""

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


class KafkaStreamingGenerator:
    """Streaming generator that integrates with Apache Kafka.

    Consumes data from Kafka topics and produces synthetic data to output topics.

    Example:
        >>> from genesis.streaming import KafkaStreamingGenerator
        >>>
        >>> generator = KafkaStreamingGenerator(
        ...     bootstrap_servers="localhost:9092",
        ...     input_topic="real-data",
        ...     output_topic="synthetic-data",
        ... )
        >>> generator.fit_from_topic(timeout_seconds=60)
        >>> generator.start_streaming()
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: Optional[str] = None,
        output_topic: Optional[str] = None,
        consumer_group: str = "genesis-consumer",
        method: str = "gaussian_copula",
        config: Optional[GeneratorConfig] = None,
    ) -> None:
        """Initialize Kafka streaming generator.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            input_topic: Topic to consume real data from
            output_topic: Topic to produce synthetic data to
            consumer_group: Kafka consumer group ID
            method: Generator method
            config: Generator configuration
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer_group = consumer_group
        self.method = method
        self.config = config or GeneratorConfig()

        self._consumer = None
        self._producer = None
        self._generator: Optional[StreamingGenerator] = None
        self._is_running = False
        self._stop_event = threading.Event()

    def _check_kafka(self) -> None:
        """Check if kafka-python is available."""
        try:
            from kafka import KafkaConsumer, KafkaProducer  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "kafka-python is required for Kafka streaming. "
                "Install with: pip install kafka-python"
            ) from e

    def _init_kafka(self) -> None:
        """Initialize Kafka consumer and producer."""
        self._check_kafka()
        import json

        from kafka import KafkaConsumer, KafkaProducer

        if self.input_topic:
            self._consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="earliest",
            )

        if self.output_topic:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            )

    def fit_from_topic(
        self,
        timeout_seconds: int = 60,
        max_records: int = 10000,
        discrete_columns: Optional[List[str]] = None,
    ) -> "KafkaStreamingGenerator":
        """Fit generator from data consumed from Kafka topic.

        Args:
            timeout_seconds: How long to consume before fitting
            max_records: Maximum records to consume
            discrete_columns: Categorical columns

        Returns:
            Self for method chaining
        """
        self._init_kafka()

        if self._consumer is None:
            raise ValueError("No input topic configured")

        records = []
        start_time = time.time()

        logger.info(f"Consuming from {self.input_topic} for {timeout_seconds}s...")

        while len(records) < max_records and time.time() - start_time < timeout_seconds:
            messages = self._consumer.poll(timeout_ms=1000)
            for _tp, msgs in messages.items():
                for msg in msgs:
                    records.append(msg.value)
                    if len(records) >= max_records:
                        break

        if not records:
            raise ValueError("No records consumed from Kafka")

        df = pd.DataFrame(records)
        logger.info(f"Consumed {len(df)} records, fitting generator...")

        self._generator = StreamingGenerator(method=self.method)
        self._generator.fit(df, discrete_columns=discrete_columns)

        return self

    def produce_batch(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Produce a batch of synthetic data to the output topic.

        Args:
            n_samples: Number of samples to generate
            conditions: Optional generation conditions

        Returns:
            Number of records produced
        """
        if self._generator is None:
            raise ValueError("Generator not fitted. Call fit_from_topic() first.")

        if self._producer is None:
            self._init_kafka()

        synthetic = self._generator.generate(n_samples, conditions=conditions)

        for record in synthetic.to_dict(orient="records"):
            self._producer.send(self.output_topic, value=record)

        self._producer.flush()
        logger.info(f"Produced {len(synthetic)} records to {self.output_topic}")

        return len(synthetic)

    def start_streaming(
        self,
        generate_ratio: float = 1.0,
        batch_process_size: int = 100,
    ) -> None:
        """Start continuous streaming pipeline.

        Consumes real data, updates model incrementally, and produces synthetic data.

        Args:
            generate_ratio: Ratio of synthetic to real records to produce
            batch_process_size: Records to accumulate before processing
        """
        if self._generator is None:
            raise ValueError("Generator not fitted. Call fit_from_topic() first.")

        self._init_kafka()
        self._is_running = True
        self._stop_event.clear()

        buffer = []

        logger.info("Starting streaming pipeline...")

        while not self._stop_event.is_set():
            messages = self._consumer.poll(timeout_ms=1000)

            for _tp, msgs in messages.items():
                for msg in msgs:
                    buffer.append(msg.value)

            if len(buffer) >= batch_process_size:
                # Process batch
                df = pd.DataFrame(buffer)

                # Update model incrementally
                self._generator.partial_fit(df)

                # Generate and produce synthetic data
                n_synthetic = int(len(df) * generate_ratio)
                if n_synthetic > 0:
                    synthetic = self._generator.generate(n_synthetic)

                    for record in synthetic.to_dict(orient="records"):
                        self._producer.send(self.output_topic, value=record)

                    self._producer.flush()

                buffer = []
                logger.debug(f"Processed batch: {len(df)} real → {n_synthetic} synthetic")

        self._is_running = False
        logger.info("Streaming pipeline stopped")

    def stop_streaming(self) -> None:
        """Stop the streaming pipeline."""
        self._stop_event.set()

    def close(self) -> None:
        """Close Kafka connections."""
        if self._consumer:
            self._consumer.close()
        if self._producer:
            self._producer.close()
        logger.info("Kafka connections closed")

    @property
    def is_running(self) -> bool:
        """Check if streaming is running."""
        return self._is_running


class WebSocketStreamingGenerator:
    """Streaming generator over WebSocket for real-time applications.

    Provides a WebSocket interface for streaming synthetic data generation.
    """

    def __init__(
        self,
        generator: StreamingGenerator,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        """Initialize WebSocket streaming generator.

        Args:
            generator: Fitted streaming generator
            host: Host to bind to
            port: Port to bind to
        """
        self.generator = generator
        self.host = host
        self.port = port
        self._server = None

    async def handler(self, websocket, path: str) -> None:
        """Handle WebSocket connection."""
        import json

        logger.info(f"New WebSocket connection from {websocket.remote_address}")

        try:
            async for message in websocket:
                request = json.loads(message)

                action = request.get("action", "generate")

                if action == "generate":
                    n_samples = request.get("n_samples", 100)
                    conditions = request.get("conditions")

                    synthetic = self.generator.generate(n_samples, conditions=conditions)

                    response = {
                        "action": "data",
                        "data": synthetic.to_dict(orient="records"),
                        "n_samples": len(synthetic),
                    }
                    await websocket.send(json.dumps(response))

                elif action == "stream":
                    n_batches = request.get("n_batches", 10)
                    batch_size = request.get("batch_size", 100)

                    for batch in self.generator.generate_stream(n_batches, batch_size):
                        response = {
                            "action": "batch",
                            "data": batch.to_dict(orient="records"),
                            "n_samples": len(batch),
                        }
                        await websocket.send(json.dumps(response))

                    await websocket.send(json.dumps({"action": "complete"}))

                elif action == "stats":
                    stats = self.generator.stats
                    response = {
                        "action": "stats",
                        "batches_generated": stats.batches_generated,
                        "samples_generated": stats.samples_generated,
                        "samples_per_second": stats.samples_per_second,
                    }
                    await websocket.send(json.dumps(response))

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send(json.dumps({"error": str(e)}))

    def run(self) -> None:
        """Run the WebSocket server."""
        try:
            import asyncio

            import websockets
        except ImportError as e:
            raise ImportError("websockets is required. Install with: pip install websockets") from e

        async def serve():
            async with websockets.serve(self.handler, self.host, self.port):
                logger.info(f"WebSocket server running at ws://{self.host}:{self.port}")
                await asyncio.Future()  # Run forever

        asyncio.run(serve())
