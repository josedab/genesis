"""Kafka integration for streaming synthetic data generation."""

import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.core.config import GeneratorConfig
from genesis.streaming.generator import StreamingGenerator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Default configuration constants
DEFAULT_POLL_TIMEOUT_MS = 1000
DEFAULT_MAX_RECORDS = 10000
DEFAULT_BATCH_PROCESS_SIZE = 100


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
        max_records: int = DEFAULT_MAX_RECORDS,
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

        records: List[Any] = []
        start_time = time.time()

        logger.info(f"Consuming from {self.input_topic} for {timeout_seconds}s...")

        while len(records) < max_records and time.time() - start_time < timeout_seconds:
            messages = self._consumer.poll(timeout_ms=DEFAULT_POLL_TIMEOUT_MS)
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
        batch_process_size: int = DEFAULT_BATCH_PROCESS_SIZE,
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

        buffer: List[Any] = []

        logger.info("Starting streaming pipeline...")

        while not self._stop_event.is_set():
            messages = self._consumer.poll(timeout_ms=DEFAULT_POLL_TIMEOUT_MS)

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
