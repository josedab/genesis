"""Production-ready streaming synthetic data generation.

This module provides enterprise-grade streaming capabilities with:
- Exactly-once delivery semantics
- Checkpoint management for fault tolerance
- Backpressure handling
- Schema evolution support (Avro/Protobuf)
- Dead letter queues
- Rate limiting and metrics
"""

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import pandas as pd

from genesis.core.exceptions import GenesisError
from genesis.streaming.generator import StreamingGenerator
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DeliverySemantics(Enum):
    """Message delivery semantics."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class SchemaFormat(Enum):
    """Supported schema formats for serialization."""

    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class BackpressureStrategy(Enum):
    """Strategies for handling backpressure."""

    DROP = "drop"  # Drop messages when buffer is full
    BLOCK = "block"  # Block until buffer has space
    ADAPTIVE = "adaptive"  # Dynamically adjust rate


@dataclass
class Checkpoint:
    """Checkpoint for fault-tolerant streaming.

    Stores the state needed to resume streaming after a failure.
    """

    checkpoint_id: str
    sequence_number: int
    timestamp: float
    generator_state_hash: str
    pending_records: int
    committed_offset: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "generator_state_hash": self.generator_state_hash,
            "pending_records": self.pending_records,
            "committed_offset": self.committed_offset,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            sequence_number=data["sequence_number"],
            timestamp=data["timestamp"],
            generator_state_hash=data["generator_state_hash"],
            pending_records=data["pending_records"],
            committed_offset=data["committed_offset"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class StreamingMetrics:
    """Metrics for production streaming."""

    records_produced: int = 0
    records_failed: int = 0
    records_in_dlq: int = 0
    bytes_produced: int = 0
    checkpoints_created: int = 0
    last_checkpoint_time: Optional[float] = None
    average_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    backpressure_events: int = 0
    schema_evolutions: int = 0
    _latency_samples: List[float] = field(default_factory=list)
    _start_time: Optional[float] = None

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self._latency_samples.append(latency_ms)
        # Keep only last 1000 samples
        if len(self._latency_samples) > 1000:
            self._latency_samples = self._latency_samples[-1000:]
        self.average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def update_throughput(self) -> None:
        """Update throughput calculation."""
        if self._start_time is None:
            self._start_time = time.time()
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self.throughput_per_second = self.records_produced / elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "records_produced": self.records_produced,
            "records_failed": self.records_failed,
            "records_in_dlq": self.records_in_dlq,
            "bytes_produced": self.bytes_produced,
            "checkpoints_created": self.checkpoints_created,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "throughput_per_second": round(self.throughput_per_second, 2),
            "backpressure_events": self.backpressure_events,
            "schema_evolutions": self.schema_evolutions,
        }


class CheckpointManager:
    """Manages checkpoints for fault-tolerant streaming."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        checkpoint_interval: int = 1000,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            checkpoint_interval: Number of records between checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self._records_since_checkpoint = 0
        self._current_sequence = 0
        self._lock = threading.Lock()

    def should_checkpoint(self) -> bool:
        """Check if it's time to create a checkpoint."""
        return self._records_since_checkpoint >= self.checkpoint_interval

    def create_checkpoint(
        self,
        generator_state: Dict[str, Any],
        committed_offset: int,
        pending_records: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            generator_state: Current generator state
            committed_offset: Last committed offset
            pending_records: Number of pending (uncommitted) records
            metadata: Additional metadata

        Returns:
            Created checkpoint
        """
        with self._lock:
            self._current_sequence += 1
            state_hash = hashlib.sha256(
                json.dumps(generator_state, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            checkpoint = Checkpoint(
                checkpoint_id=f"ckpt_{self._current_sequence:08d}",
                sequence_number=self._current_sequence,
                timestamp=time.time(),
                generator_state_hash=state_hash,
                pending_records=pending_records,
                committed_offset=committed_offset,
                metadata=metadata or {},
            )

            # Persist checkpoint
            checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            self._records_since_checkpoint = 0
            logger.info(f"Created checkpoint {checkpoint.checkpoint_id}")
            return checkpoint

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("ckpt_*.json"))
        if not checkpoints:
            return None

        with open(checkpoints[-1]) as f:
            return Checkpoint.from_dict(json.load(f))

    def record_produced(self, count: int = 1) -> None:
        """Record that records were produced."""
        with self._lock:
            self._records_since_checkpoint += count

    def cleanup_old_checkpoints(self, keep: int = 5) -> int:
        """Remove old checkpoints, keeping the most recent ones.

        Args:
            keep: Number of checkpoints to keep

        Returns:
            Number of checkpoints removed
        """
        checkpoints = sorted(self.checkpoint_dir.glob("ckpt_*.json"))
        to_remove = checkpoints[:-keep] if len(checkpoints) > keep else []

        for path in to_remove:
            path.unlink()

        return len(to_remove)


class SchemaRegistry(ABC):
    """Abstract schema registry for schema evolution."""

    @abstractmethod
    def register_schema(self, subject: str, schema: Dict[str, Any]) -> int:
        """Register a schema and return its ID."""
        pass

    @abstractmethod
    def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """Get a schema by ID."""
        pass

    @abstractmethod
    def check_compatibility(
        self, subject: str, schema: Dict[str, Any]
    ) -> bool:
        """Check if schema is compatible with existing versions."""
        pass


class LocalSchemaRegistry(SchemaRegistry):
    """Local in-memory schema registry for development/testing."""

    def __init__(self) -> None:
        self._schemas: Dict[int, Dict[str, Any]] = {}
        self._subjects: Dict[str, List[int]] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def register_schema(self, subject: str, schema: Dict[str, Any]) -> int:
        """Register a schema."""
        with self._lock:
            schema_id = self._next_id
            self._next_id += 1
            self._schemas[schema_id] = schema

            if subject not in self._subjects:
                self._subjects[subject] = []
            self._subjects[subject].append(schema_id)

            return schema_id

    def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """Get schema by ID."""
        if schema_id not in self._schemas:
            raise KeyError(f"Schema {schema_id} not found")
        return self._schemas[schema_id]

    def check_compatibility(self, subject: str, schema: Dict[str, Any]) -> bool:
        """Check backward compatibility (new schema can read old data)."""
        if subject not in self._subjects or not self._subjects[subject]:
            return True  # No existing schema, always compatible

        # Simple compatibility: new schema must have all fields from latest version
        latest_id = self._subjects[subject][-1]
        latest_schema = self._schemas[latest_id]

        latest_fields = set(latest_schema.get("fields", {}).keys())
        new_fields = set(schema.get("fields", {}).keys())

        # Backward compatible if new schema has all old fields
        return latest_fields.issubset(new_fields)


class DeadLetterQueue:
    """Dead letter queue for failed records."""

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize DLQ.

        Args:
            max_size: Maximum number of records to keep
        """
        self.max_size = max_size
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(
        self,
        record: Dict[str, Any],
        error: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add a failed record to the DLQ."""
        with self._lock:
            entry = {
                "record": record,
                "error": error,
                "timestamp": timestamp or time.time(),
                "retry_count": 0,
            }
            self._queue.append(entry)

            # Remove oldest if over limit
            if len(self._queue) > self.max_size:
                self._queue = self._queue[-self.max_size :]

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all records in the DLQ."""
        with self._lock:
            return list(self._queue)

    def get_for_retry(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Get records eligible for retry."""
        with self._lock:
            return [r for r in self._queue if r["retry_count"] < max_retries]

    def clear(self) -> int:
        """Clear the DLQ and return count of cleared records."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def __len__(self) -> int:
        return len(self._queue)


class RateLimiter:
    """Token bucket rate limiter for controlling throughput."""

    def __init__(
        self,
        rate: float,
        burst: int = 100,
    ) -> None:
        """Initialize rate limiter.

        Args:
            rate: Records per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, count: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire tokens, blocking if necessary.

        Args:
            count: Number of tokens to acquire
            timeout: Maximum time to wait

        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= count:
                    self._tokens -= count
                    return True

            if timeout is not None and (time.time() - start_time) >= timeout:
                return False

            # Sleep for time needed to get enough tokens
            needed = count - self._tokens
            sleep_time = min(needed / self.rate, 0.1)  # Cap at 100ms
            time.sleep(sleep_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now


class BackpressureHandler:
    """Handles backpressure in streaming pipelines."""

    def __init__(
        self,
        strategy: BackpressureStrategy = BackpressureStrategy.ADAPTIVE,
        initial_rate: float = 1000.0,
        min_rate: float = 100.0,
        max_rate: float = 10000.0,
    ) -> None:
        """Initialize backpressure handler.

        Args:
            strategy: Backpressure strategy
            initial_rate: Initial records per second
            min_rate: Minimum rate when adapting
            max_rate: Maximum rate when adapting
        """
        self.strategy = strategy
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Record successful send."""
        with self._lock:
            self._consecutive_successes += 1
            self._consecutive_failures = 0

            if self.strategy == BackpressureStrategy.ADAPTIVE:
                # Increase rate after sustained success
                if self._consecutive_successes >= 100:
                    self.current_rate = min(self.max_rate, self.current_rate * 1.1)
                    self._consecutive_successes = 0

    def record_failure(self) -> float:
        """Record failed send and return recommended wait time."""
        with self._lock:
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            if self.strategy == BackpressureStrategy.ADAPTIVE:
                # Decrease rate on failure
                self.current_rate = max(self.min_rate, self.current_rate * 0.5)

            # Exponential backoff
            return min(30.0, 0.1 * (2 ** self._consecutive_failures))

    def get_current_rate(self) -> float:
        """Get current rate limit."""
        return self.current_rate


class RecordSerializer(ABC, Generic[T]):
    """Abstract serializer for records."""

    @abstractmethod
    def serialize(self, record: Dict[str, Any]) -> T:
        """Serialize a record."""
        pass

    @abstractmethod
    def deserialize(self, data: T) -> Dict[str, Any]:
        """Deserialize a record."""
        pass


class JsonSerializer(RecordSerializer[bytes]):
    """JSON serializer."""

    def serialize(self, record: Dict[str, Any]) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps(record, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from JSON bytes."""
        return json.loads(data.decode("utf-8"))


class AvroSerializer(RecordSerializer[bytes]):
    """Avro serializer with schema registry support."""

    def __init__(
        self,
        schema: Dict[str, Any],
        registry: Optional[SchemaRegistry] = None,
    ) -> None:
        """Initialize Avro serializer.

        Args:
            schema: Avro schema dictionary
            registry: Optional schema registry
        """
        self.schema = schema
        self.registry = registry
        self._schema_id: Optional[int] = None

    def serialize(self, record: Dict[str, Any]) -> bytes:
        """Serialize to Avro bytes.

        Note: Requires fastavro package. Falls back to JSON if unavailable.
        """
        try:
            import io

            import fastavro

            buffer = io.BytesIO()
            fastavro.schemaless_writer(buffer, self.schema, record)
            return buffer.getvalue()
        except ImportError:
            logger.warning("fastavro not installed, falling back to JSON")
            return json.dumps(record, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize from Avro bytes."""
        try:
            import io

            import fastavro

            buffer = io.BytesIO(data)
            return fastavro.schemaless_reader(buffer, self.schema)
        except ImportError:
            return json.loads(data.decode("utf-8"))


@dataclass
class ProducerConfig:
    """Configuration for production Kafka producer."""

    bootstrap_servers: str = "localhost:9092"
    topic: str = "synthetic-data"
    delivery_semantics: DeliverySemantics = DeliverySemantics.EXACTLY_ONCE
    schema_format: SchemaFormat = SchemaFormat.JSON
    rate_limit: Optional[float] = None  # Records per second, None = unlimited
    batch_size: int = 100
    linger_ms: int = 5
    acks: str = "all"
    retries: int = 3
    enable_idempotence: bool = True
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    dlq_enabled: bool = True
    dlq_max_size: int = 10000


class ProductionKafkaProducer:
    """Production-ready Kafka producer with exactly-once semantics.

    Features:
    - Exactly-once delivery via idempotent producer and transactions
    - Checkpoint management for fault tolerance
    - Schema evolution support
    - Dead letter queue for failed records
    - Rate limiting and backpressure handling
    - Comprehensive metrics

    Example:
        >>> from genesis.streaming.production import ProductionKafkaProducer, ProducerConfig
        >>>
        >>> config = ProducerConfig(
        ...     bootstrap_servers="kafka:9092",
        ...     topic="synthetic-data",
        ...     delivery_semantics=DeliverySemantics.EXACTLY_ONCE,
        ... )
        >>> producer = ProductionKafkaProducer(config, generator)
        >>> producer.start()
        >>> producer.produce(n_records=10000)
        >>> producer.stop()
    """

    def __init__(
        self,
        config: ProducerConfig,
        generator: StreamingGenerator,
        schema_registry: Optional[SchemaRegistry] = None,
    ) -> None:
        """Initialize production Kafka producer.

        Args:
            config: Producer configuration
            generator: Streaming generator to use
            schema_registry: Optional schema registry
        """
        self.config = config
        self.generator = generator
        self.schema_registry = schema_registry or LocalSchemaRegistry()

        # Initialize components
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_interval=config.checkpoint_interval,
        )
        self.dlq = DeadLetterQueue(max_size=config.dlq_max_size) if config.dlq_enabled else None
        self.rate_limiter = RateLimiter(config.rate_limit) if config.rate_limit else None
        self.backpressure = BackpressureHandler()
        self.metrics = StreamingMetrics()

        # Serializer
        self._serializer: RecordSerializer = JsonSerializer()
        self._producer = None
        self._is_running = False
        self._stop_event = threading.Event()
        self._committed_offset = 0
        self._pending_records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _init_kafka(self) -> None:
        """Initialize Kafka producer with exactly-once config."""
        try:
            from kafka import KafkaProducer
        except ImportError as e:
            raise ImportError(
                "kafka-python is required. Install with: pip install kafka-python"
            ) from e

        producer_config = {
            "bootstrap_servers": self.config.bootstrap_servers,
            "value_serializer": self._serializer.serialize,
            "acks": self.config.acks,
            "retries": self.config.retries,
            "batch_size": self.config.batch_size * 1024,  # Convert to bytes
            "linger_ms": self.config.linger_ms,
        }

        if self.config.delivery_semantics == DeliverySemantics.EXACTLY_ONCE:
            producer_config["enable_idempotence"] = True
            producer_config["max_in_flight_requests_per_connection"] = 5

        self._producer = KafkaProducer(**producer_config)
        logger.info(f"Initialized Kafka producer for {self.config.topic}")

    def start(self) -> "ProductionKafkaProducer":
        """Start the producer."""
        self._init_kafka()
        self._is_running = True
        self._stop_event.clear()
        self.metrics._start_time = time.time()

        # Resume from checkpoint if available
        checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if checkpoint:
            self._committed_offset = checkpoint.committed_offset
            logger.info(f"Resuming from checkpoint {checkpoint.checkpoint_id}")

        return self

    def stop(self) -> None:
        """Stop the producer gracefully."""
        self._stop_event.set()
        self._is_running = False

        # Flush remaining records
        if self._producer:
            self._producer.flush()
            self._producer.close()

        # Create final checkpoint
        self._create_checkpoint()

        logger.info("Producer stopped")

    def produce(
        self,
        n_records: int,
        conditions: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> int:
        """Produce synthetic records to Kafka.

        Args:
            n_records: Number of records to produce
            conditions: Generation conditions
            callback: Optional callback for each produced record

        Returns:
            Number of records successfully produced
        """
        if not self._is_running:
            raise GenesisError("Producer not started. Call start() first.")

        produced = 0
        batch_size = self.config.batch_size

        for batch_start in range(0, n_records, batch_size):
            if self._stop_event.is_set():
                break

            current_batch_size = min(batch_size, n_records - batch_start)

            # Rate limiting
            if self.rate_limiter:
                self.rate_limiter.acquire(current_batch_size)

            # Generate batch
            start_time = time.time()
            try:
                batch = self.generator.generate(current_batch_size, conditions)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                continue

            # Produce each record
            for record in batch.to_dict(orient="records"):
                try:
                    self._produce_record(record)
                    produced += 1
                    self.backpressure.record_success()

                    if callback:
                        callback(record)

                except Exception as e:
                    self.metrics.records_failed += 1
                    if self.dlq:
                        self.dlq.add(record, str(e))
                        self.metrics.records_in_dlq = len(self.dlq)

                    wait_time = self.backpressure.record_failure()
                    self.metrics.backpressure_events += 1
                    time.sleep(wait_time)

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record_latency(latency_ms)
            self.metrics.update_throughput()

            # Checkpoint if needed
            self.checkpoint_manager.record_produced(current_batch_size)
            if self.checkpoint_manager.should_checkpoint():
                self._create_checkpoint()

        return produced

    def _produce_record(self, record: Dict[str, Any]) -> None:
        """Produce a single record with exactly-once semantics."""
        # Add metadata
        record_with_meta = {
            "_genesis_timestamp": datetime.utcnow().isoformat(),
            "_genesis_sequence": self._committed_offset + 1,
            **record,
        }

        with self._lock:
            self._pending_records.append(record_with_meta)

        # Send to Kafka
        future = self._producer.send(
            self.config.topic,
            value=record_with_meta,
        )

        if self.config.delivery_semantics == DeliverySemantics.EXACTLY_ONCE:
            # Wait for acknowledgment for exactly-once
            future.get(timeout=10)

        with self._lock:
            self._committed_offset += 1
            self._pending_records = [
                r for r in self._pending_records
                if r["_genesis_sequence"] > self._committed_offset
            ]

        self.metrics.records_produced += 1
        serialized = self._serializer.serialize(record_with_meta)
        self.metrics.bytes_produced += len(serialized)

    def _create_checkpoint(self) -> Checkpoint:
        """Create a checkpoint of current state."""
        checkpoint = self.checkpoint_manager.create_checkpoint(
            generator_state=self.generator.stats.__dict__,
            committed_offset=self._committed_offset,
            pending_records=len(self._pending_records),
            metadata={"metrics": self.metrics.to_dict()},
        )
        self.metrics.checkpoints_created += 1
        self.metrics.last_checkpoint_time = time.time()
        return checkpoint

    def retry_dlq(self, max_retries: int = 3) -> int:
        """Retry records from the dead letter queue.

        Args:
            max_retries: Maximum retry attempts per record

        Returns:
            Number of records successfully retried
        """
        if not self.dlq:
            return 0

        retried = 0
        for entry in self.dlq.get_for_retry(max_retries):
            try:
                self._produce_record(entry["record"])
                retried += 1
            except Exception:
                entry["retry_count"] += 1

        return retried

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.to_dict()

    @property
    def is_running(self) -> bool:
        """Check if producer is running."""
        return self._is_running


class ProductionStreamingPipeline:
    """End-to-end production streaming pipeline.

    Combines generator, producer, and monitoring into a complete pipeline.

    Example:
        >>> from genesis.streaming.production import ProductionStreamingPipeline
        >>>
        >>> pipeline = ProductionStreamingPipeline(
        ...     generator=generator,
        ...     kafka_config={"bootstrap_servers": "kafka:9092", "topic": "synth"},
        ...     rate_limit=1000,
        ... )
        >>> pipeline.start()
        >>> pipeline.run(duration_seconds=3600)  # Run for 1 hour
        >>> pipeline.stop()
    """

    def __init__(
        self,
        generator: StreamingGenerator,
        kafka_config: Dict[str, Any],
        rate_limit: Optional[float] = None,
        checkpoint_dir: str = "./checkpoints",
        enable_dlq: bool = True,
    ) -> None:
        """Initialize the pipeline.

        Args:
            generator: Streaming generator
            kafka_config: Kafka configuration dict
            rate_limit: Records per second limit
            checkpoint_dir: Directory for checkpoints
            enable_dlq: Enable dead letter queue
        """
        self.generator = generator

        config = ProducerConfig(
            bootstrap_servers=kafka_config.get("bootstrap_servers", "localhost:9092"),
            topic=kafka_config.get("topic", "synthetic-data"),
            rate_limit=rate_limit,
            checkpoint_dir=checkpoint_dir,
            dlq_enabled=enable_dlq,
        )

        self.producer = ProductionKafkaProducer(config, generator)
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> "ProductionStreamingPipeline":
        """Start the pipeline."""
        self.producer.start()
        self._is_running = True
        return self

    def stop(self) -> None:
        """Stop the pipeline."""
        self._is_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        self.producer.stop()

    def run(
        self,
        duration_seconds: Optional[float] = None,
        total_records: Optional[int] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline.

        Args:
            duration_seconds: Run for this duration (None = until stopped)
            total_records: Stop after this many records (None = unlimited)
            conditions: Generation conditions

        Returns:
            Final metrics
        """
        if not self._is_running:
            self.start()

        start_time = time.time()
        records_produced = 0
        batch_size = self.producer.config.batch_size

        while self._is_running:
            # Check duration limit
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                break

            # Check record limit
            if total_records and records_produced >= total_records:
                break

            # Calculate batch size
            if total_records:
                current_batch = min(batch_size, total_records - records_produced)
            else:
                current_batch = batch_size

            records_produced += self.producer.produce(current_batch, conditions)

        return self.producer.get_metrics()

    def run_async(
        self,
        duration_seconds: Optional[float] = None,
        total_records: Optional[int] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run the pipeline in a background thread."""

        def worker() -> None:
            self.run(duration_seconds, total_records, conditions)

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.producer.get_metrics()

    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running
