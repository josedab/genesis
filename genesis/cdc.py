"""Database Change Data Capture (CDC) for Genesis.

This module provides real-time synthetic data generation from production
database changes using Debezium-style CDC patterns.

Example:
    >>> from genesis.cdc import CDCGenerator, CDCConfig
    >>>
    >>> # Configure CDC source
    >>> config = CDCConfig(
    ...     source_type="postgresql",
    ...     connection_string="postgresql://localhost/mydb",
    ...     tables=["users", "orders"],
    ... )
    >>>
    >>> # Start CDC-based synthetic generation
    >>> generator = CDCGenerator(config)
    >>> generator.start()
    >>>
    >>> # Get synthetic changes
    >>> for change in generator.stream_changes():
    ...     print(f"Synthetic {change.operation} on {change.table}")
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.base import SyntheticGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.exceptions import ConfigurationError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class CDCSourceType(Enum):
    """Supported CDC source types."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"
    DEBEZIUM = "debezium"  # Generic Debezium connector
    KAFKA = "kafka"  # Kafka topic with CDC events
    FILE = "file"  # File-based CDC (for testing)


class OperationType(Enum):
    """CDC operation types."""

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    SNAPSHOT = "SNAPSHOT"  # Initial snapshot
    TRUNCATE = "TRUNCATE"


@dataclass
class CDCConfig:
    """Configuration for CDC-based generation.

    Attributes:
        source_type: Type of CDC source
        connection_string: Database connection string
        tables: List of tables to track
        kafka_bootstrap_servers: Kafka servers (for Kafka source)
        kafka_topics: Kafka topics to consume
        debezium_connector_config: Debezium connector configuration
        batch_size: Number of changes to batch before synthesis
        latency_ms: Target latency in milliseconds
        enable_schema_evolution: Handle schema changes
        synthetic_delay_factor: Multiply original delays by this factor
    """

    source_type: CDCSourceType = CDCSourceType.POSTGRESQL
    connection_string: Optional[str] = None
    tables: List[str] = field(default_factory=list)
    kafka_bootstrap_servers: Optional[str] = None
    kafka_topics: List[str] = field(default_factory=list)
    debezium_connector_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 100
    latency_ms: int = 100
    enable_schema_evolution: bool = True
    synthetic_delay_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "connection_string": self.connection_string,
            "tables": self.tables,
            "kafka_bootstrap_servers": self.kafka_bootstrap_servers,
            "kafka_topics": self.kafka_topics,
            "batch_size": self.batch_size,
            "latency_ms": self.latency_ms,
            "enable_schema_evolution": self.enable_schema_evolution,
            "synthetic_delay_factor": self.synthetic_delay_factor,
        }


@dataclass
class CDCEvent:
    """A single CDC event.

    Attributes:
        event_id: Unique event identifier
        operation: Type of operation
        table: Table name
        schema: Schema/database name
        timestamp: Event timestamp
        before: Record state before change (for UPDATE/DELETE)
        after: Record state after change (for INSERT/UPDATE)
        key: Primary key values
        metadata: Additional metadata
    """

    event_id: str
    operation: OperationType
    table: str
    schema: str
    timestamp: datetime
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    key: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "operation": self.operation.value,
            "table": self.table,
            "schema": self.schema,
            "timestamp": self.timestamp.isoformat(),
            "before": self.before,
            "after": self.after,
            "key": self.key,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CDCEvent":
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            operation=OperationType(data["operation"]),
            table=data["table"],
            schema=data.get("schema", "public"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            before=data.get("before"),
            after=data.get("after"),
            key=data.get("key"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SyntheticCDCEvent:
    """A synthetic CDC event generated from a real event.

    Contains both the original event reference and the
    synthesized data.
    """

    original_event_id: str
    synthetic_event_id: str
    operation: OperationType
    table: str
    schema: str
    timestamp: datetime
    synthetic_before: Optional[Dict[str, Any]] = None
    synthetic_after: Optional[Dict[str, Any]] = None
    synthetic_key: Optional[Dict[str, Any]] = None
    lineage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_event_id": self.original_event_id,
            "synthetic_event_id": self.synthetic_event_id,
            "operation": self.operation.value,
            "table": self.table,
            "schema": self.schema,
            "timestamp": self.timestamp.isoformat(),
            "synthetic_before": self.synthetic_before,
            "synthetic_after": self.synthetic_after,
            "synthetic_key": self.synthetic_key,
            "lineage": self.lineage,
        }


class CDCSource(ABC):
    """Abstract base class for CDC sources."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the CDC source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the CDC source."""
        pass

    @abstractmethod
    def get_events(self, timeout: float = 1.0) -> List[CDCEvent]:
        """Get pending CDC events.

        Args:
            timeout: Timeout in seconds

        Returns:
            List of CDC events
        """
        pass

    @abstractmethod
    def get_table_schema(self, table: str) -> Dict[str, Any]:
        """Get schema for a table.

        Args:
            table: Table name

        Returns:
            Schema dictionary
        """
        pass


class FileCDCSource(CDCSource):
    """File-based CDC source for testing.

    Reads CDC events from JSON files.
    """

    def __init__(self, file_path: str):
        """Initialize file CDC source.

        Args:
            file_path: Path to JSON file with CDC events
        """
        self.file_path = file_path
        self._events: List[CDCEvent] = []
        self._current_index = 0
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> None:
        """Load events from file."""
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)

            if "events" in data:
                self._events = [CDCEvent.from_dict(e) for e in data["events"]]
            if "schemas" in data:
                self._schemas = data["schemas"]

            logger.info(f"Loaded {len(self._events)} CDC events from {self.file_path}")
        except FileNotFoundError:
            logger.warning(f"CDC file not found: {self.file_path}")
            self._events = []

    def disconnect(self) -> None:
        """Clear loaded events."""
        self._events = []
        self._current_index = 0

    def get_events(self, timeout: float = 1.0) -> List[CDCEvent]:
        """Get next batch of events."""
        if self._current_index >= len(self._events):
            return []

        # Return next event
        event = self._events[self._current_index]
        self._current_index += 1
        return [event]

    def get_table_schema(self, table: str) -> Dict[str, Any]:
        """Get table schema."""
        return self._schemas.get(table, {})


class KafkaCDCSource(CDCSource):
    """Kafka-based CDC source.

    Consumes CDC events from Kafka topics (Debezium format).
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topics: List[str],
        group_id: str = "genesis-cdc",
        auto_offset_reset: str = "earliest",
    ):
        """Initialize Kafka CDC source.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            topics: Topics to consume
            group_id: Consumer group ID
            auto_offset_reset: Auto offset reset policy
        """
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self._consumer = None
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> None:
        """Connect to Kafka."""
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise ImportError(
                "kafka-python is required for Kafka CDC. "
                "Install with: pip install kafka-python"
            )

        self._consumer = KafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        logger.info(f"Connected to Kafka: {self.bootstrap_servers}")

    def disconnect(self) -> None:
        """Disconnect from Kafka."""
        if self._consumer:
            self._consumer.close()
            self._consumer = None

    def get_events(self, timeout: float = 1.0) -> List[CDCEvent]:
        """Get events from Kafka."""
        if not self._consumer:
            return []

        events = []
        messages = self._consumer.poll(timeout_ms=int(timeout * 1000))

        for topic_partition, records in messages.items():
            for record in records:
                event = self._parse_debezium_event(record.value)
                if event:
                    events.append(event)

        return events

    def _parse_debezium_event(self, message: Dict[str, Any]) -> Optional[CDCEvent]:
        """Parse Debezium format message to CDCEvent."""
        try:
            payload = message.get("payload", message)

            # Determine operation
            op = payload.get("op", "c")
            operation_map = {
                "c": OperationType.INSERT,
                "u": OperationType.UPDATE,
                "d": OperationType.DELETE,
                "r": OperationType.SNAPSHOT,
                "t": OperationType.TRUNCATE,
            }
            operation = operation_map.get(op, OperationType.INSERT)

            # Extract source info
            source = payload.get("source", {})
            table = source.get("table", "unknown")
            schema = source.get("schema", "public")

            # Extract timestamp
            ts_ms = payload.get("ts_ms", int(time.time() * 1000))
            timestamp = datetime.fromtimestamp(ts_ms / 1000)

            return CDCEvent(
                event_id=str(uuid.uuid4()),
                operation=operation,
                table=table,
                schema=schema,
                timestamp=timestamp,
                before=payload.get("before"),
                after=payload.get("after"),
                key=message.get("key"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse Debezium event: {e}")
            return None

    def get_table_schema(self, table: str) -> Dict[str, Any]:
        """Get table schema (from Debezium schema registry if available)."""
        return self._schemas.get(table, {})


class DatabaseCDCSource(CDCSource):
    """Database polling-based CDC source.

    Uses timestamp or sequence columns to detect changes.
    Note: For production, use Debezium or native replication.
    """

    def __init__(
        self,
        connection_string: str,
        tables: List[str],
        timestamp_column: str = "updated_at",
        poll_interval: float = 1.0,
    ):
        """Initialize database CDC source.

        Args:
            connection_string: Database connection string
            tables: Tables to monitor
            timestamp_column: Column to use for change detection
            poll_interval: Polling interval in seconds
        """
        self.connection_string = connection_string
        self.tables = tables
        self.timestamp_column = timestamp_column
        self.poll_interval = poll_interval
        self._connection = None
        self._last_timestamps: Dict[str, datetime] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> None:
        """Connect to database."""
        try:
            from sqlalchemy import create_engine, inspect

            self._engine = create_engine(self.connection_string)
            self._connection = self._engine.connect()

            # Get schemas
            inspector = inspect(self._engine)
            for table in self.tables:
                columns = inspector.get_columns(table)
                self._schemas[table] = {
                    "columns": {col["name"]: str(col["type"]) for col in columns}
                }

            logger.info(f"Connected to database, monitoring {len(self.tables)} tables")
        except ImportError:
            raise ImportError(
                "SQLAlchemy is required for database CDC. "
                "Install with: pip install sqlalchemy"
            )

    def disconnect(self) -> None:
        """Disconnect from database."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def get_events(self, timeout: float = 1.0) -> List[CDCEvent]:
        """Poll for changes."""
        if not self._connection:
            return []

        from sqlalchemy import text

        events = []

        for table in self.tables:
            last_ts = self._last_timestamps.get(table, datetime.min)

            # Query for changes
            query = text(
                f"SELECT * FROM {table} WHERE {self.timestamp_column} > :last_ts "
                f"ORDER BY {self.timestamp_column}"
            )

            result = self._connection.execute(query, {"last_ts": last_ts})
            rows = result.fetchall()

            for row in rows:
                row_dict = dict(row._mapping)
                timestamp = row_dict.get(self.timestamp_column, datetime.now())

                events.append(
                    CDCEvent(
                        event_id=str(uuid.uuid4()),
                        operation=OperationType.UPDATE,  # Simplified
                        table=table,
                        schema="public",
                        timestamp=timestamp,
                        after=row_dict,
                    )
                )

                # Update last timestamp
                if timestamp > last_ts:
                    self._last_timestamps[table] = timestamp

        return events

    def get_table_schema(self, table: str) -> Dict[str, Any]:
        """Get table schema."""
        return self._schemas.get(table, {})


class CDCSynthesizer:
    """Synthesize CDC events into privacy-safe synthetic data.

    This class transforms real CDC events into synthetic events
    while preserving temporal patterns and relationships.
    """

    def __init__(
        self,
        privacy_config: Optional[PrivacyConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the synthesizer.

        Args:
            privacy_config: Privacy configuration
            seed: Random seed
        """
        self.privacy_config = privacy_config or PrivacyConfig()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Cache for table generators
        self._table_generators: Dict[str, SyntheticGenerator] = {}
        self._table_data: Dict[str, pd.DataFrame] = {}

        # Key mapping (original -> synthetic)
        self._key_mapping: Dict[str, Dict[str, str]] = {}

    def learn_table(
        self,
        table: str,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
    ) -> None:
        """Learn distribution for a table.

        Args:
            table: Table name
            data: Sample data from the table
            discrete_columns: Categorical columns
        """
        generator = SyntheticGenerator(
            method="auto",
            privacy=self.privacy_config,
        )
        generator.fit(data, discrete_columns=discrete_columns)

        self._table_generators[table] = generator
        self._table_data[table] = data
        self._key_mapping[table] = {}

        logger.info(f"Learned distribution for table: {table}")

    def synthesize_event(self, event: CDCEvent) -> SyntheticCDCEvent:
        """Synthesize a single CDC event.

        Args:
            event: Original CDC event

        Returns:
            Synthetic CDC event
        """
        table = event.table

        # Get or create synthetic key
        if event.key:
            original_key = json.dumps(event.key, sort_keys=True)
            if original_key not in self._key_mapping.get(table, {}):
                self._key_mapping.setdefault(table, {})[original_key] = str(uuid.uuid4())[:8]
            synthetic_key = {"id": self._key_mapping[table][original_key]}
        else:
            synthetic_key = None

        # Synthesize data
        synthetic_before = None
        synthetic_after = None

        if event.before and table in self._table_generators:
            synthetic_before = self._synthesize_record(table, event.before)

        if event.after and table in self._table_generators:
            synthetic_after = self._synthesize_record(table, event.after)

        return SyntheticCDCEvent(
            original_event_id=event.event_id,
            synthetic_event_id=str(uuid.uuid4()),
            operation=event.operation,
            table=table,
            schema=event.schema,
            timestamp=event.timestamp,
            synthetic_before=synthetic_before,
            synthetic_after=synthetic_after,
            synthetic_key=synthetic_key,
            lineage={
                "original_table": table,
                "original_timestamp": event.timestamp.isoformat(),
                "synthesis_method": "cdc_streaming",
            },
        )

    def _synthesize_record(
        self,
        table: str,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize a single record.

        Args:
            table: Table name
            record: Original record

        Returns:
            Synthetic record
        """
        generator = self._table_generators.get(table)

        if generator:
            # Generate one synthetic record
            synthetic_df = generator.generate(n_samples=1)
            return synthetic_df.iloc[0].to_dict()
        else:
            # Fallback: basic perturbation
            return self._perturb_record(record)

    def _perturb_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb a record when no generator is available."""
        perturbed = {}

        for key, value in record.items():
            if isinstance(value, (int, float)):
                # Add noise to numeric values
                noise = self._rng.normal(0, abs(value) * 0.1 + 1)
                perturbed[key] = type(value)(value + noise)
            elif isinstance(value, str):
                # Hash string values
                perturbed[key] = hashlib.md5(value.encode()).hexdigest()[:len(value)]
            else:
                perturbed[key] = value

        return perturbed


class CDCGenerator:
    """Main CDC-based synthetic data generator.

    Connects to a CDC source, learns table distributions,
    and generates synthetic CDC events in real-time.

    Example:
        >>> config = CDCConfig(
        ...     source_type=CDCSourceType.KAFKA,
        ...     kafka_bootstrap_servers="localhost:9092",
        ...     kafka_topics=["dbserver.public.users"],
        ... )
        >>> generator = CDCGenerator(config)
        >>> generator.start()
        >>>
        >>> for synthetic_event in generator.stream_changes():
        ...     process(synthetic_event)
    """

    def __init__(
        self,
        config: CDCConfig,
        privacy_config: Optional[PrivacyConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the CDC generator.

        Args:
            config: CDC configuration
            privacy_config: Privacy configuration
            seed: Random seed
        """
        self.config = config
        self.privacy_config = privacy_config or PrivacyConfig()
        self.seed = seed

        # Create CDC source
        self._source = self._create_source()

        # Create synthesizer
        self._synthesizer = CDCSynthesizer(
            privacy_config=self.privacy_config,
            seed=seed,
        )

        # State
        self._running = False
        self._event_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stats = {
            "events_received": 0,
            "events_synthesized": 0,
            "errors": 0,
        }

    def _create_source(self) -> CDCSource:
        """Create the appropriate CDC source."""
        if self.config.source_type == CDCSourceType.FILE:
            return FileCDCSource(self.config.connection_string or "cdc_events.json")
        elif self.config.source_type == CDCSourceType.KAFKA:
            return KafkaCDCSource(
                bootstrap_servers=self.config.kafka_bootstrap_servers or "localhost:9092",
                topics=self.config.kafka_topics,
            )
        elif self.config.source_type in (
            CDCSourceType.POSTGRESQL,
            CDCSourceType.MYSQL,
            CDCSourceType.SQLSERVER,
        ):
            if not self.config.connection_string:
                raise ConfigurationError("connection_string required for database CDC")
            return DatabaseCDCSource(
                connection_string=self.config.connection_string,
                tables=self.config.tables,
            )
        else:
            raise ConfigurationError(f"Unsupported CDC source type: {self.config.source_type}")

    def learn_tables(
        self,
        table_data: Dict[str, pd.DataFrame],
        discrete_columns: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Learn distributions for tables.

        Args:
            table_data: Dictionary of table name to sample DataFrame
            discrete_columns: Dictionary of table name to discrete columns
        """
        discrete_columns = discrete_columns or {}

        for table, data in table_data.items():
            self._synthesizer.learn_table(
                table,
                data,
                discrete_columns=discrete_columns.get(table),
            )

    def start(self) -> None:
        """Start the CDC generator."""
        if self._running:
            return

        self._source.connect()
        self._running = True

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        logger.info("CDC generator started")

    def stop(self) -> None:
        """Stop the CDC generator."""
        self._running = False

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None

        self._source.disconnect()
        logger.info("CDC generator stopped")

    def _worker_loop(self) -> None:
        """Worker loop for processing CDC events."""
        while self._running:
            try:
                # Get events from source
                events = self._source.get_events(timeout=self.config.latency_ms / 1000)

                for event in events:
                    self._stats["events_received"] += 1

                    # Synthesize event
                    synthetic_event = self._synthesizer.synthesize_event(event)
                    self._stats["events_synthesized"] += 1

                    # Add to queue
                    self._event_queue.put(synthetic_event)

            except Exception as e:
                logger.error(f"Error in CDC worker: {e}")
                self._stats["errors"] += 1
                time.sleep(0.1)

    def stream_changes(
        self,
        timeout: float = 1.0,
    ) -> Generator[SyntheticCDCEvent, None, None]:
        """Stream synthetic CDC events.

        Args:
            timeout: Timeout for getting events

        Yields:
            Synthetic CDC events
        """
        while self._running or not self._event_queue.empty():
            try:
                event = self._event_queue.get(timeout=timeout)
                yield event
            except queue.Empty:
                continue

    def get_next_change(self, timeout: float = 1.0) -> Optional[SyntheticCDCEvent]:
        """Get the next synthetic CDC event.

        Args:
            timeout: Timeout in seconds

        Returns:
            Synthetic CDC event or None if timeout
        """
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_batch(
        self,
        batch_size: int = 100,
        timeout: float = 1.0,
    ) -> List[SyntheticCDCEvent]:
        """Get a batch of synthetic CDC events.

        Args:
            batch_size: Maximum batch size
            timeout: Timeout in seconds

        Returns:
            List of synthetic CDC events
        """
        batch = []
        deadline = time.time() + timeout

        while len(batch) < batch_size and time.time() < deadline:
            remaining = deadline - time.time()
            event = self.get_next_change(timeout=max(0.01, remaining))
            if event:
                batch.append(event)

        return batch

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics.

        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()

    def __enter__(self) -> "CDCGenerator":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


class CDCEventWriter:
    """Write synthetic CDC events to various destinations."""

    @staticmethod
    def to_kafka(
        events: List[SyntheticCDCEvent],
        bootstrap_servers: str,
        topic: str,
    ) -> int:
        """Write events to Kafka.

        Args:
            events: Events to write
            bootstrap_servers: Kafka bootstrap servers
            topic: Target topic

        Returns:
            Number of events written
        """
        try:
            from kafka import KafkaProducer
        except ImportError:
            raise ImportError("kafka-python required: pip install kafka-python")

        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        for event in events:
            producer.send(topic, value=event.to_dict())

        producer.flush()
        producer.close()

        return len(events)

    @staticmethod
    def to_json_file(
        events: List[SyntheticCDCEvent],
        file_path: str,
        append: bool = True,
    ) -> int:
        """Write events to JSON file.

        Args:
            events: Events to write
            file_path: Output file path
            append: Append to existing file

        Returns:
            Number of events written
        """
        mode = "a" if append else "w"

        with open(file_path, mode) as f:
            for event in events:
                f.write(json.dumps(event.to_dict()) + "\n")

        return len(events)

    @staticmethod
    def to_database(
        events: List[SyntheticCDCEvent],
        connection_string: str,
        table_prefix: str = "synthetic_",
    ) -> int:
        """Write events to database.

        Args:
            events: Events to write
            connection_string: Database connection string
            table_prefix: Prefix for synthetic tables

        Returns:
            Number of events written
        """
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise ImportError("SQLAlchemy required: pip install sqlalchemy")

        engine = create_engine(connection_string)
        written = 0

        with engine.connect() as conn:
            for event in events:
                if event.operation == OperationType.INSERT and event.synthetic_after:
                    table = f"{table_prefix}{event.table}"
                    columns = list(event.synthetic_after.keys())
                    values = list(event.synthetic_after.values())

                    placeholders = ", ".join([f":{c}" for c in columns])
                    col_names = ", ".join(columns)

                    query = text(f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})")
                    conn.execute(query, event.synthetic_after)
                    written += 1

            conn.commit()

        return written


# Convenience functions
def create_cdc_generator(
    source_type: Union[str, CDCSourceType],
    connection_string: Optional[str] = None,
    tables: Optional[List[str]] = None,
    **kwargs: Any,
) -> CDCGenerator:
    """Create a CDC generator with simplified configuration.

    Args:
        source_type: Type of CDC source
        connection_string: Database/Kafka connection string
        tables: Tables to monitor
        **kwargs: Additional configuration

    Returns:
        Configured CDCGenerator

    Example:
        >>> generator = create_cdc_generator(
        ...     "postgresql",
        ...     connection_string="postgresql://localhost/mydb",
        ...     tables=["users", "orders"],
        ... )
    """
    if isinstance(source_type, str):
        source_type = CDCSourceType(source_type)

    config = CDCConfig(
        source_type=source_type,
        connection_string=connection_string,
        tables=tables or [],
        **kwargs,
    )

    return CDCGenerator(config)


__all__ = [
    # Core classes
    "CDCGenerator",
    "CDCSynthesizer",
    "CDCEventWriter",
    # Configuration
    "CDCConfig",
    # Data classes
    "CDCEvent",
    "SyntheticCDCEvent",
    # Enums
    "CDCSourceType",
    "OperationType",
    # Sources
    "CDCSource",
    "FileCDCSource",
    "KafkaCDCSource",
    "DatabaseCDCSource",
    # Convenience functions
    "create_cdc_generator",
]
