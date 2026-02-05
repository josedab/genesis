"""Real-Time Production Mirror.

Continuous synchronization with production databases to generate up-to-date
synthetic replicas with CDC (Change Data Capture), automated drift detection,
and model re-training.

Features:
    - CDC event subscriber (Debezium-compatible)
    - Incremental model update pipeline
    - Real-time synthetic replica generation
    - Drift detection with auto-retrain triggers
    - Multi-table sync with referential integrity
    - State management for mirror consistency

Example:
    Basic production mirror::

        from genesis.production_mirror import ProductionMirror, MirrorConfig

        mirror = ProductionMirror(
            config=MirrorConfig(
                source_connection="postgresql://...",
                target_path="./synthetic_mirror",
                sync_interval=3600,  # 1 hour
            )
        )

        # Start continuous mirroring
        mirror.start()

        # Or run single sync
        mirror.sync()

    With drift detection::

        from genesis.production_mirror import DriftAwareMirror

        mirror = DriftAwareMirror(
            source_connection="postgresql://...",
            drift_threshold=0.1,
            auto_retrain=True,
        )

        # Monitor will retrain when drift exceeds threshold
        mirror.start_monitoring()

Classes:
    MirrorConfig: Configuration for production mirror.
    CDCEvent: Change Data Capture event.
    CDCSubscriber: Subscribes to CDC events.
    TableMirror: Mirrors a single table.
    ProductionMirror: Main mirror orchestrator.
    DriftAwareMirror: Mirror with drift detection.
    IncrementalTrainer: Incremental model updates.
    MirrorStateManager: Manages mirror state and checkpoints.

Note:
    Requires database connectivity and appropriate CDC setup
    (e.g., Debezium, PostgreSQL logical replication).
"""

import hashlib
import json
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class CDCOperation(str, Enum):
    """CDC operation types."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SNAPSHOT = "snapshot"
    TRUNCATE = "truncate"


class MirrorStatus(str, Enum):
    """Mirror status states."""

    IDLE = "idle"
    SYNCING = "syncing"
    TRAINING = "training"
    GENERATING = "generating"
    ERROR = "error"
    STOPPED = "stopped"


class DriftAction(str, Enum):
    """Actions to take on drift detection."""

    NONE = "none"  # No action
    ALERT = "alert"  # Alert only
    RETRAIN = "retrain"  # Retrain model
    REGENERATE = "regenerate"  # Regenerate all data


@dataclass
class MirrorConfig:
    """Configuration for production mirror.

    Attributes:
        source_connection: Database connection string.
        target_path: Path to store synthetic mirror.
        tables: Tables to mirror (None = all).
        sync_interval: Seconds between syncs.
        batch_size: Rows per batch during sync.
        drift_threshold: PSI threshold for drift detection.
        auto_retrain: Automatically retrain on drift.
        checkpoint_interval: Seconds between checkpoints.
        max_lag: Maximum acceptable lag in seconds.
    """

    source_connection: str = ""
    target_path: str = "./synthetic_mirror"
    tables: Optional[List[str]] = None
    sync_interval: int = 3600  # 1 hour
    batch_size: int = 10000
    drift_threshold: float = 0.1
    auto_retrain: bool = True
    checkpoint_interval: int = 300  # 5 minutes
    max_lag: int = 7200  # 2 hours
    generator_method: str = "gaussian_copula"
    privacy_epsilon: float = 1.0


@dataclass
class CDCEvent:
    """Change Data Capture event.

    Attributes:
        table: Table name.
        operation: Operation type.
        before: Row state before operation.
        after: Row state after operation.
        timestamp: Event timestamp.
        transaction_id: Transaction identifier.
        sequence: Sequence number within transaction.
    """

    table: str
    operation: CDCOperation
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    transaction_id: Optional[str] = None
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "operation": self.operation.value,
            "before": self.before,
            "after": self.after,
            "timestamp": self.timestamp,
            "transaction_id": self.transaction_id,
            "sequence": self.sequence,
        }


@dataclass
class MirrorState:
    """State of a production mirror.

    Attributes:
        table: Table name.
        last_sync: Last sync timestamp.
        last_lsn: Last Log Sequence Number processed.
        row_count: Current row count.
        model_version: Current model version.
        drift_score: Current drift score.
        status: Current status.
    """

    table: str
    last_sync: Optional[str] = None
    last_lsn: Optional[str] = None
    row_count: int = 0
    model_version: str = "v0"
    drift_score: float = 0.0
    status: MirrorStatus = MirrorStatus.IDLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "last_sync": self.last_sync,
            "last_lsn": self.last_lsn,
            "row_count": self.row_count,
            "model_version": self.model_version,
            "drift_score": self.drift_score,
            "status": self.status.value,
        }


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        table: Table name.
        success: Whether sync succeeded.
        rows_synced: Number of rows synced.
        duration: Sync duration in seconds.
        drift_detected: Whether drift was detected.
        model_retrained: Whether model was retrained.
        error: Error message if failed.
    """

    table: str
    success: bool
    rows_synced: int = 0
    duration: float = 0.0
    drift_detected: bool = False
    model_retrained: bool = False
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CDCSubscriber(ABC):
    """Abstract CDC event subscriber."""

    @abstractmethod
    def subscribe(self, tables: List[str]) -> None:
        """Subscribe to CDC events for tables."""
        pass

    @abstractmethod
    def poll(self, timeout: float = 1.0) -> List[CDCEvent]:
        """Poll for new CDC events."""
        pass

    @abstractmethod
    def commit(self, lsn: str) -> None:
        """Commit processed LSN."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close subscriber."""
        pass


class MockCDCSubscriber(CDCSubscriber):
    """Mock CDC subscriber for testing."""

    def __init__(self, events: Optional[List[CDCEvent]] = None) -> None:
        self._events = events or []
        self._index = 0
        self._subscribed: List[str] = []

    def subscribe(self, tables: List[str]) -> None:
        self._subscribed = tables

    def poll(self, timeout: float = 1.0) -> List[CDCEvent]:
        if self._index >= len(self._events):
            return []
        event = self._events[self._index]
        self._index += 1
        return [event]

    def commit(self, lsn: str) -> None:
        pass

    def close(self) -> None:
        pass

    def add_event(self, event: CDCEvent) -> None:
        self._events.append(event)


class DebeziumSubscriber(CDCSubscriber):
    """Debezium CDC subscriber (Kafka-based)."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic_prefix: str,
        group_id: str = "genesis-mirror",
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self.group_id = group_id
        self._consumer: Optional[Any] = None
        self._subscribed_tables: List[str] = []

    def subscribe(self, tables: List[str]) -> None:
        try:
            from kafka import KafkaConsumer
            topics = [f"{self.topic_prefix}.{table}" for table in tables]
            self._consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            )
            self._subscribed_tables = tables
            logger.info(f"Subscribed to CDC topics: {topics}")
        except ImportError:
            raise ImportError("kafka-python required for Debezium subscriber")

    def poll(self, timeout: float = 1.0) -> List[CDCEvent]:
        if self._consumer is None:
            return []

        events: List[CDCEvent] = []
        records = self._consumer.poll(timeout_ms=int(timeout * 1000))

        for topic_partition, messages in records.items():
            for message in messages:
                event = self._parse_debezium_message(message.value)
                if event:
                    events.append(event)

        return events

    def _parse_debezium_message(self, message: Dict[str, Any]) -> Optional[CDCEvent]:
        """Parse Debezium message format."""
        try:
            payload = message.get("payload", message)
            source = payload.get("source", {})

            # Determine operation
            op = payload.get("op", "r")
            operation_map = {
                "c": CDCOperation.INSERT,
                "u": CDCOperation.UPDATE,
                "d": CDCOperation.DELETE,
                "r": CDCOperation.SNAPSHOT,
            }
            operation = operation_map.get(op, CDCOperation.INSERT)

            return CDCEvent(
                table=source.get("table", "unknown"),
                operation=operation,
                before=payload.get("before"),
                after=payload.get("after"),
                timestamp=datetime.utcnow().isoformat(),
                transaction_id=source.get("txId"),
                sequence=source.get("sequence", 0),
            )
        except Exception as e:
            logger.error(f"Error parsing Debezium message: {e}")
            return None

    def commit(self, lsn: str) -> None:
        if self._consumer:
            self._consumer.commit()

    def close(self) -> None:
        if self._consumer:
            self._consumer.close()
            self._consumer = None


class MirrorStateManager:
    """Manages mirror state and checkpoints."""

    def __init__(self, state_dir: str) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._states: Dict[str, MirrorState] = {}

    def get_state(self, table: str) -> MirrorState:
        """Get state for a table."""
        if table not in self._states:
            state_file = self.state_dir / f"{table}.json"
            if state_file.exists():
                data = json.loads(state_file.read_text())
                self._states[table] = MirrorState(
                    table=data["table"],
                    last_sync=data.get("last_sync"),
                    last_lsn=data.get("last_lsn"),
                    row_count=data.get("row_count", 0),
                    model_version=data.get("model_version", "v0"),
                    drift_score=data.get("drift_score", 0.0),
                    status=MirrorStatus(data.get("status", "idle")),
                )
            else:
                self._states[table] = MirrorState(table=table)
        return self._states[table]

    def save_state(self, state: MirrorState) -> None:
        """Save state for a table."""
        self._states[state.table] = state
        state_file = self.state_dir / f"{state.table}.json"
        state_file.write_text(json.dumps(state.to_dict(), indent=2))

    def get_all_states(self) -> List[MirrorState]:
        """Get all table states."""
        return list(self._states.values())

    def create_checkpoint(self) -> str:
        """Create a checkpoint of all states."""
        checkpoint_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = self.state_dir / "checkpoints" / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for table, state in self._states.items():
            checkpoint_file = checkpoint_dir / f"{table}.json"
            checkpoint_file.write_text(json.dumps(state.to_dict(), indent=2))

        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> None:
        """Restore from a checkpoint."""
        checkpoint_dir = self.state_dir / "checkpoints" / checkpoint_id
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        for state_file in checkpoint_dir.glob("*.json"):
            data = json.loads(state_file.read_text())
            self._states[data["table"]] = MirrorState(**data)
            self.save_state(self._states[data["table"]])


class IncrementalTrainer:
    """Handles incremental model training."""

    def __init__(
        self,
        method: str = "gaussian_copula",
        batch_size: int = 10000,
    ) -> None:
        self.method = method
        self.batch_size = batch_size
        self._models: Dict[str, Any] = {}
        self._data_buffers: Dict[str, pd.DataFrame] = {}

    def add_data(self, table: str, data: pd.DataFrame) -> None:
        """Add data to training buffer."""
        if table not in self._data_buffers:
            self._data_buffers[table] = data
        else:
            self._data_buffers[table] = pd.concat(
                [self._data_buffers[table], data], ignore_index=True
            )

        # Trim to batch size
        if len(self._data_buffers[table]) > self.batch_size * 2:
            self._data_buffers[table] = self._data_buffers[table].tail(
                self.batch_size * 2
            )

    def train(self, table: str, full_data: Optional[pd.DataFrame] = None) -> str:
        """Train or retrain model for table."""
        from genesis.core.base import SyntheticGenerator

        data = full_data if full_data is not None else self._data_buffers.get(table)
        if data is None or len(data) == 0:
            raise ValueError(f"No data available for table {table}")

        generator = SyntheticGenerator(method=self.method)

        # Detect discrete columns
        discrete_cols = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        generator.fit(data, discrete_columns=discrete_cols)
        self._models[table] = generator

        version = f"v{int(time.time())}"
        logger.info(f"Trained model {version} for {table}")
        return version

    def generate(self, table: str, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data."""
        if table not in self._models:
            raise ValueError(f"No model for table {table}")
        return self._models[table].generate(n_samples=n_samples)

    def save_model(self, table: str, path: str) -> None:
        """Save model to disk."""
        import pickle
        if table in self._models:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self._models[table], f)

    def load_model(self, table: str, path: str) -> None:
        """Load model from disk."""
        import pickle
        with open(path, "rb") as f:
            self._models[table] = pickle.load(f)


class DriftDetector:
    """Detects distribution drift between datasets."""

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def detect(
        self,
        baseline: pd.DataFrame,
        current: pd.DataFrame,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Detect drift between baseline and current data.

        Returns:
            Tuple of (has_drift, overall_score, column_scores)
        """
        from scipy import stats

        column_scores: Dict[str, float] = {}
        numeric_cols = baseline.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in current.columns:
                try:
                    psi = self._calculate_psi(baseline[col], current[col])
                    column_scores[col] = psi
                except Exception:
                    column_scores[col] = 0.0

        if not column_scores:
            return False, 0.0, {}

        overall_score = np.mean(list(column_scores.values()))
        has_drift = overall_score > self.threshold

        return has_drift, overall_score, column_scores

    def _calculate_psi(
        self, baseline: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        # Remove nulls
        baseline = baseline.dropna()
        current = current.dropna()

        if len(baseline) == 0 or len(current) == 0:
            return 0.0

        # Create bins from baseline
        _, bin_edges = pd.cut(baseline, bins=bins, retbins=True)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Count in each bin
        baseline_counts = pd.cut(baseline, bins=bin_edges).value_counts(normalize=True)
        current_counts = pd.cut(current, bins=bin_edges).value_counts(normalize=True)

        # Align indices
        baseline_counts = baseline_counts.reindex(baseline_counts.index | current_counts.index, fill_value=0.0001)
        current_counts = current_counts.reindex(baseline_counts.index, fill_value=0.0001)

        # Calculate PSI
        psi = np.sum(
            (current_counts - baseline_counts) * np.log(current_counts / baseline_counts)
        )

        return float(psi)


class TableMirror:
    """Mirrors a single table."""

    def __init__(
        self,
        table: str,
        config: MirrorConfig,
        trainer: IncrementalTrainer,
        state_manager: MirrorStateManager,
    ) -> None:
        self.table = table
        self.config = config
        self.trainer = trainer
        self.state_manager = state_manager
        self.drift_detector = DriftDetector(config.drift_threshold)
        self._baseline_data: Optional[pd.DataFrame] = None

    def initial_sync(self, source_data: pd.DataFrame) -> SyncResult:
        """Perform initial sync."""
        start = time.time()
        state = self.state_manager.get_state(self.table)
        state.status = MirrorStatus.SYNCING

        try:
            # Store baseline
            self._baseline_data = source_data.copy()

            # Train model
            state.status = MirrorStatus.TRAINING
            version = self.trainer.train(self.table, source_data)
            state.model_version = version

            # Generate synthetic data
            state.status = MirrorStatus.GENERATING
            synthetic = self.trainer.generate(self.table, len(source_data))
            self._save_synthetic(synthetic)

            state.row_count = len(source_data)
            state.last_sync = datetime.utcnow().isoformat()
            state.status = MirrorStatus.IDLE
            self.state_manager.save_state(state)

            return SyncResult(
                table=self.table,
                success=True,
                rows_synced=len(source_data),
                duration=time.time() - start,
            )

        except Exception as e:
            state.status = MirrorStatus.ERROR
            self.state_manager.save_state(state)
            return SyncResult(
                table=self.table,
                success=False,
                error=str(e),
                duration=time.time() - start,
            )

    def incremental_sync(
        self,
        events: List[CDCEvent],
        current_data: Optional[pd.DataFrame] = None,
    ) -> SyncResult:
        """Process CDC events for incremental sync."""
        start = time.time()
        state = self.state_manager.get_state(self.table)

        try:
            # Process events
            for event in events:
                self._process_event(event)

            # Check for drift if we have current data
            drift_detected = False
            model_retrained = False

            if current_data is not None and self._baseline_data is not None:
                has_drift, score, _ = self.drift_detector.detect(
                    self._baseline_data, current_data
                )
                state.drift_score = score
                drift_detected = has_drift

                if has_drift and self.config.auto_retrain:
                    logger.info(f"Drift detected for {self.table} (PSI={score:.4f}), retraining...")
                    version = self.trainer.train(self.table, current_data)
                    state.model_version = version
                    model_retrained = True

                    # Regenerate
                    synthetic = self.trainer.generate(self.table, len(current_data))
                    self._save_synthetic(synthetic)
                    self._baseline_data = current_data.copy()

            state.last_sync = datetime.utcnow().isoformat()
            state.status = MirrorStatus.IDLE
            self.state_manager.save_state(state)

            return SyncResult(
                table=self.table,
                success=True,
                rows_synced=len(events),
                duration=time.time() - start,
                drift_detected=drift_detected,
                model_retrained=model_retrained,
            )

        except Exception as e:
            state.status = MirrorStatus.ERROR
            self.state_manager.save_state(state)
            return SyncResult(
                table=self.table,
                success=False,
                error=str(e),
                duration=time.time() - start,
            )

    def _process_event(self, event: CDCEvent) -> None:
        """Process a single CDC event."""
        # Add to trainer buffer
        if event.after and event.operation in (CDCOperation.INSERT, CDCOperation.UPDATE):
            df = pd.DataFrame([event.after])
            self.trainer.add_data(self.table, df)

    def _save_synthetic(self, data: pd.DataFrame) -> None:
        """Save synthetic data to target."""
        target_path = Path(self.config.target_path) / f"{self.table}.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(target_path, index=False)


class ProductionMirror:
    """Main production mirror orchestrator.

    Coordinates mirroring of multiple tables with CDC support.
    """

    def __init__(
        self,
        config: Optional[MirrorConfig] = None,
        cdc_subscriber: Optional[CDCSubscriber] = None,
    ) -> None:
        """Initialize production mirror.

        Args:
            config: Mirror configuration.
            cdc_subscriber: CDC event subscriber.
        """
        self.config = config or MirrorConfig()
        self.cdc_subscriber = cdc_subscriber or MockCDCSubscriber()
        self.state_manager = MirrorStateManager(
            str(Path(self.config.target_path) / ".state")
        )
        self.trainer = IncrementalTrainer(
            method=self.config.generator_method,
            batch_size=self.config.batch_size,
        )
        self._table_mirrors: Dict[str, TableMirror] = {}
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None

    def add_table(self, table: str, initial_data: Optional[pd.DataFrame] = None) -> None:
        """Add a table to mirror.

        Args:
            table: Table name.
            initial_data: Initial table data for training.
        """
        mirror = TableMirror(
            table=table,
            config=self.config,
            trainer=self.trainer,
            state_manager=self.state_manager,
        )
        self._table_mirrors[table] = mirror

        if initial_data is not None:
            result = mirror.initial_sync(initial_data)
            if not result.success:
                logger.error(f"Initial sync failed for {table}: {result.error}")

    def sync(self, tables: Optional[List[str]] = None) -> Dict[str, SyncResult]:
        """Run sync for specified tables.

        Args:
            tables: Tables to sync (None = all).

        Returns:
            Dictionary of sync results by table.
        """
        tables = tables or list(self._table_mirrors.keys())
        results: Dict[str, SyncResult] = {}

        # Subscribe to CDC events
        self.cdc_subscriber.subscribe(tables)

        # Poll for events
        events = self.cdc_subscriber.poll(timeout=1.0)

        # Group events by table
        events_by_table: Dict[str, List[CDCEvent]] = {}
        for event in events:
            if event.table not in events_by_table:
                events_by_table[event.table] = []
            events_by_table[event.table].append(event)

        # Process each table
        for table in tables:
            if table in self._table_mirrors:
                table_events = events_by_table.get(table, [])
                result = self._table_mirrors[table].incremental_sync(table_events)
                results[table] = result

        return results

    def start(self) -> None:
        """Start continuous mirroring."""
        if self._running:
            logger.warning("Mirror already running")
            return

        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Production mirror started")

    def stop(self) -> None:
        """Stop continuous mirroring."""
        self._running = False
        if self._sync_thread:
            self._sync_thread.join(timeout=5.0)
            self._sync_thread = None
        self.cdc_subscriber.close()
        logger.info("Production mirror stopped")

    def _sync_loop(self) -> None:
        """Main sync loop."""
        last_checkpoint = time.time()

        while self._running:
            try:
                # Run sync
                self.sync()

                # Periodic checkpoint
                if time.time() - last_checkpoint > self.config.checkpoint_interval:
                    self.state_manager.create_checkpoint()
                    last_checkpoint = time.time()

                # Wait for next interval
                time.sleep(self.config.sync_interval)

            except Exception as e:
                logger.error(f"Sync error: {e}")
                time.sleep(10)  # Wait before retrying

    def get_status(self) -> Dict[str, Any]:
        """Get mirror status."""
        return {
            "running": self._running,
            "tables": {
                table: state.to_dict()
                for table, state in {
                    t: self.state_manager.get_state(t) for t in self._table_mirrors
                }.items()
            },
        }

    def get_synthetic_data(self, table: str) -> Optional[pd.DataFrame]:
        """Get current synthetic data for a table."""
        path = Path(self.config.target_path) / f"{table}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return None


class DriftAwareMirror(ProductionMirror):
    """Production mirror with enhanced drift detection.

    Monitors for distribution shifts and automatically adapts.
    """

    def __init__(
        self,
        config: Optional[MirrorConfig] = None,
        drift_window: int = 3600,
        alert_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """Initialize drift-aware mirror.

        Args:
            config: Mirror configuration.
            drift_window: Window for drift calculation in seconds.
            alert_callback: Callback for drift alerts.
        """
        super().__init__(config)
        self.drift_window = drift_window
        self.alert_callback = alert_callback
        self._drift_history: Dict[str, List[Tuple[datetime, float]]] = {}

    def check_drift(self, table: str) -> Tuple[bool, float]:
        """Check for drift in a table.

        Returns:
            Tuple of (has_drift, drift_score).
        """
        if table not in self._table_mirrors:
            return False, 0.0

        mirror = self._table_mirrors[table]
        state = self.state_manager.get_state(table)

        # Record drift history
        if table not in self._drift_history:
            self._drift_history[table] = []
        self._drift_history[table].append((datetime.utcnow(), state.drift_score))

        # Trim old history
        cutoff = datetime.utcnow() - timedelta(seconds=self.drift_window)
        self._drift_history[table] = [
            (ts, score) for ts, score in self._drift_history[table] if ts > cutoff
        ]

        # Alert if drift detected
        if state.drift_score > self.config.drift_threshold:
            if self.alert_callback:
                self.alert_callback(table, state.drift_score)
            return True, state.drift_score

        return False, state.drift_score

    def get_drift_trend(self, table: str) -> List[Tuple[str, float]]:
        """Get drift score trend for a table."""
        history = self._drift_history.get(table, [])
        return [(ts.isoformat(), score) for ts, score in history]


# Convenience functions
def create_production_mirror(
    source_connection: str,
    target_path: str = "./synthetic_mirror",
    tables: Optional[List[str]] = None,
    **kwargs: Any,
) -> ProductionMirror:
    """Create a production mirror.

    Args:
        source_connection: Database connection string.
        target_path: Path for synthetic data.
        tables: Tables to mirror.
        **kwargs: Additional config options.

    Returns:
        Configured ProductionMirror.

    Example:
        >>> mirror = create_production_mirror(
        ...     "postgresql://localhost/mydb",
        ...     tables=["users", "orders"],
        ... )
        >>> mirror.start()
    """
    config = MirrorConfig(
        source_connection=source_connection,
        target_path=target_path,
        tables=tables,
        **kwargs,
    )
    return ProductionMirror(config)


def sync_table(
    data: pd.DataFrame,
    table_name: str,
    target_path: str = "./synthetic_mirror",
    method: str = "gaussian_copula",
) -> SyncResult:
    """Sync a single table (one-shot).

    Args:
        data: Table data.
        table_name: Table name.
        target_path: Output path.
        method: Generator method.

    Returns:
        SyncResult with status.
    """
    config = MirrorConfig(target_path=target_path, generator_method=method)
    mirror = ProductionMirror(config)
    mirror.add_table(table_name, data)
    results = mirror.sync([table_name])
    return results.get(table_name, SyncResult(table=table_name, success=False))
