"""Incremental/Delta Generation for Genesis.

This module provides capabilities for generating synthetic data incrementally,
supporting CDC (Change Data Capture) workflows and maintaining referential
integrity across updates.

Features:
- Generate only changed/new records
- CDC-compatible output formats
- Delta tracking and versioning
- Referential integrity maintenance
- Merge operations for slowly changing dimensions

Example:
    >>> from genesis.delta import DeltaGenerator, ChangeTracker
    >>>
    >>> tracker = ChangeTracker()
    >>> generator = DeltaGenerator(base_generator, tracker)
    >>>
    >>> # Initial load
    >>> initial_data = generator.generate_full(10000)
    >>>
    >>> # Incremental updates
    >>> delta = generator.generate_delta(
    ...     insert_ratio=0.1,
    ...     update_ratio=0.05,
    ...     delete_ratio=0.02
    ... )
"""

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.exceptions import GenesisError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ChangeType(Enum):
    """Types of changes in CDC."""

    INSERT = "I"
    UPDATE = "U"
    DELETE = "D"
    UPSERT = "X"


@dataclass
class ChangeRecord:
    """Represents a single change record."""

    change_type: ChangeType
    record_id: str
    timestamp: datetime
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "op": self.change_type.value,
            "id": self.record_id,
            "ts": self.timestamp.isoformat(),
            "before": self.before,
            "after": self.after,
            "version": self.version,
            **self.metadata,
        }


@dataclass
class DeltaResult:
    """Result of a delta generation operation."""

    inserts: pd.DataFrame
    updates: pd.DataFrame
    deletes: pd.DataFrame
    change_records: List[ChangeRecord]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_changes(self) -> int:
        """Total number of changes."""
        return len(self.inserts) + len(self.updates) + len(self.deletes)

    def to_cdc_format(self, format_type: str = "debezium") -> List[Dict[str, Any]]:
        """Convert to CDC format.

        Args:
            format_type: One of 'debezium', 'maxwell', 'canal'

        Returns:
            List of CDC records
        """
        if format_type == "debezium":
            return self._to_debezium()
        elif format_type == "maxwell":
            return self._to_maxwell()
        else:
            return [r.to_dict() for r in self.change_records]

    def _to_debezium(self) -> List[Dict[str, Any]]:
        """Convert to Debezium format."""
        records = []

        for cr in self.change_records:
            record = {
                "payload": {
                    "op": cr.change_type.value.lower(),
                    "ts_ms": int(cr.timestamp.timestamp() * 1000),
                    "before": cr.before,
                    "after": cr.after,
                },
                "schema": {},
            }
            records.append(record)

        return records

    def _to_maxwell(self) -> List[Dict[str, Any]]:
        """Convert to Maxwell format."""
        records = []

        for cr in self.change_records:
            type_map = {
                ChangeType.INSERT: "insert",
                ChangeType.UPDATE: "update",
                ChangeType.DELETE: "delete",
            }

            record = {
                "type": type_map.get(cr.change_type, "unknown"),
                "ts": int(cr.timestamp.timestamp()),
                "data": cr.after or cr.before,
                "old": cr.before if cr.change_type == ChangeType.UPDATE else None,
            }
            records.append(record)

        return records


class ChangeTracker:
    """Tracks changes to generated data over time.

    Example:
        >>> tracker = ChangeTracker(id_column="user_id")
        >>> tracker.track_initial(initial_df)
        >>> tracker.track_changes(new_df)
        >>> changes = tracker.get_changes()
    """

    def __init__(
        self,
        id_column: str = "id",
        version_column: Optional[str] = None,
        track_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize change tracker.

        Args:
            id_column: Column containing unique identifiers
            version_column: Column for version tracking (optional)
            track_columns: Columns to track for changes (None = all)
        """
        self.id_column = id_column
        self.version_column = version_column
        self.track_columns = track_columns

        self._current_state: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, int] = defaultdict(int)
        self._history: List[ChangeRecord] = []

    def track_initial(self, data: pd.DataFrame) -> None:
        """Track initial data state.

        Args:
            data: Initial DataFrame to track
        """
        for _, row in data.iterrows():
            record_id = str(row[self.id_column])
            self._current_state[record_id] = row.to_dict()
            self._versions[record_id] = 1

        logger.info(f"Tracking {len(data)} initial records")

    def track_changes(
        self,
        new_data: pd.DataFrame,
        timestamp: Optional[datetime] = None,
    ) -> List[ChangeRecord]:
        """Track changes between current state and new data.

        Args:
            new_data: New DataFrame state
            timestamp: Timestamp for changes

        Returns:
            List of detected changes
        """
        timestamp = timestamp or datetime.utcnow()
        changes = []

        new_ids = set(str(r[self.id_column]) for _, r in new_data.iterrows())
        current_ids = set(self._current_state.keys())

        # Detect inserts
        for record_id in new_ids - current_ids:
            row = new_data[new_data[self.id_column].astype(str) == record_id].iloc[0]
            after = row.to_dict()

            change = ChangeRecord(
                change_type=ChangeType.INSERT,
                record_id=record_id,
                timestamp=timestamp,
                after=after,
                version=1,
            )
            changes.append(change)
            self._current_state[record_id] = after
            self._versions[record_id] = 1

        # Detect updates
        for record_id in new_ids & current_ids:
            row = new_data[new_data[self.id_column].astype(str) == record_id].iloc[0]
            new_values = row.to_dict()
            old_values = self._current_state[record_id]

            if self._has_changes(old_values, new_values):
                self._versions[record_id] += 1

                change = ChangeRecord(
                    change_type=ChangeType.UPDATE,
                    record_id=record_id,
                    timestamp=timestamp,
                    before=old_values,
                    after=new_values,
                    version=self._versions[record_id],
                )
                changes.append(change)
                self._current_state[record_id] = new_values

        # Detect deletes
        for record_id in current_ids - new_ids:
            old_values = self._current_state[record_id]

            change = ChangeRecord(
                change_type=ChangeType.DELETE,
                record_id=record_id,
                timestamp=timestamp,
                before=old_values,
                version=self._versions[record_id],
            )
            changes.append(change)
            del self._current_state[record_id]

        self._history.extend(changes)
        logger.info(
            f"Tracked changes: {sum(1 for c in changes if c.change_type == ChangeType.INSERT)} inserts, "
            f"{sum(1 for c in changes if c.change_type == ChangeType.UPDATE)} updates, "
            f"{sum(1 for c in changes if c.change_type == ChangeType.DELETE)} deletes"
        )

        return changes

    def _has_changes(
        self, old: Dict[str, Any], new: Dict[str, Any]
    ) -> bool:
        """Check if there are changes between old and new values."""
        columns = self.track_columns or list(old.keys())

        for col in columns:
            if col == self.id_column:
                continue
            if col not in old or col not in new:
                return True
            if str(old.get(col)) != str(new.get(col)):
                return True

        return False

    def get_current_state(self) -> pd.DataFrame:
        """Get current tracked state as DataFrame."""
        if not self._current_state:
            return pd.DataFrame()
        return pd.DataFrame(list(self._current_state.values()))

    def get_history(
        self,
        since: Optional[datetime] = None,
    ) -> List[ChangeRecord]:
        """Get change history.

        Args:
            since: Filter changes after this timestamp

        Returns:
            List of change records
        """
        if since:
            return [c for c in self._history if c.timestamp >= since]
        return list(self._history)


class DeltaGenerator:
    """Generates incremental/delta synthetic data.

    Example:
        >>> generator = DeltaGenerator(
        ...     base_generator=my_generator,
        ...     id_column="user_id"
        ... )
        >>> initial = generator.generate_full(10000)
        >>> delta = generator.generate_delta(
        ...     insert_ratio=0.1,
        ...     update_ratio=0.05
        ... )
    """

    def __init__(
        self,
        base_generator: BaseGenerator,
        id_column: str = "id",
        timestamp_column: Optional[str] = None,
        version_column: Optional[str] = None,
    ) -> None:
        """Initialize delta generator.

        Args:
            base_generator: Base generator for synthetic data
            id_column: Column for unique identifiers
            timestamp_column: Column for timestamps
            version_column: Column for version tracking
        """
        self.base_generator = base_generator
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.version_column = version_column

        self.tracker = ChangeTracker(
            id_column=id_column,
            version_column=version_column,
        )

        self._current_data: Optional[pd.DataFrame] = None

    def generate_full(
        self,
        n_rows: int,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate full initial dataset.

        Args:
            n_rows: Number of rows to generate
            random_state: Random seed

        Returns:
            Generated DataFrame
        """
        data = self.base_generator.generate(n_rows)

        # Add ID column if not present
        if self.id_column not in data.columns:
            data[self.id_column] = [str(uuid.uuid4()) for _ in range(len(data))]

        # Add timestamp column if specified
        if self.timestamp_column and self.timestamp_column not in data.columns:
            data[self.timestamp_column] = datetime.utcnow()

        # Add version column if specified
        if self.version_column and self.version_column not in data.columns:
            data[self.version_column] = 1

        self._current_data = data.copy()
        self.tracker.track_initial(data)

        return data

    def generate_delta(
        self,
        insert_ratio: float = 0.1,
        update_ratio: float = 0.05,
        delete_ratio: float = 0.02,
        update_columns: Optional[List[str]] = None,
        timestamp: Optional[datetime] = None,
        random_state: Optional[int] = None,
    ) -> DeltaResult:
        """Generate delta changes.

        Args:
            insert_ratio: Ratio of new records to insert
            update_ratio: Ratio of existing records to update
            delete_ratio: Ratio of existing records to delete
            update_columns: Columns to update (None = random)
            timestamp: Timestamp for changes
            random_state: Random seed

        Returns:
            DeltaResult with changes
        """
        if self._current_data is None:
            raise GenesisError("Must call generate_full() first")

        rng = np.random.default_rng(random_state)
        timestamp = timestamp or datetime.utcnow()
        current_size = len(self._current_data)

        # Calculate counts
        n_inserts = int(current_size * insert_ratio)
        n_updates = int(current_size * update_ratio)
        n_deletes = int(current_size * delete_ratio)

        # Generate inserts
        inserts = self._generate_inserts(n_inserts, timestamp, rng)

        # Generate updates
        updates, update_records = self._generate_updates(
            n_updates, update_columns, timestamp, rng
        )

        # Generate deletes
        deletes, delete_records = self._generate_deletes(n_deletes, timestamp, rng)

        # Build change records
        change_records = []

        for _, row in inserts.iterrows():
            change_records.append(
                ChangeRecord(
                    change_type=ChangeType.INSERT,
                    record_id=str(row[self.id_column]),
                    timestamp=timestamp,
                    after=row.to_dict(),
                    version=1,
                )
            )

        change_records.extend(update_records)
        change_records.extend(delete_records)

        # Update tracking
        self.tracker.track_changes(
            self._current_data.copy(), timestamp
        )

        return DeltaResult(
            inserts=inserts,
            updates=updates,
            deletes=deletes,
            change_records=change_records,
            timestamp=timestamp,
        )

    def _generate_inserts(
        self,
        n: int,
        timestamp: datetime,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Generate insert records."""
        if n == 0:
            return pd.DataFrame()

        inserts = self.base_generator.generate(n)

        # Add ID column
        if self.id_column not in inserts.columns:
            inserts[self.id_column] = [str(uuid.uuid4()) for _ in range(n)]

        if self.timestamp_column:
            inserts[self.timestamp_column] = timestamp

        if self.version_column:
            inserts[self.version_column] = 1

        # Add to current data
        self._current_data = pd.concat(
            [self._current_data, inserts], ignore_index=True
        )

        return inserts

    def _generate_updates(
        self,
        n: int,
        update_columns: Optional[List[str]],
        timestamp: datetime,
        rng: np.random.Generator,
    ) -> Tuple[pd.DataFrame, List[ChangeRecord]]:
        """Generate update records."""
        if n == 0 or len(self._current_data) == 0:
            return pd.DataFrame(), []

        # Select rows to update
        n = min(n, len(self._current_data))
        indices = rng.choice(len(self._current_data), size=n, replace=False)

        updates = []
        change_records = []

        # Determine columns to update
        all_columns = [
            c for c in self._current_data.columns
            if c not in [self.id_column, self.timestamp_column, self.version_column]
        ]

        if update_columns:
            columns_to_update = [c for c in update_columns if c in all_columns]
        else:
            columns_to_update = all_columns

        for idx in indices:
            before = self._current_data.iloc[idx].to_dict()
            record_id = str(before[self.id_column])

            # Generate new values for selected columns
            new_row = self._current_data.iloc[idx].copy()

            # Update random columns
            n_cols = rng.integers(1, len(columns_to_update) + 1)
            cols_to_change = rng.choice(
                columns_to_update, size=min(n_cols, len(columns_to_update)), replace=False
            )

            for col in cols_to_change:
                # Generate new value based on column type
                new_row[col] = self._generate_new_value(
                    col, new_row[col], rng
                )

            if self.timestamp_column:
                new_row[self.timestamp_column] = timestamp

            if self.version_column:
                new_row[self.version_column] = before.get(self.version_column, 0) + 1

            after = new_row.to_dict()

            # Update current data
            self._current_data.iloc[idx] = new_row

            updates.append(after)
            change_records.append(
                ChangeRecord(
                    change_type=ChangeType.UPDATE,
                    record_id=record_id,
                    timestamp=timestamp,
                    before=before,
                    after=after,
                    version=after.get(self.version_column, 1) if self.version_column else 1,
                )
            )

        return pd.DataFrame(updates), change_records

    def _generate_deletes(
        self,
        n: int,
        timestamp: datetime,
        rng: np.random.Generator,
    ) -> Tuple[pd.DataFrame, List[ChangeRecord]]:
        """Generate delete records."""
        if n == 0 or len(self._current_data) == 0:
            return pd.DataFrame(), []

        # Select rows to delete
        n = min(n, len(self._current_data))
        indices = rng.choice(len(self._current_data), size=n, replace=False)

        deletes = self._current_data.iloc[indices].copy()
        change_records = []

        for _, row in deletes.iterrows():
            change_records.append(
                ChangeRecord(
                    change_type=ChangeType.DELETE,
                    record_id=str(row[self.id_column]),
                    timestamp=timestamp,
                    before=row.to_dict(),
                    version=row.get(self.version_column, 1) if self.version_column else 1,
                )
            )

        # Remove from current data
        self._current_data = self._current_data.drop(
            self._current_data.index[indices]
        ).reset_index(drop=True)

        return deletes, change_records

    def _generate_new_value(
        self,
        column: str,
        current_value: Any,
        rng: np.random.Generator,
    ) -> Any:
        """Generate a new value for a column."""
        if pd.isna(current_value):
            return current_value

        if isinstance(current_value, (int, np.integer)):
            # Adjust by small random amount
            delta = rng.integers(-10, 11)
            return int(current_value) + delta

        elif isinstance(current_value, (float, np.floating)):
            # Adjust by small percentage
            delta = rng.uniform(-0.1, 0.1)
            return current_value * (1 + delta)

        elif isinstance(current_value, str):
            # For strings, we'd ideally regenerate via the generator
            # For now, just append a suffix
            suffix = str(rng.integers(100, 1000))
            return f"{current_value}_{suffix}"

        return current_value


class ReferentialIntegrityManager:
    """Manages referential integrity across related tables.

    Example:
        >>> manager = ReferentialIntegrityManager()
        >>> manager.add_relationship("orders", "customer_id", "customers", "id")
        >>> delta = manager.generate_consistent_delta(generators)
    """

    def __init__(self) -> None:
        self._relationships: List[Dict[str, str]] = []
        self._primary_keys: Dict[str, Set[str]] = defaultdict(set)

    def add_relationship(
        self,
        child_table: str,
        child_column: str,
        parent_table: str,
        parent_column: str,
    ) -> None:
        """Add a foreign key relationship.

        Args:
            child_table: Table containing foreign key
            child_column: Foreign key column
            parent_table: Referenced table
            parent_column: Referenced column
        """
        self._relationships.append({
            "child_table": child_table,
            "child_column": child_column,
            "parent_table": parent_table,
            "parent_column": parent_column,
        })

    def track_keys(self, table: str, data: pd.DataFrame, id_column: str) -> None:
        """Track primary keys for a table.

        Args:
            table: Table name
            data: DataFrame
            id_column: Primary key column
        """
        self._primary_keys[table] = set(data[id_column].astype(str))

    def validate_delta(
        self,
        table: str,
        delta: DeltaResult,
        id_column: str,
    ) -> Tuple[bool, List[str]]:
        """Validate that delta maintains referential integrity.

        Args:
            table: Table being modified
            delta: Delta changes
            id_column: Primary key column

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Check if deletes would violate FK constraints
        for rel in self._relationships:
            if rel["parent_table"] == table:
                # Deleting from parent table - check children
                deleted_ids = set(
                    str(row[id_column])
                    for _, row in delta.deletes.iterrows()
                )

                # Would need to check child tables for orphans
                if deleted_ids:
                    errors.append(
                        f"Deleting from {table} may create orphans in {rel['child_table']}"
                    )

        return len(errors) == 0, errors

    def get_valid_foreign_keys(
        self,
        parent_table: str,
        n: int,
        rng: np.random.Generator,
    ) -> List[str]:
        """Get valid foreign key values from tracked keys.

        Args:
            parent_table: Parent table name
            n: Number of values needed
            rng: Random generator

        Returns:
            List of valid foreign key values
        """
        if parent_table not in self._primary_keys:
            raise GenesisError(f"Table {parent_table} not tracked")

        valid_keys = list(self._primary_keys[parent_table])
        if not valid_keys:
            raise GenesisError(f"No keys tracked for {parent_table}")

        return list(rng.choice(valid_keys, size=n, replace=True))


class SCDGenerator:
    """Generates Slowly Changing Dimension (SCD) data.

    Supports:
    - Type 1: Overwrite
    - Type 2: Add new row with versioning
    - Type 3: Add previous value columns

    Example:
        >>> generator = SCDGenerator(
        ...     base_generator=my_generator,
        ...     scd_type=2,
        ...     effective_date_col="effective_date",
        ...     end_date_col="end_date"
        ... )
    """

    def __init__(
        self,
        base_generator: BaseGenerator,
        scd_type: int = 2,
        id_column: str = "id",
        effective_date_col: str = "effective_date",
        end_date_col: str = "end_date",
        current_flag_col: str = "is_current",
    ) -> None:
        """Initialize SCD generator.

        Args:
            base_generator: Base generator
            scd_type: SCD type (1, 2, or 3)
            id_column: Natural key column
            effective_date_col: Effective date column
            end_date_col: End date column (SCD2)
            current_flag_col: Current flag column (SCD2)
        """
        self.base_generator = base_generator
        self.scd_type = scd_type
        self.id_column = id_column
        self.effective_date_col = effective_date_col
        self.end_date_col = end_date_col
        self.current_flag_col = current_flag_col

        self._dimension_data: Optional[pd.DataFrame] = None
        self._surrogate_key = 0

    def generate_initial(
        self,
        n_rows: int,
        effective_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate initial dimension data."""
        data = self.base_generator.generate(n_rows)
        effective_date = effective_date or datetime.utcnow()

        if self.scd_type == 2:
            data["surrogate_key"] = range(1, len(data) + 1)
            data[self.effective_date_col] = effective_date
            data[self.end_date_col] = None
            data[self.current_flag_col] = True
            self._surrogate_key = len(data)

        self._dimension_data = data.copy()
        return data

    def apply_changes(
        self,
        changes: List[Dict[str, Any]],
        change_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Apply changes to dimension data.

        Args:
            changes: List of changes with 'id' and updated fields
            change_date: Date of changes

        Returns:
            Updated dimension DataFrame
        """
        if self._dimension_data is None:
            raise GenesisError("Must call generate_initial() first")

        change_date = change_date or datetime.utcnow()

        if self.scd_type == 1:
            return self._apply_scd1_changes(changes)
        elif self.scd_type == 2:
            return self._apply_scd2_changes(changes, change_date)
        elif self.scd_type == 3:
            return self._apply_scd3_changes(changes)
        else:
            raise ValueError(f"Unsupported SCD type: {self.scd_type}")

    def _apply_scd1_changes(
        self, changes: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply Type 1 (overwrite) changes."""
        for change in changes:
            record_id = change.get(self.id_column)
            if record_id is None:
                continue

            mask = self._dimension_data[self.id_column] == record_id
            for col, value in change.items():
                if col != self.id_column:
                    self._dimension_data.loc[mask, col] = value

        return self._dimension_data

    def _apply_scd2_changes(
        self,
        changes: List[Dict[str, Any]],
        change_date: datetime,
    ) -> pd.DataFrame:
        """Apply Type 2 (add row) changes."""
        new_rows = []

        for change in changes:
            record_id = change.get(self.id_column)
            if record_id is None:
                continue

            # Close current record
            mask = (
                (self._dimension_data[self.id_column] == record_id)
                & (self._dimension_data[self.current_flag_col] == True)  # noqa
            )

            if mask.any():
                # Close existing record
                self._dimension_data.loc[mask, self.end_date_col] = change_date
                self._dimension_data.loc[mask, self.current_flag_col] = False

                # Create new record
                old_row = self._dimension_data[mask].iloc[0].to_dict()
                self._surrogate_key += 1

                new_row = {**old_row, **change}
                new_row["surrogate_key"] = self._surrogate_key
                new_row[self.effective_date_col] = change_date
                new_row[self.end_date_col] = None
                new_row[self.current_flag_col] = True

                new_rows.append(new_row)

        if new_rows:
            self._dimension_data = pd.concat(
                [self._dimension_data, pd.DataFrame(new_rows)],
                ignore_index=True
            )

        return self._dimension_data

    def _apply_scd3_changes(
        self, changes: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Apply Type 3 (add column) changes."""
        for change in changes:
            record_id = change.get(self.id_column)
            if record_id is None:
                continue

            mask = self._dimension_data[self.id_column] == record_id

            for col, value in change.items():
                if col == self.id_column:
                    continue

                prev_col = f"prev_{col}"
                if prev_col not in self._dimension_data.columns:
                    self._dimension_data[prev_col] = None

                # Store previous value
                self._dimension_data.loc[mask, prev_col] = (
                    self._dimension_data.loc[mask, col]
                )
                # Update current value
                self._dimension_data.loc[mask, col] = value

        return self._dimension_data
