# Incremental/Delta Generation

Genesis supports incremental synthetic data generation for CDC (Change Data Capture) workflows, slowly changing dimensions, and maintaining referential integrity across updates.

## Overview

| Class | Purpose |
|-------|---------|
| **DeltaGenerator** | Generate incremental changes |
| **ChangeTracker** | Track changes over time |
| **SCDGenerator** | Slowly Changing Dimensions |
| **ReferentialIntegrityManager** | Maintain FK relationships |

## Delta Generation

Generate synthetic deltas (inserts, updates, deletes) for continuous data pipelines:

```python
from genesis import SyntheticGenerator
from genesis.delta import DeltaGenerator

# Create base generator
base_generator = SyntheticGenerator(method="gaussian_copula")
base_generator.fit(training_data)

# Create delta generator
delta_gen = DeltaGenerator(
    base_generator=base_generator,
    id_column="user_id",
    timestamp_column="updated_at",
    version_column="version",
)

# Initial full load
initial_data = delta_gen.generate_full(100000)

# Generate incremental delta
delta = delta_gen.generate_delta(
    insert_ratio=0.10,   # 10% new records
    update_ratio=0.05,   # 5% updates
    delete_ratio=0.02,   # 2% deletes
)

print(f"Inserts: {len(delta.inserts)}")
print(f"Updates: {len(delta.updates)}")
print(f"Deletes: {len(delta.deletes)}")
print(f"Total changes: {delta.total_changes}")
```

### CDC Output Formats

Export changes in standard CDC formats:

```python
# Debezium format
debezium_records = delta.to_cdc_format("debezium")

# Maxwell format
maxwell_records = delta.to_cdc_format("maxwell")

# Raw format
raw_records = [r.to_dict() for r in delta.change_records]
```

#### Debezium Format

```json
{
  "payload": {
    "op": "u",
    "ts_ms": 1705312200000,
    "before": {"user_id": "u123", "name": "Alice", "score": 100},
    "after": {"user_id": "u123", "name": "Alice", "score": 150}
  },
  "schema": {}
}
```

#### Maxwell Format

```json
{
  "type": "update",
  "ts": 1705312200,
  "data": {"user_id": "u123", "name": "Alice", "score": 150},
  "old": {"user_id": "u123", "name": "Alice", "score": 100}
}
```

## Change Tracking

Track changes between data snapshots:

```python
from genesis.delta import ChangeTracker, ChangeType

tracker = ChangeTracker(id_column="user_id")

# Track initial state
tracker.track_initial(initial_data)

# Later, track changes
changes = tracker.track_changes(new_data)

for change in changes:
    if change.change_type == ChangeType.INSERT:
        print(f"New record: {change.record_id}")
    elif change.change_type == ChangeType.UPDATE:
        print(f"Updated: {change.record_id} v{change.version}")
        print(f"  Before: {change.before}")
        print(f"  After: {change.after}")
    elif change.change_type == ChangeType.DELETE:
        print(f"Deleted: {change.record_id}")

# Get current state
current_state = tracker.get_current_state()

# Get change history
history = tracker.get_history(since=datetime(2026, 1, 1))
```

### Track Specific Columns

```python
tracker = ChangeTracker(
    id_column="user_id",
    track_columns=["name", "email", "status"],  # Only track these
)
```

## Slowly Changing Dimensions

Generate data for dimension tables with historical tracking:

### SCD Type 1 (Overwrite)

```python
from genesis.delta import SCDGenerator

generator = SCDGenerator(
    base_generator=my_generator,
    scd_type=1,
    id_column="customer_id",
)

# Initial dimension
dimension = generator.generate_initial(10000)

# Apply changes (overwrites existing records)
changes = [{"customer_id": "c123", "name": "New Name"}]
dimension = generator.apply_changes(changes)
```

### SCD Type 2 (Add Row)

```python
generator = SCDGenerator(
    base_generator=my_generator,
    scd_type=2,
    id_column="customer_id",
    effective_date_col="effective_date",
    end_date_col="end_date",
    current_flag_col="is_current",
)

# Initial dimension with SCD2 columns
dimension = generator.generate_initial(10000)
# Columns: surrogate_key, customer_id, ..., effective_date, end_date, is_current

# Apply changes (creates new row, closes old)
changes = [{"customer_id": "c123", "name": "New Name"}]
dimension = generator.apply_changes(changes)

# Old record: end_date = change_date, is_current = False
# New record: effective_date = change_date, end_date = NULL, is_current = True
```

### SCD Type 3 (Add Column)

```python
generator = SCDGenerator(
    base_generator=my_generator,
    scd_type=3,
    id_column="customer_id",
)

# Apply changes (adds prev_* columns)
changes = [{"customer_id": "c123", "status": "premium"}]
dimension = generator.apply_changes(changes)

# Result: status = "premium", prev_status = "standard"
```

## Referential Integrity

Maintain foreign key relationships across tables:

```python
from genesis.delta import ReferentialIntegrityManager

manager = ReferentialIntegrityManager()

# Define relationships
manager.add_relationship(
    child_table="orders",
    child_column="customer_id",
    parent_table="customers",
    parent_column="id",
)

manager.add_relationship(
    child_table="order_items",
    child_column="order_id",
    parent_table="orders",
    parent_column="id",
)

# Track parent keys
manager.track_keys("customers", customers_df, "id")
manager.track_keys("orders", orders_df, "id")

# Get valid foreign keys for new records
import numpy as np
rng = np.random.default_rng(42)
valid_customer_ids = manager.get_valid_foreign_keys("customers", 100, rng)

# Validate delta maintains integrity
is_valid, errors = manager.validate_delta(
    table="customers",
    delta=delta_result,
    id_column="id",
)

if not is_valid:
    for error in errors:
        print(f"Warning: {error}")
```

## Complete Example

```python
from genesis import SyntheticGenerator
from genesis.delta import DeltaGenerator, ChangeTracker

# Setup
base_gen = SyntheticGenerator(method="gaussian_copula")
base_gen.fit(training_data)

delta_gen = DeltaGenerator(
    base_generator=base_gen,
    id_column="transaction_id",
    timestamp_column="created_at",
)

# Initial load
print("Generating initial dataset...")
initial = delta_gen.generate_full(1_000_000)
print(f"Generated {len(initial)} initial records")

# Simulate daily deltas
for day in range(30):
    print(f"\nDay {day + 1}:")
    
    delta = delta_gen.generate_delta(
        insert_ratio=0.05,
        update_ratio=0.02,
        delete_ratio=0.01,
    )
    
    print(f"  Inserts: {len(delta.inserts)}")
    print(f"  Updates: {len(delta.updates)}")
    print(f"  Deletes: {len(delta.deletes)}")
    
    # Export to Kafka in Debezium format
    for record in delta.to_cdc_format("debezium"):
        kafka_producer.send("transactions", record)
```

## Configuration Reference

### DeltaGenerator

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_generator` | BaseGenerator | Underlying synthetic generator |
| `id_column` | str | Primary key column |
| `timestamp_column` | str | Timestamp column (optional) |
| `version_column` | str | Version tracking column (optional) |

### generate_delta()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `insert_ratio` | float | 0.1 | Ratio of new records |
| `update_ratio` | float | 0.05 | Ratio of updated records |
| `delete_ratio` | float | 0.02 | Ratio of deleted records |
| `update_columns` | List[str] | None | Columns to update |
| `timestamp` | datetime | now | Timestamp for changes |

### ChangeTracker

| Parameter | Type | Description |
|-----------|------|-------------|
| `id_column` | str | Unique identifier column |
| `version_column` | str | Version tracking column |
| `track_columns` | List[str] | Columns to track for changes |

### SCDGenerator

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_generator` | BaseGenerator | Underlying generator |
| `scd_type` | int | SCD type (1, 2, or 3) |
| `id_column` | str | Natural key column |
| `effective_date_col` | str | Effective date column |
| `end_date_col` | str | End date column |
| `current_flag_col` | str | Current flag column |
