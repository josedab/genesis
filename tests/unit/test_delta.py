"""Tests for incremental/delta generation."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from genesis.delta import (
    ChangeRecord,
    ChangeTracker,
    ChangeType,
    DeltaResult,
    ReferentialIntegrityManager,
    SCDGenerator,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    return pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4", "u5"],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [100, 200, 300, 400, 500],
    })


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert ChangeType.INSERT.value == "I"
        assert ChangeType.UPDATE.value == "U"
        assert ChangeType.DELETE.value == "D"
        assert ChangeType.UPSERT.value == "X"


class TestChangeRecord:
    """Tests for ChangeRecord dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        record = ChangeRecord(
            change_type=ChangeType.INSERT,
            record_id="rec_001",
            timestamp=datetime(2026, 1, 15, 10, 30, 0),
            after={"id": "rec_001", "name": "Test"},
        )

        result = record.to_dict()

        assert result["op"] == "I"
        assert result["id"] == "rec_001"
        assert result["after"]["name"] == "Test"


class TestDeltaResult:
    """Tests for DeltaResult dataclass."""

    def test_total_changes(self) -> None:
        """Test total changes calculation."""
        result = DeltaResult(
            inserts=pd.DataFrame({"id": [1, 2, 3]}),
            updates=pd.DataFrame({"id": [4, 5]}),
            deletes=pd.DataFrame({"id": [6]}),
            change_records=[],
        )

        assert result.total_changes == 6

    def test_to_cdc_format_debezium(self) -> None:
        """Test Debezium CDC format."""
        records = [
            ChangeRecord(
                change_type=ChangeType.INSERT,
                record_id="1",
                timestamp=datetime.utcnow(),
                after={"id": "1", "name": "Test"},
            )
        ]

        result = DeltaResult(
            inserts=pd.DataFrame(),
            updates=pd.DataFrame(),
            deletes=pd.DataFrame(),
            change_records=records,
        )

        cdc = result.to_cdc_format("debezium")

        assert len(cdc) == 1
        assert "payload" in cdc[0]
        assert cdc[0]["payload"]["op"] == "i"

    def test_to_cdc_format_maxwell(self) -> None:
        """Test Maxwell CDC format."""
        records = [
            ChangeRecord(
                change_type=ChangeType.INSERT,
                record_id="1",
                timestamp=datetime.utcnow(),
                after={"id": "1", "name": "Test"},
            )
        ]

        result = DeltaResult(
            inserts=pd.DataFrame(),
            updates=pd.DataFrame(),
            deletes=pd.DataFrame(),
            change_records=records,
        )

        cdc = result.to_cdc_format("maxwell")

        assert len(cdc) == 1
        assert cdc[0]["type"] == "insert"


class TestChangeTracker:
    """Tests for ChangeTracker."""

    def test_track_initial(self, sample_data: pd.DataFrame) -> None:
        """Test tracking initial data."""
        tracker = ChangeTracker(id_column="user_id")
        tracker.track_initial(sample_data)

        state = tracker.get_current_state()
        assert len(state) == 5

    def test_detect_inserts(self, sample_data: pd.DataFrame) -> None:
        """Test detecting insertions."""
        tracker = ChangeTracker(id_column="user_id")
        tracker.track_initial(sample_data)

        # Add a new record
        new_data = pd.concat([
            sample_data,
            pd.DataFrame({"user_id": ["u6"], "name": ["Frank"], "score": [600]}),
        ], ignore_index=True)

        changes = tracker.track_changes(new_data)

        inserts = [c for c in changes if c.change_type == ChangeType.INSERT]
        assert len(inserts) == 1
        assert inserts[0].record_id == "u6"

    def test_detect_updates(self, sample_data: pd.DataFrame) -> None:
        """Test detecting updates."""
        tracker = ChangeTracker(id_column="user_id")
        tracker.track_initial(sample_data)

        # Modify a record
        new_data = sample_data.copy()
        new_data.loc[new_data["user_id"] == "u1", "name"] = "Alice Updated"

        changes = tracker.track_changes(new_data)

        updates = [c for c in changes if c.change_type == ChangeType.UPDATE]
        assert len(updates) == 1
        assert updates[0].record_id == "u1"
        assert updates[0].before["name"] == "Alice"
        assert updates[0].after["name"] == "Alice Updated"

    def test_detect_deletes(self, sample_data: pd.DataFrame) -> None:
        """Test detecting deletions."""
        tracker = ChangeTracker(id_column="user_id")
        tracker.track_initial(sample_data)

        # Remove a record
        new_data = sample_data[sample_data["user_id"] != "u3"]

        changes = tracker.track_changes(new_data)

        deletes = [c for c in changes if c.change_type == ChangeType.DELETE]
        assert len(deletes) == 1
        assert deletes[0].record_id == "u3"

    def test_version_tracking(self, sample_data: pd.DataFrame) -> None:
        """Test version number tracking."""
        tracker = ChangeTracker(id_column="user_id")
        tracker.track_initial(sample_data)

        # First update
        new_data = sample_data.copy()
        new_data.loc[new_data["user_id"] == "u1", "score"] = 150
        changes1 = tracker.track_changes(new_data)

        assert changes1[0].version == 2

        # Second update
        new_data.loc[new_data["user_id"] == "u1", "score"] = 200
        changes2 = tracker.track_changes(new_data)

        assert changes2[0].version == 3

    def test_get_history(self, sample_data: pd.DataFrame) -> None:
        """Test getting change history."""
        tracker = ChangeTracker(id_column="user_id")
        tracker.track_initial(sample_data)

        new_data = sample_data[sample_data["user_id"] != "u1"]
        tracker.track_changes(new_data)

        history = tracker.get_history()
        assert len(history) == 1


class TestReferentialIntegrityManager:
    """Tests for ReferentialIntegrityManager."""

    def test_add_relationship(self) -> None:
        """Test adding relationships."""
        manager = ReferentialIntegrityManager()

        manager.add_relationship(
            child_table="orders",
            child_column="customer_id",
            parent_table="customers",
            parent_column="id",
        )

        assert len(manager._relationships) == 1

    def test_track_keys(self) -> None:
        """Test tracking primary keys."""
        manager = ReferentialIntegrityManager()

        data = pd.DataFrame({"id": ["c1", "c2", "c3"], "name": ["A", "B", "C"]})
        manager.track_keys("customers", data, "id")

        assert "customers" in manager._primary_keys
        assert "c1" in manager._primary_keys["customers"]

    def test_get_valid_foreign_keys(self) -> None:
        """Test getting valid foreign keys."""
        manager = ReferentialIntegrityManager()

        data = pd.DataFrame({"id": ["c1", "c2", "c3"], "name": ["A", "B", "C"]})
        manager.track_keys("customers", data, "id")

        np.random.seed(42)
        fks = manager.get_valid_foreign_keys("customers", 5, np.random.default_rng(42))

        assert len(fks) == 5
        assert all(fk in ["c1", "c2", "c3"] for fk in fks)


class TestSCDGenerator:
    """Tests for SCDGenerator."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock base generator."""
        class MockGenerator:
            def generate(self, n: int) -> pd.DataFrame:
                return pd.DataFrame({
                    "id": [f"id_{i}" for i in range(n)],
                    "name": [f"Name_{i}" for i in range(n)],
                    "value": list(range(n)),
                })

        return MockGenerator()

    def test_generate_initial_scd2(self, mock_generator) -> None:
        """Test initial SCD Type 2 generation."""
        generator = SCDGenerator(
            base_generator=mock_generator,
            scd_type=2,
            id_column="id",
        )

        data = generator.generate_initial(5)

        assert "surrogate_key" in data.columns
        assert "effective_date" in data.columns
        assert "end_date" in data.columns
        assert "is_current" in data.columns
        assert all(data["is_current"])

    def test_apply_scd2_changes(self, mock_generator) -> None:
        """Test applying SCD Type 2 changes."""
        generator = SCDGenerator(
            base_generator=mock_generator,
            scd_type=2,
            id_column="id",
        )

        generator.generate_initial(3)

        changes = [{"id": "id_0", "value": 999}]
        result = generator.apply_changes(changes)

        # Should have 4 records now (3 original + 1 new version)
        assert len(result) == 4

        # Old record should be closed
        old_record = result[(result["id"] == "id_0") & (result["is_current"] == False)]  # noqa
        assert len(old_record) == 1
        assert old_record.iloc[0]["end_date"] is not None

        # New record should be current
        new_record = result[(result["id"] == "id_0") & (result["is_current"] == True)]  # noqa
        assert len(new_record) == 1
        assert new_record.iloc[0]["value"] == 999

    def test_apply_scd1_changes(self, mock_generator) -> None:
        """Test applying SCD Type 1 changes (overwrite)."""
        generator = SCDGenerator(
            base_generator=mock_generator,
            scd_type=1,
            id_column="id",
        )

        generator.generate_initial(3)

        changes = [{"id": "id_0", "value": 999}]
        result = generator.apply_changes(changes)

        # Should still have 3 records (overwritten)
        assert len(result) == 3

        updated = result[result["id"] == "id_0"]
        assert updated.iloc[0]["value"] == 999

    def test_apply_scd3_changes(self, mock_generator) -> None:
        """Test applying SCD Type 3 changes (add column)."""
        generator = SCDGenerator(
            base_generator=mock_generator,
            scd_type=3,
            id_column="id",
        )

        generator.generate_initial(3)

        changes = [{"id": "id_0", "value": 999}]
        result = generator.apply_changes(changes)

        # Should have prev_value column
        assert "prev_value" in result.columns

        updated = result[result["id"] == "id_0"]
        assert updated.iloc[0]["value"] == 999
        assert updated.iloc[0]["prev_value"] == 0  # Original value
