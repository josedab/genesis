"""Tests for data lineage and provenance tracking."""

import json
import tempfile

import numpy as np
import pandas as pd
import pytest

from genesis.lineage import (
    DataLineage,
    DataManifest,
    LineageBlock,
    LineageChain,
    SourceMetadata,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(100),
            "value": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Create synthetic data for testing."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "id": range(100),
            "value": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )


class TestSourceMetadata:
    """Tests for SourceMetadata."""

    def test_from_dataframe(self, sample_data: pd.DataFrame) -> None:
        """Test creating metadata from DataFrame."""
        metadata = SourceMetadata.from_dataframe(sample_data, "test_source", "Test description")

        assert metadata.name == "test_source"
        assert metadata.n_rows == 100
        assert metadata.n_columns == 3
        assert len(metadata.columns) == 3
        assert metadata.hash_sha256 is not None
        assert len(metadata.hash_sha256) == 64  # SHA-256 hex

    def test_column_stats_extracted(self, sample_data: pd.DataFrame) -> None:
        """Test that column statistics are extracted."""
        metadata = SourceMetadata.from_dataframe(sample_data, "test")

        # Find numeric column info
        value_col = next(c for c in metadata.columns if c["name"] == "value")
        assert "min" in value_col
        assert "max" in value_col
        assert "mean" in value_col


class TestDataLineage:
    """Tests for DataLineage tracking."""

    def test_record_source(self, sample_data: pd.DataFrame) -> None:
        """Test recording source data."""
        lineage = DataLineage(creator="test_user", purpose="testing")
        source_id = lineage.record_source(sample_data, "customers")

        assert source_id is not None
        assert lineage._source is not None
        assert lineage._source.name == "customers"

    def test_create_manifest(self, sample_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
        """Test creating a complete manifest."""

        # Mock generator
        class MockGenerator:
            def get_parameters(self):
                return {"config": {"method": "test"}, "privacy": {}}

        lineage = DataLineage(creator="test", purpose="testing")
        lineage.record_source(sample_data, "source")
        lineage.record_generation(MockGenerator(), synthetic_data, 1.5)

        manifest = lineage.create_manifest()

        assert manifest.source is not None
        assert manifest.generation is not None
        assert manifest.synthetic_hash is not None
        assert manifest.creator == "test"

    def test_manifest_save_load(self, sample_data: pd.DataFrame) -> None:
        """Test saving and loading manifest."""
        lineage = DataLineage()
        lineage.record_source(sample_data, "test")

        class MockGenerator:
            def get_parameters(self):
                return {"config": {}, "privacy": {}}

        lineage.record_generation(MockGenerator(), sample_data, 1.0)
        manifest = lineage.create_manifest()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            manifest.save(f.name)
            loaded = DataManifest.load(f.name)

        assert loaded.manifest_id == manifest.manifest_id
        assert loaded.source.name == manifest.source.name


class TestLineageBlock:
    """Tests for LineageBlock."""

    def test_block_hash_computed(self) -> None:
        """Test that block hash is computed on creation."""
        block = LineageBlock(
            block_id="test-123",
            previous_hash="0" * 64,
            timestamp="2024-01-01T00:00:00",
            action="source",
            data_hash="abc123",
            metadata={"name": "test"},
        )

        assert block.block_hash is not None
        assert len(block.block_hash) == 64

    def test_block_verification(self) -> None:
        """Test block verification."""
        block = LineageBlock(
            block_id="test-123",
            previous_hash="0" * 64,
            timestamp="2024-01-01T00:00:00",
            action="source",
            data_hash="abc123",
            metadata={"name": "test"},
        )

        assert block.verify()

        # Tamper with block
        block.metadata["name"] = "tampered"
        assert not block.verify()

    def test_to_dict_from_dict(self) -> None:
        """Test serialization roundtrip."""
        block = LineageBlock(
            block_id="test-123",
            previous_hash="0" * 64,
            timestamp="2024-01-01T00:00:00",
            action="generation",
            data_hash="def456",
            metadata={"samples": 1000},
        )

        d = block.to_dict()
        restored = LineageBlock.from_dict(d)

        assert restored.block_id == block.block_id
        assert restored.block_hash == block.block_hash
        assert restored.verify()


class TestLineageChain:
    """Tests for LineageChain blockchain-style tracking."""

    def test_empty_chain_is_valid(self) -> None:
        """Test that empty chain verifies."""
        chain = LineageChain()
        assert chain.verify()
        assert len(chain) == 0

    def test_add_source_block(self, sample_data: pd.DataFrame) -> None:
        """Test adding source block."""
        chain = LineageChain()
        block_hash = chain.add_source(sample_data, "customers", "Customer data")

        assert len(chain) == 1
        assert block_hash is not None
        assert chain.verify()

    def test_chain_integrity(self, sample_data: pd.DataFrame) -> None:
        """Test that chain maintains integrity."""
        chain = LineageChain()

        # Add multiple blocks
        chain.add_source(sample_data, "source1")

        class MockGenerator:
            def get_parameters(self):
                return {"config": {"method": "test"}, "privacy": {}}

        chain.add_generation(MockGenerator(), sample_data)
        chain.add_transformation("filter", {"condition": "x > 0"}, sample_data)

        assert len(chain) == 3
        assert chain.verify()

        # Verify chain linkage
        for i in range(1, len(chain._blocks)):
            assert chain._blocks[i].previous_hash == chain._blocks[i - 1].block_hash

    def test_chain_detects_tampering(self, sample_data: pd.DataFrame) -> None:
        """Test that chain detects tampering."""
        chain = LineageChain()
        chain.add_source(sample_data, "test")

        assert chain.verify()

        # Tamper with block
        chain._blocks[0].metadata["name"] = "tampered"
        assert not chain.verify()

    def test_export_and_load(self, sample_data: pd.DataFrame) -> None:
        """Test exporting and loading chain."""
        chain = LineageChain()
        chain.add_source(sample_data, "test_data")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            chain.export(f.name)
            loaded = LineageChain.load(f.name)

        assert len(loaded) == len(chain)
        assert loaded.verify()
        assert loaded._blocks[0].block_hash == chain._blocks[0].block_hash

    def test_get_audit_trail(self, sample_data: pd.DataFrame) -> None:
        """Test getting human-readable audit trail."""
        chain = LineageChain()
        chain.add_source(sample_data, "customers")

        trail = chain.get_audit_trail()

        assert len(trail) == 1
        assert trail[0]["action"] == "source"
        assert trail[0]["name"] == "customers"
        assert "timestamp" in trail[0]

    def test_add_quality_check(self, sample_data: pd.DataFrame) -> None:
        """Test adding quality check block."""
        chain = LineageChain()
        chain.add_source(sample_data, "test")

        class MockReport:
            def to_dict(self):
                return {"overall_score": 0.95, "fidelity": 0.92}

        chain.add_quality_check(MockReport(), passed=True)

        assert len(chain) == 2
        assert chain._blocks[1].action == "quality"
        assert chain._blocks[1].metadata["passed"] is True
        assert chain.verify()


class TestLineageChainInvalidLoads:
    """Tests for handling invalid chain loads."""

    def test_load_tampered_chain_raises(self, sample_data: pd.DataFrame) -> None:
        """Test that loading tampered chain raises error."""
        chain = LineageChain()
        chain.add_source(sample_data, "test")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            chain.export(f.name)

            # Tamper with file
            with open(f.name) as fr:
                data = json.load(fr)
            data["blocks"][0]["metadata"]["name"] = "tampered"
            with open(f.name, "w") as fw:
                json.dump(data, fw)

            with pytest.raises(ValueError, match="failed verification"):
                LineageChain.load(f.name)
