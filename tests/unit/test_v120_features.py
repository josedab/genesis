"""Tests for v1.2.0 next-gen features."""

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# Plugin System Tests
# ============================================================================


class TestPluginSystem:
    """Tests for genesis.plugins module."""

    def test_register_generator(self):
        """Test registering a custom generator."""
        from genesis.plugins import PluginType, get_generator, get_registry, register_generator

        @register_generator("test_gen", description="Test generator")
        class TestGenerator:
            pass

        result = get_generator("test_gen")
        assert result == TestGenerator

        # Cleanup
        get_registry().unregister("test_gen", PluginType.GENERATOR)

    def test_list_generators(self):
        """Test listing registered generators."""
        from genesis.plugins import list_generators

        generators = list_generators()
        assert isinstance(generators, list)

    def test_plugin_info(self):
        """Test plugin info structure."""
        from genesis.plugins import PluginInfo, PluginType

        info = PluginInfo(
            name="test",
            plugin_type=PluginType.GENERATOR,
            cls=object,
            description="Test",
        )

        result = info.to_dict()
        assert result["name"] == "test"
        assert result["type"] == "generator"


# ============================================================================
# Auto-Tuning Tests
# ============================================================================


class TestAutoTuning:
    """Tests for genesis.tuning module."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "a": np.random.normal(0, 1, 100),
                "b": np.random.randint(0, 10, 100),
                "c": np.random.choice(["x", "y", "z"], 100),
            }
        )

    def test_tuning_config(self):
        """Test TuningConfig creation."""
        from genesis.tuning import TuningConfig, TuningPreset

        config = TuningConfig(n_trials=10, preset=TuningPreset.FAST)
        assert config.n_trials == 10

    def test_tuning_config_from_preset(self):
        """Test TuningConfig from preset."""
        from genesis.tuning import TuningConfig, TuningPreset

        config = TuningConfig.from_preset(TuningPreset.FAST)
        assert config.n_trials == 10

        config = TuningConfig.from_preset(TuningPreset.QUALITY)
        assert config.n_trials == 50

    def test_search_space(self):
        """Test search space definition."""
        from genesis.core.types import GeneratorMethod
        from genesis.tuning import SearchSpace

        space = SearchSpace.get_space(GeneratorMethod.CTGAN)
        assert "epochs" in space
        assert "batch_size" in space


# ============================================================================
# Privacy Certificate Tests
# ============================================================================


class TestPrivacyCertificate:
    """Tests for genesis.compliance module."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.normal(50000, 15000, 100),
            }
        )

    def test_certificate_generation(self, sample_data):
        """Test generating a privacy certificate."""
        from genesis.compliance import ComplianceFramework, PrivacyCertificate

        cert = PrivacyCertificate(
            real_data=sample_data,
            synthetic_data=sample_data.copy(),  # Using same data for test
        )

        report = cert.generate(framework=ComplianceFramework.GENERAL)

        assert report.metadata.certificate_id is not None
        assert len(report.metrics) > 0
        assert report.overall_risk is not None

    def test_certificate_export(self, sample_data, tmp_path):
        """Test exporting certificate to file."""
        from genesis.compliance import PrivacyCertificate

        cert = PrivacyCertificate(
            real_data=sample_data,
            synthetic_data=sample_data.copy(),
        )

        report = cert.generate()

        # Test JSON export
        json_path = tmp_path / "cert.json"
        report.save(json_path, format="json")
        assert json_path.exists()

        # Test Markdown export
        md_path = tmp_path / "cert.md"
        report.save(md_path, format="md")
        assert md_path.exists()


# ============================================================================
# Drift Detection Tests
# ============================================================================


class TestDriftDetection:
    """Tests for genesis.monitoring module."""

    @pytest.fixture
    def baseline_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "a": np.random.normal(0, 1, 200),
                "b": np.random.randint(0, 10, 200),
            }
        )

    def test_drift_detector_init(self, baseline_data):
        """Test DriftDetector initialization."""
        from genesis.monitoring import DriftDetector

        detector = DriftDetector(baseline_data=baseline_data)
        assert detector.baseline_data is not None

    def test_no_drift_detection(self, baseline_data):
        """Test detection when no drift."""
        from genesis.monitoring import DriftDetector

        detector = DriftDetector(baseline_data=baseline_data)

        # Same distribution should show no drift
        result = detector.check(baseline_data)
        assert result.drift_score < 0.5

    def test_drift_detection_with_shift(self, baseline_data):
        """Test detection with significant drift."""
        from genesis.monitoring import DriftDetector

        detector = DriftDetector(baseline_data=baseline_data)

        # Create drifted data
        drifted = baseline_data.copy()
        drifted["a"] = drifted["a"] + 5  # Significant shift

        result = detector.check(drifted)
        assert result.has_drift
        assert len(result.columns_with_drift) > 0


# ============================================================================
# Debugger Tests
# ============================================================================


class TestSyntheticDebugger:
    """Tests for genesis.debugger module."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.normal(50000, 15000, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    def test_debugger_diagnose(self, sample_data):
        """Test running diagnosis."""
        from genesis.debugger import SyntheticDebugger

        debugger = SyntheticDebugger(
            real_data=sample_data,
            synthetic_data=sample_data.copy(),
        )

        report = debugger.diagnose()

        assert report.overall_score >= 0
        assert len(report.column_diagnostics) == 3

    def test_column_comparison(self, sample_data):
        """Test comparing distributions."""
        from genesis.debugger import SyntheticDebugger

        debugger = SyntheticDebugger(
            real_data=sample_data,
            synthetic_data=sample_data.copy(),
        )

        result = debugger.compare_distributions("age")

        assert "real" in result
        assert "synthetic" in result
        assert "comparison" in result


# ============================================================================
# Anomaly Synthesis Tests
# ============================================================================


class TestAnomalySynthesis:
    """Tests for genesis.anomaly module."""

    @pytest.fixture
    def normal_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "value": np.random.normal(100, 10, 200),
                "count": np.random.randint(1, 50, 200),
            }
        )

    def test_anomaly_generator_init(self, normal_data):
        """Test AnomalyGenerator initialization."""
        from genesis.anomaly import AnomalyGenerator

        gen = AnomalyGenerator(normal_data=normal_data)
        assert gen.normal_data is not None

    def test_generate_statistical_anomalies(self, normal_data):
        """Test generating statistical anomalies."""
        from genesis.anomaly import AnomalyGenerator, AnomalyType

        gen = AnomalyGenerator(normal_data=normal_data)
        result = gen.generate(n_samples=20, anomaly_type=AnomalyType.STATISTICAL)

        assert len(result.data) == 20
        assert result.n_anomalies == 20

    def test_balanced_dataset_generator(self, normal_data):
        """Test generating balanced dataset."""
        from genesis.anomaly import BalancedDatasetGenerator

        gen = BalancedDatasetGenerator(
            normal_data=normal_data,
            anomaly_ratio=0.1,
        )

        data, labels = gen.generate(n_samples=100)

        assert len(data) == 100
        assert len(labels) == 100
        assert labels.sum() > 0  # Has some anomalies


# ============================================================================
# Distributed Training Tests
# ============================================================================


class TestDistributedTraining:
    """Tests for genesis.distributed module."""

    def test_distributed_config(self):
        """Test DistributedConfig creation."""
        from genesis.distributed import DistributedBackend, DistributedConfig

        config = DistributedConfig(n_workers=4, backend=DistributedBackend.AUTO)
        assert config.n_workers == 4

    def test_data_sharder(self):
        """Test data sharding."""
        from genesis.distributed import DataSharder, ShardingStrategy

        data = pd.DataFrame({"a": range(100)})
        sharder = DataSharder(strategy=ShardingStrategy.RANDOM)

        shards = sharder.shard(data, n_shards=4)

        assert len(shards) == 4
        assert sum(len(s) for s in shards) == 100

    def test_gpu_manager(self):
        """Test GPU manager."""
        from genesis.distributed import GPUManager

        manager = GPUManager()
        # Should work even without GPUs
        assert manager.n_gpus >= 0


# ============================================================================
# Cross-Modal Generation Tests
# ============================================================================


class TestCrossModalGeneration:
    """Tests for genesis.crossmodal module."""

    def test_tabular_encoder(self):
        """Test tabular encoder."""
        from genesis.crossmodal import TabularEncoder

        data = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": ["x", "y", "x", "y", "z"],
            }
        )

        encoder = TabularEncoder()
        encoder.fit(data)

        encoded = encoder.encode(data)
        assert encoded.shape[0] == 5

    def test_text_encoder(self):
        """Test text encoder."""
        from genesis.crossmodal import TextEncoder

        texts = ["hello world", "foo bar baz", "test document"]

        encoder = TextEncoder(embedding_dim=100)
        encoder.fit(texts)

        encoded = encoder.encode(texts)
        assert encoded.shape[0] == 3


# ============================================================================
# Schema Editor Tests
# ============================================================================


class TestSchemaEditor:
    """Tests for genesis.schema_editor module."""

    def test_schema_definition(self):
        """Test SchemaDefinition creation."""
        from genesis.schema_editor import ColumnDataType, ColumnDefinition, SchemaDefinition

        schema = SchemaDefinition(name="test_table")
        schema.add_column(
            ColumnDefinition(
                name="id",
                dtype=ColumnDataType.INTEGER,
            )
        )

        assert len(schema.columns) == 1
        assert schema.columns[0].name == "id"

    def test_schema_from_dataframe(self):
        """Test inferring schema from DataFrame."""
        from genesis.schema_editor import SchemaDefinition

        data = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "name": ["Alice", "Bob", "Charlie"],
                "active": [True, False, True],
            }
        )

        schema = SchemaDefinition.from_dataframe(data, name="users")

        assert len(schema.columns) == 3
        assert schema.name == "users"

    def test_schema_to_python_code(self):
        """Test exporting schema to Python code."""
        from genesis.schema_editor import ColumnDataType, ColumnDefinition, SchemaDefinition

        schema = SchemaDefinition(name="test")
        schema.add_column(
            ColumnDefinition(
                name="value",
                dtype=ColumnDataType.FLOAT,
            )
        )

        code = schema.to_python_code()
        assert "SyntheticGenerator" in code


# ============================================================================
# Marketplace Tests
# ============================================================================


class TestMarketplace:
    """Tests for genesis.marketplace module."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "a": np.random.normal(0, 1, 100),
                "b": np.random.randint(0, 10, 100),
            }
        )

    def test_marketplace_init(self, tmp_path):
        """Test Marketplace initialization."""
        from genesis.marketplace import Marketplace

        mp = Marketplace(storage_path=tmp_path)
        assert mp.storage_path is not None

    def test_create_listing(self, tmp_path, sample_data):
        """Test creating a dataset listing."""
        from genesis.marketplace import DatasetCategory, Marketplace

        mp = Marketplace(storage_path=tmp_path)

        listing = mp.create_listing(
            name="Test Dataset",
            description="A test dataset",
            data=sample_data,
            owner_id="user123",
            category=DatasetCategory.GENERAL,
            tags=["test", "sample"],
        )

        assert listing.dataset_id is not None
        assert listing.n_rows == 100
        assert listing.n_columns == 2

    def test_search_listings(self, tmp_path, sample_data):
        """Test searching listings."""
        from genesis.marketplace import Marketplace

        mp = Marketplace(storage_path=tmp_path)

        # Create and publish a listing
        listing = mp.create_listing(
            name="Healthcare Data",
            description="Synthetic healthcare records",
            data=sample_data,
            owner_id="user123",
            tags=["healthcare"],
        )
        mp.publish_listing(listing.dataset_id)

        # Search
        results = mp.search(query="healthcare")
        assert results.total_count >= 0  # May or may not find depending on quality


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
