"""Tests for Edge Generation module."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genesis.edge_generation import (
    EdgeExporter,
    EdgeGeneratorConfig,
    EdgeGeneratorFactory,
    EdgeRuntime,
    ExportFormat,
    ExportResult,
    ModelOptimizer,
    OptimizationLevel,
    QuantizationConfig,
    QuantizationType,
    RuntimeStats,
    export_to_numpy,
    export_to_onnx,
    load_edge_runtime,
)


class TestEdgeGeneratorConfig:
    """Tests for EdgeGeneratorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EdgeGeneratorConfig(n_features=10)

        assert config.n_features == 10
        assert config.latent_dim == 32
        assert config.hidden_dims == [64, 32]
        assert config.batch_size == 64

    def test_custom_config(self):
        """Test custom configuration."""
        config = EdgeGeneratorConfig(
            n_features=5,
            latent_dim=16,
            hidden_dims=[32],
            column_names=["a", "b", "c", "d", "e"],
        )

        assert config.n_features == 5
        assert config.latent_dim == 16
        assert len(config.column_names) == 5

    def test_to_dict(self):
        """Test configuration serialization."""
        config = EdgeGeneratorConfig(
            n_features=8,
            latent_dim=24,
        )

        data = config.to_dict()

        assert data["n_features"] == 8
        assert data["latent_dim"] == 24


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_quantization(self):
        """Test default quantization config."""
        config = QuantizationConfig()

        assert config.quant_type == QuantizationType.DYNAMIC
        assert config.per_channel is True

    def test_static_quantization(self):
        """Test static quantization config."""
        config = QuantizationConfig(
            quant_type=QuantizationType.STATIC_INT8,
            calibration_data=np.random.randn(100, 10).astype(np.float32),
        )

        assert config.quant_type == QuantizationType.STATIC_INT8
        assert config.calibration_data is not None


class TestModelOptimizer:
    """Tests for ModelOptimizer."""

    def test_estimate_size_reduction_dynamic(self):
        """Test size reduction estimation for dynamic quantization."""
        optimizer = ModelOptimizer()

        original_size = 1000000  # 1 MB
        estimated = optimizer.estimate_size_reduction(
            original_size,
            QuantizationType.DYNAMIC,
        )

        # Dynamic quantization ~70% reduction
        assert estimated < original_size * 0.5

    def test_estimate_size_reduction_float16(self):
        """Test size reduction estimation for FP16."""
        optimizer = ModelOptimizer()

        original_size = 1000000
        estimated = optimizer.estimate_size_reduction(
            original_size,
            QuantizationType.FLOAT16,
        )

        # FP16 should be ~50% of original
        assert estimated == original_size * 0.5


class TestEdgeExporter:
    """Tests for EdgeExporter."""

    def test_export_numpy(self):
        """Test NumPy export."""
        exporter = EdgeExporter()
        config = EdgeGeneratorConfig(
            n_features=5,
            latent_dim=16,
            hidden_dims=[32],
        )

        # Create mock generator with weights
        class MockGenerator:
            def __init__(self):
                self.weights = {
                    "fc0.weight": np.random.randn(32, 16).astype(np.float32),
                    "fc0.bias": np.zeros(32, dtype=np.float32),
                    "output.weight": np.random.randn(5, 32).astype(np.float32),
                    "output.bias": np.zeros(5, dtype=np.float32),
                }

        generator = MockGenerator()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            result = exporter.export_numpy(generator, f.name, config)

            assert result.format == ExportFormat.NUMPY
            assert result.size_bytes > 0
            assert os.path.exists(f.name)

            # Check weights saved correctly
            data = np.load(f.name)
            assert "fc0.weight" in data.files

            os.unlink(f.name)

    def test_create_minimal_onnx_generator(self):
        """Test creating minimal ONNX generator."""
        pytest.importorskip("onnx")

        exporter = EdgeExporter()
        config = EdgeGeneratorConfig(
            n_features=4,
            latent_dim=8,
            hidden_dims=[16],
        )

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            exporter._create_minimal_onnx_generator(f.name, config)

            assert os.path.exists(f.name)
            assert os.path.getsize(f.name) > 0

            os.unlink(f.name)


class TestEdgeRuntime:
    """Tests for EdgeRuntime."""

    @pytest.fixture
    def numpy_model_path(self):
        """Create test NumPy model."""
        config = EdgeGeneratorConfig(
            n_features=4,
            latent_dim=8,
            hidden_dims=[16],
        )

        weights = {
            "fc0.weight": np.random.randn(16, 8).astype(np.float32) * 0.1,
            "fc0.bias": np.zeros(16, dtype=np.float32),
            "output.weight": np.random.randn(4, 16).astype(np.float32) * 0.1,
            "output.bias": np.zeros(4, dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez_compressed(f.name, **weights)

            # Save config
            config_path = f.name.replace(".npz", ".config.json")
            with open(config_path, "w") as cf:
                json.dump(config.to_dict(), cf)

            yield f.name

            os.unlink(f.name)
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_load_numpy_model(self, numpy_model_path):
        """Test loading NumPy model."""
        runtime = EdgeRuntime.load(numpy_model_path)

        assert runtime._weights is not None
        assert runtime._format == ExportFormat.NUMPY

    def test_generate_numpy(self, numpy_model_path):
        """Test generation with NumPy runtime."""
        runtime = EdgeRuntime.load(numpy_model_path)

        data = runtime.generate(n_samples=50, seed=42)

        assert len(data) == 50
        assert data.shape[1] == 4

    def test_generate_deterministic(self, numpy_model_path):
        """Test deterministic generation with seed."""
        runtime = EdgeRuntime.load(numpy_model_path)

        data1 = runtime.generate(n_samples=10, seed=123)
        data2 = runtime.generate(n_samples=10, seed=123)

        pd.testing.assert_frame_equal(data1, data2)

    def test_get_stats(self, numpy_model_path):
        """Test getting runtime statistics."""
        runtime = EdgeRuntime.load(numpy_model_path)

        stats = runtime.get_stats(n_warmup=2, n_benchmark=10)

        assert isinstance(stats, RuntimeStats)
        assert stats.model_size_bytes > 0
        assert stats.latency_ms > 0
        assert stats.throughput > 0

    def test_benchmark(self, numpy_model_path):
        """Test benchmark across batch sizes."""
        runtime = EdgeRuntime.load(numpy_model_path)

        results = runtime.benchmark(batch_sizes=[1, 4, 8])

        assert len(results) == 3
        assert all("batch_size" in r for r in results)
        assert all("latency_mean_ms" in r for r in results)
        assert all("throughput" in r for r in results)


class TestEdgeGeneratorFactory:
    """Tests for EdgeGeneratorFactory."""

    def test_create_minimal_generator(self):
        """Test creating minimal generator."""
        weights, config = EdgeGeneratorFactory.create_minimal_generator(
            n_features=6,
            latent_dim=12,
            hidden_dim=24,
        )

        assert config.n_features == 6
        assert config.latent_dim == 12
        assert "fc0.weight" in weights
        assert weights["fc0.weight"].shape == (24, 12)
        assert weights["output.weight"].shape == (6, 24)


class TestExportResult:
    """Tests for ExportResult."""

    def test_to_dict(self):
        """Test result serialization."""
        result = ExportResult(
            path="/tmp/model.onnx",
            format=ExportFormat.ONNX,
            size_bytes=512000,
            optimization_level=OptimizationLevel.EXTENDED,
        )

        data = result.to_dict()

        assert data["path"] == "/tmp/model.onnx"
        assert data["format"] == "onnx"
        assert data["size_mb"] == 0.5
        assert data["optimization_level"] == "extended"


class TestRuntimeStats:
    """Tests for RuntimeStats."""

    def test_to_dict(self):
        """Test stats serialization."""
        stats = RuntimeStats(
            model_size_bytes=1048576,  # 1 MB
            latency_ms=5.5,
            throughput=180.0,
            memory_peak_mb=2.0,
            platform="onnx",
        )

        data = stats.to_dict()

        assert data["size_mb"] == 1.0
        assert data["latency_ms"] == 5.5
        assert data["throughput"] == 180.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_export_to_numpy(self):
        """Test export_to_numpy function."""
        class MockGen:
            weights = {
                "fc0.weight": np.random.randn(16, 8).astype(np.float32),
                "fc0.bias": np.zeros(16, dtype=np.float32),
                "output.weight": np.random.randn(4, 16).astype(np.float32),
                "output.bias": np.zeros(4, dtype=np.float32),
            }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            result = export_to_numpy(MockGen(), f.name)

            assert result.format == ExportFormat.NUMPY
            assert os.path.exists(f.name)

            os.unlink(f.name)

    def test_load_edge_runtime(self):
        """Test load_edge_runtime function."""
        # Create temp model
        weights = {
            "fc0.weight": np.random.randn(8, 4).astype(np.float32),
            "fc0.bias": np.zeros(8, dtype=np.float32),
            "output.weight": np.random.randn(2, 8).astype(np.float32),
            "output.bias": np.zeros(2, dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez_compressed(f.name, **weights)

            runtime = load_edge_runtime(f.name)

            assert runtime is not None
            assert runtime._weights is not None

            os.unlink(f.name)


@pytest.mark.skipif(
    not pytest.importorskip("onnx", reason="onnx not installed"),
    reason="onnx required",
)
class TestONNXExport:
    """Tests requiring ONNX library."""

    def test_export_onnx_minimal(self):
        """Test exporting minimal ONNX model."""
        exporter = EdgeExporter()
        config = EdgeGeneratorConfig(
            n_features=3,
            latent_dim=6,
            hidden_dims=[12],
        )

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            result = exporter.export_onnx(
                generator=None,  # Will create minimal
                output_path=f.name,
                config=config,
                optimize=False,
                quantize=False,
            )

            assert result.format == ExportFormat.ONNX
            assert os.path.exists(f.name)

            os.unlink(f.name)
            # Clean up metadata
            meta_path = f.name + ".meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)


@pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="onnxruntime required",
)
class TestONNXRuntime:
    """Tests requiring ONNX Runtime."""

    @pytest.fixture
    def onnx_model_path(self):
        """Create test ONNX model."""
        import onnx
        from onnx import helper, numpy_helper, TensorProto

        # Create simple model
        latent_dim = 8
        hidden_dim = 16
        output_dim = 4

        # Weights
        w1 = np.random.randn(hidden_dim, latent_dim).astype(np.float32) * 0.1
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = np.random.randn(output_dim, hidden_dim).astype(np.float32) * 0.1
        b2 = np.zeros(output_dim, dtype=np.float32)

        nodes = [
            helper.make_node(
                "Gemm", ["latent", "w1", "b1"], ["h1"], transB=1
            ),
            helper.make_node("Relu", ["h1"], ["h1_relu"]),
            helper.make_node(
                "Gemm", ["h1_relu", "w2", "b2"], ["output"], transB=1
            ),
        ]

        inputs = [
            helper.make_tensor_value_info("latent", TensorProto.FLOAT, [None, latent_dim])
        ]
        outputs = [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, output_dim])
        ]
        initializers = [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(b2, "b2"),
        ]

        graph = helper.make_graph(nodes, "test", inputs, outputs, initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)

            yield f.name

            os.unlink(f.name)

    def test_load_onnx_model(self, onnx_model_path):
        """Test loading ONNX model."""
        runtime = EdgeRuntime.load(onnx_model_path)

        assert runtime._session is not None
        assert runtime._format == ExportFormat.ONNX

    def test_generate_onnx(self, onnx_model_path):
        """Test generation with ONNX runtime."""
        config = EdgeGeneratorConfig(n_features=4, latent_dim=8)
        runtime = EdgeRuntime.load(onnx_model_path, config=config)

        data = runtime.generate(n_samples=20, seed=42)

        assert len(data) == 20
        assert data.shape[1] == 4
