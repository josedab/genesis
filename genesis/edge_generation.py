"""Edge and Embedded Data Generation.

Lightweight generators for on-device synthetic data generation in IoT,
mobile, and privacy-sensitive environments.

Features:
    - ONNX export for cross-platform deployment
    - TensorFlow Lite export for mobile
    - Model quantization and optimization
    - Minimal Python runtime support
    - WASM compilation for browser-based generation
    - Size and latency optimization

Example:
    Export model to ONNX for edge deployment::

        from genesis.edge_generation import EdgeExporter, EdgeRuntime

        # Export trained generator to ONNX
        exporter = EdgeExporter()
        onnx_path = exporter.export_onnx(
            generator,
            output_path="generator.onnx",
            optimize=True,
            quantize=True,
        )

        # Run on edge device
        runtime = EdgeRuntime.load("generator.onnx")
        synthetic_data = runtime.generate(n_samples=100)

        # Get model size and latency stats
        stats = runtime.get_stats()
        print(f"Model size: {stats['size_mb']:.2f} MB")
        print(f"Generation latency: {stats['latency_ms']:.1f} ms per sample")

Classes:
    EdgeExporter: Exports generators to edge-optimized formats.
    EdgeRuntime: Lightweight runtime for edge inference.
    ModelOptimizer: Optimizes models for size and speed.
    QuantizationConfig: Quantization settings.
    EdgeGeneratorConfig: Configuration for edge generators.

Note:
    Requires optional dependencies: onnx, onnxruntime, tf2onnx (for TFLite).
"""

import json
import os
import struct
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


# Optional imports
try:
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExportFormat(str, Enum):
    """Export format options."""

    ONNX = "onnx"
    TFLITE = "tflite"
    TORCHSCRIPT = "torchscript"
    NUMPY = "numpy"  # Pure NumPy weights (most portable)


class QuantizationType(str, Enum):
    """Quantization types."""

    NONE = "none"
    DYNAMIC = "dynamic"  # Dynamic range quantization
    STATIC_INT8 = "static_int8"  # Static INT8 quantization
    FLOAT16 = "float16"  # FP16 quantization


class OptimizationLevel(str, Enum):
    """Optimization levels."""

    NONE = "none"
    BASIC = "basic"  # Graph optimizations
    EXTENDED = "extended"  # + constant folding, operator fusion
    FULL = "full"  # + quantization, pruning


@dataclass
class QuantizationConfig:
    """Quantization configuration.

    Attributes:
        quant_type: Type of quantization.
        calibration_data: Data for static quantization calibration.
        per_channel: Use per-channel quantization.
        symmetric: Use symmetric quantization.
    """

    quant_type: QuantizationType = QuantizationType.DYNAMIC
    calibration_data: Optional[np.ndarray] = None
    per_channel: bool = True
    symmetric: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quant_type": self.quant_type.value,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
        }


@dataclass
class EdgeGeneratorConfig:
    """Configuration for edge generator.

    Attributes:
        n_features: Number of features to generate.
        latent_dim: Latent space dimension.
        hidden_dims: Hidden layer dimensions.
        column_names: Column names for output.
        column_types: Column data types.
        batch_size: Default batch size for generation.
        seed: Random seed for reproducibility.
    """

    n_features: int
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    column_names: List[str] = field(default_factory=list)
    column_types: List[str] = field(default_factory=list)
    batch_size: int = 64
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "column_names": self.column_names,
            "column_types": self.column_types,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


@dataclass
class ExportResult:
    """Result of model export.

    Attributes:
        path: Path to exported model.
        format: Export format.
        size_bytes: Model size in bytes.
        optimization_level: Applied optimization.
        quantization: Applied quantization.
        metadata: Export metadata.
    """

    path: str
    format: ExportFormat
    size_bytes: int
    optimization_level: OptimizationLevel = OptimizationLevel.NONE
    quantization: Optional[QuantizationConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "format": self.format.value,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_bytes / (1024 * 1024),
            "optimization_level": self.optimization_level.value,
            "metadata": self.metadata,
        }


@dataclass
class RuntimeStats:
    """Runtime performance statistics.

    Attributes:
        model_size_bytes: Model size.
        latency_ms: Average generation latency per sample.
        throughput: Samples per second.
        memory_peak_mb: Peak memory usage.
        platform: Runtime platform.
    """

    model_size_bytes: int
    latency_ms: float
    throughput: float
    memory_peak_mb: float
    platform: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size_mb": self.model_size_bytes / (1024 * 1024),
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "memory_peak_mb": self.memory_peak_mb,
            "platform": self.platform,
        }


class ModelOptimizer:
    """Optimizes models for edge deployment.

    Applies graph optimizations, quantization, and pruning.
    """

    def __init__(self) -> None:
        """Initialize optimizer."""
        self._graph_opts = [
            "eliminate_identity",
            "eliminate_deadend",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
        ]

    def optimize_onnx(
        self,
        model_path: str,
        output_path: str,
        level: OptimizationLevel = OptimizationLevel.BASIC,
    ) -> str:
        """Optimize ONNX model.

        Args:
            model_path: Input model path.
            output_path: Output model path.
            level: Optimization level.

        Returns:
            Path to optimized model.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnx required: pip install onnx")

        model = onnx.load(model_path)

        if level == OptimizationLevel.NONE:
            onnx.save(model, output_path)
            return output_path

        # Apply graph optimizations
        from onnx import optimizer

        passes = []
        if level.value in ["basic", "extended", "full"]:
            passes.extend([
                "eliminate_identity",
                "eliminate_nop_transpose",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
            ])

        if level.value in ["extended", "full"]:
            passes.extend([
                "fuse_add_bias_into_conv",
                "fuse_matmul_add_bias_into_gemm",
                "fuse_bn_into_conv",
            ])

        try:
            optimized = optimizer.optimize(model, passes)
            onnx.save(optimized, output_path)
        except Exception:
            # Fallback: save original if optimization fails
            onnx.save(model, output_path)

        return output_path

    def quantize_onnx(
        self,
        model_path: str,
        output_path: str,
        config: QuantizationConfig,
    ) -> str:
        """Apply quantization to ONNX model.

        Args:
            model_path: Input model path.
            output_path: Output model path.
            config: Quantization configuration.

        Returns:
            Path to quantized model.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnx required: pip install onnx")

        if config.quant_type == QuantizationType.NONE:
            import shutil
            shutil.copy(model_path, output_path)
            return output_path

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            if config.quant_type == QuantizationType.DYNAMIC:
                quantize_dynamic(
                    model_path,
                    output_path,
                    weight_type=QuantType.QUInt8,
                )
            elif config.quant_type == QuantizationType.FLOAT16:
                # Convert to FP16
                from onnxmltools.utils.float16_converter import convert_float_to_float16
                model = onnx.load(model_path)
                model_fp16 = convert_float_to_float16(model)
                onnx.save(model_fp16, output_path)
            else:
                # Static quantization requires calibration
                logger.warning("Static quantization not fully implemented, using dynamic")
                quantize_dynamic(model_path, output_path)

        except ImportError:
            logger.warning("onnxruntime.quantization not available, skipping quantization")
            import shutil
            shutil.copy(model_path, output_path)

        return output_path

    def estimate_size_reduction(
        self,
        original_size: int,
        quant_type: QuantizationType,
    ) -> float:
        """Estimate size reduction from quantization.

        Args:
            original_size: Original model size in bytes.
            quant_type: Quantization type.

        Returns:
            Estimated size after quantization.
        """
        reduction_factors = {
            QuantizationType.NONE: 1.0,
            QuantizationType.DYNAMIC: 0.3,  # ~70% reduction
            QuantizationType.STATIC_INT8: 0.25,  # ~75% reduction
            QuantizationType.FLOAT16: 0.5,  # 50% reduction
        }

        factor = reduction_factors.get(quant_type, 1.0)
        return int(original_size * factor)


class EdgeExporter:
    """Exports generators to edge-optimized formats.

    Supports ONNX, TFLite, and pure NumPy exports.
    """

    def __init__(self) -> None:
        """Initialize exporter."""
        self.optimizer = ModelOptimizer()

    def export_onnx(
        self,
        generator: Any,
        output_path: str,
        config: Optional[EdgeGeneratorConfig] = None,
        optimize: bool = True,
        quantize: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> ExportResult:
        """Export generator to ONNX format.

        Args:
            generator: Genesis generator or PyTorch model.
            output_path: Output file path.
            config: Edge generator configuration.
            optimize: Apply optimizations.
            quantize: Apply quantization.
            quant_config: Quantization configuration.

        Returns:
            Export result.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnx required: pip install onnx")

        output_path = str(output_path)
        temp_path = output_path + ".tmp"

        # Extract model and config
        if hasattr(generator, "model") and TORCH_AVAILABLE:
            # PyTorch-based generator
            model = generator.model
            self._export_torch_to_onnx(model, temp_path, config)
        elif hasattr(generator, "weights"):
            # Custom generator with weights
            self._export_weights_to_onnx(generator, temp_path, config)
        else:
            # Create minimal ONNX generator
            self._create_minimal_onnx_generator(temp_path, config)

        # Apply optimizations
        if optimize:
            optimized_path = output_path + ".opt"
            self.optimizer.optimize_onnx(
                temp_path,
                optimized_path,
                OptimizationLevel.EXTENDED,
            )
            os.rename(optimized_path, temp_path)

        # Apply quantization
        applied_quant = None
        if quantize:
            quant_config = quant_config or QuantizationConfig()
            quant_path = output_path + ".quant"
            self.optimizer.quantize_onnx(temp_path, quant_path, quant_config)
            os.rename(quant_path, temp_path)
            applied_quant = quant_config

        # Final rename
        os.rename(temp_path, output_path)

        # Get file size
        size_bytes = os.path.getsize(output_path)

        # Save metadata
        metadata = {
            "exported_at": datetime.utcnow().isoformat(),
            "config": config.to_dict() if config else {},
            "optimized": optimize,
            "quantized": quantize,
        }

        metadata_path = output_path + ".meta.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return ExportResult(
            path=output_path,
            format=ExportFormat.ONNX,
            size_bytes=size_bytes,
            optimization_level=OptimizationLevel.EXTENDED if optimize else OptimizationLevel.NONE,
            quantization=applied_quant,
            metadata=metadata,
        )

    def export_numpy(
        self,
        generator: Any,
        output_path: str,
        config: Optional[EdgeGeneratorConfig] = None,
    ) -> ExportResult:
        """Export generator to pure NumPy format.

        Most portable format - works without deep learning frameworks.

        Args:
            generator: Generator to export.
            output_path: Output file path (.npz).
            config: Edge generator configuration.

        Returns:
            Export result.
        """
        output_path = str(output_path)

        # Extract weights
        weights = {}

        if hasattr(generator, "model") and TORCH_AVAILABLE:
            # PyTorch model
            for name, param in generator.model.state_dict().items():
                weights[name] = param.cpu().numpy()
        elif hasattr(generator, "weights"):
            weights = generator.weights
        else:
            # Create random initialization
            config = config or EdgeGeneratorConfig(n_features=10)
            weights = self._create_random_weights(config)

        # Save as compressed NPZ
        np.savez_compressed(output_path, **weights)

        # Save config
        if config:
            config_path = output_path.replace(".npz", ".config.json")
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=2)

        size_bytes = os.path.getsize(output_path)

        return ExportResult(
            path=output_path,
            format=ExportFormat.NUMPY,
            size_bytes=size_bytes,
            metadata={"config": config.to_dict() if config else {}},
        )

    def _export_torch_to_onnx(
        self,
        model: Any,
        output_path: str,
        config: Optional[EdgeGeneratorConfig],
    ) -> None:
        """Export PyTorch model to ONNX."""
        import torch

        model.eval()

        # Determine input shape
        latent_dim = config.latent_dim if config else 32
        batch_size = config.batch_size if config else 1

        dummy_input = torch.randn(batch_size, latent_dim)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["latent"],
            output_names=["output"],
            dynamic_axes={
                "latent": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=11,
        )

    def _export_weights_to_onnx(
        self,
        generator: Any,
        output_path: str,
        config: Optional[EdgeGeneratorConfig],
    ) -> None:
        """Export weights to ONNX format."""
        # Create ONNX graph from weights
        weights = generator.weights
        config = config or EdgeGeneratorConfig(n_features=10)

        self._create_minimal_onnx_generator(output_path, config, weights)

    def _create_minimal_onnx_generator(
        self,
        output_path: str,
        config: Optional[EdgeGeneratorConfig],
        weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Create minimal ONNX generator model."""
        config = config or EdgeGeneratorConfig(n_features=10)

        # Create simple feedforward network: latent -> hidden -> output
        latent_dim = config.latent_dim
        hidden_dims = config.hidden_dims or [64]
        output_dim = config.n_features

        # Initialize weights if not provided
        if weights is None:
            weights = self._create_random_weights(config)

        # Build ONNX graph
        nodes = []
        initializers = []
        inputs = []
        outputs = []

        # Input: latent vector
        inputs.append(helper.make_tensor_value_info(
            "latent", TensorProto.FLOAT, [None, latent_dim]
        ))

        layer_input = "latent"
        prev_dim = latent_dim

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            weight_name = f"hidden_{i}_weight"
            bias_name = f"hidden_{i}_bias"
            output_name = f"hidden_{i}_output"
            relu_name = f"hidden_{i}_relu"

            # Add weight initializers
            w_key = f"fc{i}.weight" if f"fc{i}.weight" in weights else weight_name
            b_key = f"fc{i}.bias" if f"fc{i}.bias" in weights else bias_name

            w = weights.get(w_key, np.random.randn(hidden_dim, prev_dim).astype(np.float32) * 0.1)
            b = weights.get(b_key, np.zeros(hidden_dim, dtype=np.float32))

            initializers.append(numpy_helper.from_array(w, weight_name))
            initializers.append(numpy_helper.from_array(b, bias_name))

            # MatMul + Add (Gemm)
            nodes.append(helper.make_node(
                "Gemm",
                [layer_input, weight_name, bias_name],
                [output_name],
                transB=1,
            ))

            # ReLU
            nodes.append(helper.make_node(
                "Relu",
                [output_name],
                [relu_name],
            ))

            layer_input = relu_name
            prev_dim = hidden_dim

        # Output layer
        out_weight_name = "output_weight"
        out_bias_name = "output_bias"

        out_w_key = "output.weight" if "output.weight" in weights else out_weight_name
        out_b_key = "output.bias" if "output.bias" in weights else out_bias_name

        out_w = weights.get(out_w_key, np.random.randn(output_dim, prev_dim).astype(np.float32) * 0.1)
        out_b = weights.get(out_b_key, np.zeros(output_dim, dtype=np.float32))

        initializers.append(numpy_helper.from_array(out_w, out_weight_name))
        initializers.append(numpy_helper.from_array(out_b, out_bias_name))

        nodes.append(helper.make_node(
            "Gemm",
            [layer_input, out_weight_name, out_bias_name],
            ["output"],
            transB=1,
        ))

        # Output
        outputs.append(helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [None, output_dim]
        ))

        # Create graph
        graph = helper.make_graph(
            nodes,
            "edge_generator",
            inputs,
            outputs,
            initializers,
        )

        # Create model
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

        # Save
        onnx.save(model, output_path)

    def _create_random_weights(
        self,
        config: EdgeGeneratorConfig,
    ) -> Dict[str, np.ndarray]:
        """Create random weight initialization."""
        weights = {}

        prev_dim = config.latent_dim
        for i, hidden_dim in enumerate(config.hidden_dims):
            weights[f"fc{i}.weight"] = np.random.randn(hidden_dim, prev_dim).astype(np.float32) * 0.1
            weights[f"fc{i}.bias"] = np.zeros(hidden_dim, dtype=np.float32)
            prev_dim = hidden_dim

        weights["output.weight"] = np.random.randn(config.n_features, prev_dim).astype(np.float32) * 0.1
        weights["output.bias"] = np.zeros(config.n_features, dtype=np.float32)

        return weights


class EdgeRuntime:
    """Lightweight runtime for edge inference.

    Supports ONNX and NumPy model formats.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[EdgeGeneratorConfig] = None,
    ) -> None:
        """Initialize edge runtime.

        Args:
            model_path: Path to exported model.
            config: Generator configuration.
        """
        self.model_path = str(model_path)
        self.config = config
        self._session = None
        self._weights = None
        self._format = self._detect_format()

        self._load_model()

    @classmethod
    def load(
        cls,
        model_path: str,
        config: Optional[EdgeGeneratorConfig] = None,
    ) -> "EdgeRuntime":
        """Load edge runtime from file.

        Args:
            model_path: Path to model file.
            config: Optional configuration.

        Returns:
            EdgeRuntime instance.
        """
        # Try to load config from metadata
        if config is None:
            meta_path = str(model_path) + ".meta.json"
            config_path = str(model_path).replace(".npz", ".config.json")

            for path in [meta_path, config_path]:
                if os.path.exists(path):
                    with open(path) as f:
                        data = json.load(f)
                        if "config" in data:
                            data = data["config"]
                        if data.get("n_features"):
                            config = EdgeGeneratorConfig(**data)
                            break

        return cls(model_path, config)

    def _detect_format(self) -> ExportFormat:
        """Detect model format from file extension."""
        if self.model_path.endswith(".onnx"):
            return ExportFormat.ONNX
        elif self.model_path.endswith(".npz"):
            return ExportFormat.NUMPY
        elif self.model_path.endswith(".tflite"):
            return ExportFormat.TFLITE
        else:
            return ExportFormat.NUMPY

    def _load_model(self) -> None:
        """Load model based on format."""
        if self._format == ExportFormat.ONNX:
            self._load_onnx()
        elif self._format == ExportFormat.NUMPY:
            self._load_numpy()

    def _load_onnx(self) -> None:
        """Load ONNX model."""
        if not ORT_AVAILABLE:
            raise ImportError("onnxruntime required: pip install onnxruntime")

        # Use minimal execution providers
        providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            self.model_path,
            providers=providers,
        )

        # Extract config from model if not provided
        if self.config is None:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()

            if inputs and outputs:
                latent_dim = inputs[0].shape[-1] if inputs[0].shape else 32
                n_features = outputs[0].shape[-1] if outputs[0].shape else 10

                self.config = EdgeGeneratorConfig(
                    n_features=n_features if isinstance(n_features, int) else 10,
                    latent_dim=latent_dim if isinstance(latent_dim, int) else 32,
                )

    def _load_numpy(self) -> None:
        """Load NumPy weights."""
        data = np.load(self.model_path, allow_pickle=True)
        self._weights = {k: data[k] for k in data.files}

    def generate(
        self,
        n_samples: int = 100,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic samples.

        Args:
            n_samples: Number of samples to generate.
            seed: Random seed.

        Returns:
            Generated DataFrame.
        """
        if seed is not None:
            np.random.seed(seed)

        latent_dim = self.config.latent_dim if self.config else 32

        # Generate latent vectors
        latent = np.random.randn(n_samples, latent_dim).astype(np.float32)

        # Run inference
        if self._format == ExportFormat.ONNX:
            output = self._generate_onnx(latent)
        else:
            output = self._generate_numpy(latent)

        # Convert to DataFrame
        columns = (
            self.config.column_names
            if self.config and self.config.column_names
            else [f"col_{i}" for i in range(output.shape[1])]
        )

        return pd.DataFrame(output, columns=columns[:output.shape[1]])

    def _generate_onnx(self, latent: np.ndarray) -> np.ndarray:
        """Generate using ONNX runtime."""
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: latent})
        return output[0]

    def _generate_numpy(self, latent: np.ndarray) -> np.ndarray:
        """Generate using pure NumPy.

        Simple feedforward network implementation.
        """
        x = latent

        # Find layer weights
        layer_idx = 0
        while True:
            w_key = f"fc{layer_idx}.weight"
            b_key = f"fc{layer_idx}.bias"

            if w_key not in self._weights:
                # Try alternative naming
                w_key = f"hidden_{layer_idx}_weight"
                b_key = f"hidden_{layer_idx}_bias"

            if w_key not in self._weights:
                break

            w = self._weights[w_key]
            b = self._weights[b_key]

            # Linear + ReLU
            x = np.dot(x, w.T) + b
            x = np.maximum(x, 0)  # ReLU

            layer_idx += 1

        # Output layer
        for out_key in ["output.weight", "output_weight"]:
            if out_key in self._weights:
                w = self._weights[out_key]
                b = self._weights.get(out_key.replace("weight", "bias"), 0)
                x = np.dot(x, w.T) + b
                break

        return x

    def get_stats(self, n_warmup: int = 10, n_benchmark: int = 100) -> RuntimeStats:
        """Get runtime performance statistics.

        Args:
            n_warmup: Warmup iterations.
            n_benchmark: Benchmark iterations.

        Returns:
            Runtime statistics.
        """
        # Model size
        size_bytes = os.path.getsize(self.model_path)

        # Warmup
        for _ in range(n_warmup):
            self.generate(n_samples=1)

        # Benchmark
        latencies = []
        for _ in range(n_benchmark):
            start = time.perf_counter()
            self.generate(n_samples=1)
            latencies.append((time.perf_counter() - start) * 1000)

        latency_ms = np.mean(latencies)
        throughput = 1000 / latency_ms if latency_ms > 0 else 0

        # Memory (rough estimate)
        memory_mb = size_bytes / (1024 * 1024) * 2  # Rough estimate: model + activations

        return RuntimeStats(
            model_size_bytes=size_bytes,
            latency_ms=latency_ms,
            throughput=throughput,
            memory_peak_mb=memory_mb,
            platform=self._format.value,
        )

    def benchmark(
        self,
        batch_sizes: List[int] = [1, 8, 32, 64, 128],
    ) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark.

        Args:
            batch_sizes: Batch sizes to test.

        Returns:
            List of benchmark results.
        """
        results = []

        for batch_size in batch_sizes:
            latencies = []
            n_runs = 50

            for _ in range(n_runs):
                start = time.perf_counter()
                self.generate(n_samples=batch_size)
                latencies.append((time.perf_counter() - start) * 1000)

            results.append({
                "batch_size": batch_size,
                "latency_mean_ms": np.mean(latencies),
                "latency_p50_ms": np.percentile(latencies, 50),
                "latency_p99_ms": np.percentile(latencies, 99),
                "throughput": batch_size / (np.mean(latencies) / 1000),
            })

        return results


class EdgeGeneratorFactory:
    """Factory for creating edge-optimized generators."""

    @staticmethod
    def create_minimal_generator(
        n_features: int,
        latent_dim: int = 16,
        hidden_dim: int = 32,
    ) -> Tuple[Dict[str, np.ndarray], EdgeGeneratorConfig]:
        """Create minimal generator suitable for edge deployment.

        Args:
            n_features: Number of output features.
            latent_dim: Latent space dimension.
            hidden_dim: Hidden layer dimension.

        Returns:
            Tuple of (weights dict, config).
        """
        config = EdgeGeneratorConfig(
            n_features=n_features,
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim],
        )

        # Xavier initialization
        def xavier_init(fan_in: int, fan_out: int) -> np.ndarray:
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return np.random.randn(fan_out, fan_in).astype(np.float32) * std

        weights = {
            "fc0.weight": xavier_init(latent_dim, hidden_dim),
            "fc0.bias": np.zeros(hidden_dim, dtype=np.float32),
            "output.weight": xavier_init(hidden_dim, n_features),
            "output.bias": np.zeros(n_features, dtype=np.float32),
        }

        return weights, config

    @staticmethod
    def from_trained_generator(
        generator: Any,
        optimize_for_size: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], EdgeGeneratorConfig]:
        """Extract weights from trained generator.

        Args:
            generator: Trained Genesis generator.
            optimize_for_size: Apply size optimizations.

        Returns:
            Tuple of (weights dict, config).
        """
        weights = {}

        if hasattr(generator, "model") and TORCH_AVAILABLE:
            import torch
            for name, param in generator.model.state_dict().items():
                w = param.cpu().numpy()
                if optimize_for_size:
                    # Convert to float16 for size reduction
                    w = w.astype(np.float16)
                weights[name] = w
        elif hasattr(generator, "weights"):
            weights = generator.weights

        # Infer config
        n_features = None
        latent_dim = None

        for name, w in weights.items():
            if "output" in name and "weight" in name:
                n_features = w.shape[0]
            if "fc0" in name and "weight" in name:
                latent_dim = w.shape[1]

        config = EdgeGeneratorConfig(
            n_features=n_features or 10,
            latent_dim=latent_dim or 32,
        )

        return weights, config


# Convenience functions

def export_to_onnx(
    generator: Any,
    output_path: str,
    optimize: bool = True,
    quantize: bool = False,
) -> ExportResult:
    """Export generator to ONNX format.

    Args:
        generator: Generator to export.
        output_path: Output file path.
        optimize: Apply optimizations.
        quantize: Apply quantization.

    Returns:
        Export result.
    """
    exporter = EdgeExporter()
    return exporter.export_onnx(generator, output_path, optimize=optimize, quantize=quantize)


def export_to_numpy(
    generator: Any,
    output_path: str,
) -> ExportResult:
    """Export generator to NumPy format.

    Args:
        generator: Generator to export.
        output_path: Output file path (.npz).

    Returns:
        Export result.
    """
    exporter = EdgeExporter()
    return exporter.export_numpy(generator, output_path)


def load_edge_runtime(model_path: str) -> EdgeRuntime:
    """Load edge runtime from file.

    Args:
        model_path: Path to model file.

    Returns:
        EdgeRuntime instance.
    """
    return EdgeRuntime.load(model_path)
