# GPU Acceleration

Genesis supports GPU acceleration for training synthetic data generators, significantly reducing generation time for large datasets.

## Overview

GPU acceleration can provide 10-50x speedups for deep learning-based generators (CTGAN, TVAE, TimeGAN). Genesis automatically detects available GPUs and provides utilities for efficient batch generation.

```python
from genesis.gpu import BatchedGenerator, MultiGPUGenerator

# Single GPU with batching
generator = BatchedGenerator(method="ctgan", device="cuda")

# Multi-GPU for large datasets
generator = MultiGPUGenerator(method="ctgan")
```

## Components

| Component | Purpose |
|-----------|---------|
| **BatchedGenerator** | Efficient batched generation on single GPU |
| **MultiGPUGenerator** | Distribute training/generation across GPUs |
| **detect_gpus()** | Detect available GPU resources |
| **optimize_batch_size()** | Find optimal batch size for your GPU |

## GPU Detection

```python
from genesis.gpu import detect_gpus, GPUInfo

# Check available GPUs
gpus = detect_gpus()

for gpu in gpus:
    print(f"""
GPU {gpu.index}: {gpu.name}
  Memory: {gpu.memory_total / 1024:.1f} GB
  Available: {gpu.memory_available / 1024:.1f} GB
  CUDA Capability: {gpu.cuda_capability}
""")

# Quick check
if detect_gpus():
    print("GPU available!")
else:
    print("No GPU - falling back to CPU")
```

## Single GPU Usage

### Basic GPU Generation

```python
from genesis import SyntheticGenerator

# Use GPU automatically if available
generator = SyntheticGenerator(
    method="ctgan",
    config={"device": "cuda"}  # or "cuda:0" for specific GPU
)

generator.fit(large_df)
synthetic = generator.generate(100000)
```

### Batched Generator

For memory-efficient generation of large datasets:

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(
    method="ctgan",
    device="cuda",
    batch_size=10000,     # Generate in batches
    memory_fraction=0.8   # Use 80% of GPU memory
)

generator.fit(large_df)

# Generate 1M samples in batches
synthetic = generator.generate(1_000_000)  # Automatically batched
```

### Batch Size Optimization

```python
from genesis.gpu import optimize_batch_size

# Find optimal batch size for your GPU
optimal_size = optimize_batch_size(
    model_type="ctgan",
    n_columns=len(df.columns),
    gpu_index=0
)

print(f"Recommended batch size: {optimal_size}")

generator = BatchedGenerator(
    method="ctgan",
    batch_size=optimal_size
)
```

## Multi-GPU Training

For very large datasets, distribute across multiple GPUs:

```python
from genesis.gpu import MultiGPUGenerator

# Use all available GPUs
generator = MultiGPUGenerator(
    method="ctgan",
    strategy="data_parallel"  # or "model_parallel"
)

generator.fit(massive_df)  # Data split across GPUs
synthetic = generator.generate(1_000_000)
```

### Parallelization Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| `data_parallel` | Large datasets | Split data across GPUs |
| `model_parallel` | Wide tables | Split model across GPUs |
| `pipeline` | Continuous generation | Pipeline stages across GPUs |

```python
# Data parallel (default)
generator = MultiGPUGenerator(
    method="ctgan",
    strategy="data_parallel",
    devices=["cuda:0", "cuda:1"]  # Specific GPUs
)

# Model parallel for wide tables
generator = MultiGPUGenerator(
    method="ctgan",
    strategy="model_parallel",
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
)
```

## Memory Management

```python
from genesis.gpu import BatchedGenerator, estimate_memory

# Estimate memory requirements
mem_estimate = estimate_memory(
    method="ctgan",
    n_rows=1_000_000,
    n_columns=50
)

print(f"Estimated GPU memory: {mem_estimate / 1024:.1f} GB")

# Configure for your memory constraints
generator = BatchedGenerator(
    method="ctgan",
    memory_limit="8GB",     # Hard limit
    memory_fraction=0.7,    # Use 70% of available
    auto_batch=True         # Automatically adjust batch size
)
```

### Memory-Efficient Training

```python
from genesis.gpu import BatchedGenerator

# For GPUs with limited memory
generator = BatchedGenerator(
    method="ctgan",
    device="cuda",
    config={
        "epochs": 300,
        "batch_size": 500,           # Smaller batch
        "discriminator_steps": 1,     # Reduce memory per step
        "gradient_checkpointing": True # Trade compute for memory
    }
)
```

## Mixed Precision Training

Use half-precision for faster training with less memory:

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(
    method="ctgan",
    device="cuda",
    mixed_precision=True,  # Use FP16 where possible
    config={
        "epochs": 300,
        "batch_size": 1000
    }
)

# Up to 2x faster, 50% less memory
generator.fit(df)
```

## Progress Monitoring

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(method="ctgan", device="cuda")
generator.fit(df)

# Monitor generation progress
for batch in generator.generate_batches(1_000_000, batch_size=50000):
    print(f"Generated batch: {len(batch)} rows")
    print(f"GPU memory: {generator.gpu_memory_used / 1024:.1f} GB")
```

## Fallback Handling

```python
from genesis.gpu import BatchedGenerator, detect_gpus

# Graceful fallback to CPU
if detect_gpus():
    device = "cuda"
    batch_size = 10000
else:
    device = "cpu"
    batch_size = 1000
    print("No GPU available, using CPU")

generator = BatchedGenerator(
    method="ctgan",
    device=device,
    batch_size=batch_size
)
```

## Pipeline Integration

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("large_dataset.parquet")
    .add_node("generate", "synthesize", {
        "method": "ctgan",
        "n_samples": 1_000_000,
        "device": "cuda",
        "batch_size": 50000
    })
    .sink("synthetic_large.parquet")
    .build()
)

pipeline.execute()
```

## CLI Usage

```bash
# Generate with GPU
genesis generate -i data.csv -o synthetic.csv -n 100000 --device cuda

# Specify batch size
genesis generate -i data.csv -o synthetic.csv -n 1000000 --device cuda --batch-size 50000

# Multi-GPU
genesis generate -i data.csv -o synthetic.csv -n 1000000 --multi-gpu
```

## Performance Tips

1. **Maximize batch size**: Use the largest batch size that fits in memory
2. **Use mixed precision**: Enable FP16 for modern GPUs (Volta+)
3. **Pin memory**: Enable pinned memory for faster CPU-GPU transfers
4. **Profile first**: Use `optimize_batch_size()` before long runs
5. **Monitor memory**: Watch GPU memory to avoid OOM errors

## Benchmarks

Approximate speedups vs CPU (RTX 3090, 50-column dataset):

| Dataset Size | CPU Time | GPU Time | Speedup |
|--------------|----------|----------|---------|
| 10K rows | 45s | 8s | 5.6x |
| 100K rows | 7m | 25s | 16.8x |
| 1M rows | 70m | 3m | 23.3x |
| 10M rows | 12h | 25m | 28.8x |

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
generator = BatchedGenerator(method="ctgan", batch_size=500)

# Or enable memory optimization
generator = BatchedGenerator(
    method="ctgan",
    memory_fraction=0.5,
    gradient_checkpointing=True
)
```

### CUDA Not Available

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check driver
from genesis.gpu import check_cuda_installation
check_cuda_installation()  # Prints diagnostic info
```

### Slow Generation

```python
# Enable async data loading
generator = BatchedGenerator(
    method="ctgan",
    num_workers=4,
    pin_memory=True
)
```
