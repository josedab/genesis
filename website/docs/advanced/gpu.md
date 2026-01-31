---
sidebar_position: 3
title: GPU Acceleration
---

# GPU Acceleration

Accelerate training and generation with NVIDIA GPUs.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- cuDNN 8.0+
- PyTorch with CUDA

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```

## Basic GPU Usage

```python
from genesis import SyntheticGenerator

# Automatic GPU detection
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cuda'}  # Use default GPU
)

generator.fit(data)
synthetic = generator.generate(1000)
```

## Device Selection

```python
# Specific GPU
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cuda:0'}  # First GPU
)

# Second GPU
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cuda:1'}
)

# CPU fallback
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cpu'}
)

# Auto-detect (GPU if available, else CPU)
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'auto'}
)
```

## BatchedGenerator

Optimized for large-scale GPU generation:

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(
    method='ctgan',
    device='cuda',
    batch_size=10000,         # Samples per batch
    num_workers=4             # Data loading workers
)

generator.fit(large_data)

# Generate 1M samples efficiently
synthetic = generator.generate(1_000_000)
```

### Memory Management

```python
generator = BatchedGenerator(
    method='ctgan',
    device='cuda',
    batch_size=10000,
    max_memory_mb=8000,       # Limit GPU memory usage
    pin_memory=True,          # Faster CPU-GPU transfer
    prefetch_factor=2         # Prefetch batches
)
```

## Mixed Precision Training

Use FP16 for faster training with less memory:

```python
generator = SyntheticGenerator(
    method='ctgan',
    config={
        'device': 'cuda',
        'mixed_precision': True
    }
)

# Typically 2x faster training
generator.fit(data)
```

### Precision Options

```python
config = {
    'device': 'cuda',
    'precision': 'fp16',      # Half precision (fastest)
    # 'precision': 'fp32',    # Full precision (default)
    # 'precision': 'bf16',    # Brain float (A100+)
}
```

## Multi-GPU Training

### Data Parallel

```python
from genesis.gpu import DistributedGenerator

generator = DistributedGenerator(
    method='ctgan',
    devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
    strategy='data_parallel'
)

generator.fit(data)
synthetic = generator.generate(n_samples)
```

### Model Parallel

For very large models:

```python
generator = DistributedGenerator(
    method='ctgan',
    devices=['cuda:0', 'cuda:1'],
    strategy='model_parallel'
)
```

## Memory Optimization

### Gradient Checkpointing

Trade compute for memory:

```python
generator = SyntheticGenerator(
    method='ctgan',
    config={
        'device': 'cuda',
        'gradient_checkpointing': True  # Lower memory, slower
    }
)
```

### Automatic Memory Management

```python
from genesis.gpu import optimize_memory

with optimize_memory():
    generator = SyntheticGenerator(method='ctgan', config={'device': 'cuda'})
    generator.fit(large_data)  # Auto-adjusts batch size
```

### Memory Monitoring

```python
from genesis.gpu import GPUMonitor

monitor = GPUMonitor()

print(f"Total memory: {monitor.total_memory_gb:.1f} GB")
print(f"Used memory: {monitor.used_memory_gb:.1f} GB")
print(f"Free memory: {monitor.free_memory_gb:.1f} GB")

# Track during training
with monitor.track():
    generator.fit(data)

print(f"Peak memory: {monitor.peak_memory_gb:.1f} GB")
```

## GPU Selection Strategies

```python
from genesis.gpu import select_gpu

# Least loaded GPU
device = select_gpu(strategy='least_loaded')

# Most memory available
device = select_gpu(strategy='most_memory')

# Specific memory requirement
device = select_gpu(min_memory_gb=8)
```

## Benchmarking

```python
from genesis.gpu import benchmark

results = benchmark(
    data,
    methods=['ctgan', 'tvae'],
    devices=['cpu', 'cuda'],
    n_samples=[1000, 10000, 100000]
)

print(results.to_dataframe())
#          method  device  samples  fit_time  gen_time  memory_mb
# 0        ctgan     cpu     1000      45.2       0.5        512
# 1        ctgan    cuda     1000       8.3       0.1       1024
# 2        ctgan    cuda    10000       8.5       0.8       1024
# 3        ctgan    cuda   100000       9.1       7.2       1280
```

## Performance Tips

### 1. Use Appropriate Batch Size

```python
# Too small: underutilizes GPU
# Too large: out of memory
# Rule of thumb: start with 500, increase until memory full

generator = SyntheticGenerator(
    method='ctgan',
    config={
        'device': 'cuda',
        'batch_size': 1000  # Adjust based on GPU memory
    }
)
```

### 2. Enable CUDA Optimizations

```python
import torch

# Enable cuDNN autotuning (first epoch slower, rest faster)
torch.backends.cudnn.benchmark = True

# Enable TF32 on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
```

### 3. Pin Memory for Large Datasets

```python
generator = BatchedGenerator(
    method='ctgan',
    device='cuda',
    pin_memory=True,
    num_workers=4
)
```

### 4. Use Async Data Loading

```python
from genesis.gpu import AsyncDataLoader

loader = AsyncDataLoader(
    data,
    batch_size=1000,
    device='cuda',
    prefetch=2
)

for batch in loader:
    # batch is already on GPU
    process(batch)
```

## Troubleshooting

### CUDA Out of Memory

```python
# Solution 1: Reduce batch size
config = {'batch_size': 256}

# Solution 2: Enable gradient checkpointing
config = {'gradient_checkpointing': True}

# Solution 3: Use mixed precision
config = {'mixed_precision': True}

# Solution 4: Clear cache
import torch
torch.cuda.empty_cache()
```

### Slow Training

```python
# Check data loading is not bottleneck
generator = BatchedGenerator(
    num_workers=8,      # Increase
    prefetch_factor=4,  # Increase
    pin_memory=True
)

# Enable cuDNN benchmark
import torch
torch.backends.cudnn.benchmark = True
```

### Multi-GPU Not Scaling

```python
# Ensure balanced workload
generator = DistributedGenerator(
    devices=['cuda:0', 'cuda:1'],
    batch_size=2000  # Per GPU, not total
)

# Check NVLink connectivity
# nvidia-smi topo --matrix
```

## Cloud GPU Setup

### AWS

```bash
# Launch p3.2xlarge (V100) or p4d.24xlarge (A100)
pip install genesis[gpu]

# Verify
python -c "from genesis.gpu import check_gpu; check_gpu()"
```

### GCP

```bash
# Use n1-standard with nvidia-tesla-v100
pip install genesis[gpu]
```

### Azure

```bash
# Use NC-series VMs
pip install genesis[gpu]
```

## Complete Example

```python
import pandas as pd
from genesis.gpu import BatchedGenerator, GPUMonitor

# Load large dataset
data = pd.read_parquet('large_dataset.parquet')  # 10M rows

# Setup GPU generator
generator = BatchedGenerator(
    method='ctgan',
    device='cuda',
    batch_size=5000,
    mixed_precision=True,
    config={
        'epochs': 300,
        'discriminator_steps': 1
    }
)

# Monitor memory
monitor = GPUMonitor()

with monitor.track():
    generator.fit(data, discrete_columns=['category', 'status'])
    
print(f"Training peak memory: {monitor.peak_memory_gb:.1f} GB")

# Generate 10M samples
synthetic = generator.generate(10_000_000)

print(f"Generated {len(synthetic)} rows")
synthetic.to_parquet('synthetic_large.parquet')
```

## Next Steps

- **[Distributed Generation](/docs/advanced/distributed)** - Multi-node GPU clusters
- **[Configuration](/docs/api/configuration)** - All GPU options
