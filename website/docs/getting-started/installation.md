---
sidebar_position: 1
title: Installation
---

# Installation

Genesis supports Python 3.8+ and can be installed via pip with various optional dependencies.

## Quick Install

```bash
pip install genesis-synth
```

This installs Genesis with core dependencies for tabular data generation.

## Installation Options

### With Deep Learning Backends

```bash
# PyTorch backend (recommended for CTGAN, TVAE)
pip install genesis-synth[pytorch]

# TensorFlow backend
pip install genesis-synth[tensorflow]
```

### With Additional Features

```bash
# Text generation with LLMs
pip install genesis-synth[llm]

# Visualization and dashboards
pip install genesis-synth[viz]

# REST API server
pip install genesis-synth[api]

# Distributed training (Ray, Dask)
pip install genesis-synth[distributed]

# Hyperparameter tuning (Optuna)
pip install genesis-synth[tuning]

# Streaming (Kafka, WebSocket)
pip install genesis-synth[streaming]
```

### Full Installation

Install everything:

```bash
pip install genesis-synth[all]
```

## Verify Installation

```python
import genesis
print(genesis.__version__)  # Should print 1.4.0
```

Test that generators work:

```python
from genesis import SyntheticGenerator
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC']
})

# Quick test
generator = SyntheticGenerator(method='gaussian_copula')
generator.fit(df)
synthetic = generator.generate(10)
print(synthetic)
```

## Install from Source

For development or to get the latest features:

```bash
git clone https://github.com/genesis-synth/genesis.git
cd genesis
pip install -e ".[dev]"
```

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 4 GB | 16 GB |
| GPU | Optional | NVIDIA with CUDA 11+ |

### GPU Support

For GPU-accelerated training:

1. Install CUDA 11.0+ and cuDNN
2. Install PyTorch with CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install Genesis:
   ```bash
   pip install genesis-synth[pytorch]
   ```

Verify GPU is detected:

```python
from genesis.gpu import detect_gpus

gpus = detect_gpus()
for gpu in gpus:
    print(f"Found: {gpu.name} ({gpu.memory_total / 1024:.1f} GB)")
```

## Docker

Run Genesis in a container:

```bash
docker pull genesisai/genesis:latest
docker run -p 8000:8000 genesisai/genesis:latest
```

Or build from source:

```bash
cd genesis
docker build -t genesis .
docker run -p 8000:8000 genesis
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you have the right extras installed:

```bash
# For CTGAN/TVAE
pip install genesis-synth[pytorch]

# For text generation
pip install genesis-synth[llm]
```

### Memory Errors

For large datasets, use batched generation:

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(method='ctgan', batch_size=10000)
generator.fit(large_df)
synthetic = generator.generate(1_000_000)
```

### CUDA Not Found

1. Verify CUDA installation: `nvidia-smi`
2. Reinstall PyTorch with CUDA support
3. Set environment variable if needed:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

## Next Steps

- [Quick Start Guide](/docs/getting-started/quickstart) - Generate your first dataset
- [Core Concepts](/docs/concepts/overview) - Understand how Genesis works
- [Examples](/docs/examples) - See Genesis in action
