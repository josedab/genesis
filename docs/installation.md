# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip or conda package manager

## Basic Installation

Install from PyPI:

```bash
pip install genesis-synth
```

## Installation Options

### With PyTorch (recommended for deep learning)

```bash
pip install genesis-synth[pytorch]
```

### With TensorFlow

```bash
pip install genesis-synth[tensorflow]
```

### With LLM Support

```bash
pip install genesis-synth[llm]
```

This includes OpenAI and HuggingFace transformers.

### With Visualization

```bash
pip install genesis-synth[viz]
```

Includes matplotlib and plotly.

### With REST API

```bash
pip install genesis-synth[api]
```

Includes FastAPI and uvicorn for serving Genesis as a REST service.

### With Hyperparameter Tuning

```bash
pip install genesis-synth[tuning]
```

Includes Optuna for automatic hyperparameter optimization.

### With Distributed Training

```bash
pip install genesis-synth[distributed]
```

Includes Ray and Dask for distributed and multi-GPU training.

### With Streaming (v1.3.0+)

```bash
pip install genesis-synth[streaming]
```

Includes Kafka and WebSocket support for real-time data generation.

### With PDF Reporting (v1.3.0+)

```bash
pip install genesis-synth[reporting]
```

Includes WeasyPrint for PDF export of quality reports and dashboards.

### Full Installation

```bash
pip install genesis-synth[all]
```

Includes all optional dependencies.

## Development Installation

For contributing to Genesis:

```bash
git clone https://github.com/genesis-synth/genesis.git
cd genesis
pip install -e ".[dev]"
```

## Verify Installation

```python
import genesis
print(genesis.__version__)  # Should print 1.4.0
```

## Troubleshooting

### PyTorch Installation Issues

If PyTorch fails to install, try installing it separately first:

```bash
# CPU only
pip install torch

# With CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# With CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow Installation Issues

For M1/M2/M3 Macs:

```bash
pip install tensorflow-macos
```

### OpenAI API

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

### WeasyPrint Installation (PDF Export)

WeasyPrint requires system dependencies. On macOS:

```bash
brew install pango
```

On Ubuntu/Debian:

```bash
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0
```
