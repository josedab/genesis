# ADR-0002: Multi-Backend Deep Learning Support

## Status

Accepted

## Context

Genesis provides deep learning-based generators (CTGAN, TVAE, TimeGAN) that require a neural network framework. The ML ecosystem is split between two dominant frameworks:

- **PyTorch**: Preferred by researchers, growing in industry, Pythonic API
- **TensorFlow**: Established in production, Keras integration, TPU support

Forcing users to install a specific framework creates friction:
- Organizations often standardize on one framework
- Framework choice affects GPU driver requirements
- Installing both wastes disk space (~2GB each)
- Some environments (edge, embedded) have constraints

We needed to support both without maintaining duplicate implementations.

## Decision

We implement a **backend abstraction layer** that:

1. **Auto-detects available backends** at runtime via lazy imports
2. **Prefers PyTorch** when both are available (more active research ecosystem)
3. **Allows explicit selection** via configuration
4. **Fails gracefully** with clear installation instructions

```python
# Backend selection logic (genesis/backends/__init__.py)
def select_backend(preferred: BackendType = BackendType.AUTO) -> str:
    if preferred == BackendType.PYTORCH:
        if check_pytorch_available():
            return "pytorch"
        raise BackendNotAvailableError("pytorch")
    
    if preferred == BackendType.TENSORFLOW:
        if check_tensorflow_available():
            return "tensorflow"
        raise BackendNotAvailableError("tensorflow")
    
    # Auto-select: prefer PyTorch
    if check_pytorch_available():
        return "pytorch"
    if check_tensorflow_available():
        return "tensorflow"
    
    raise BackendNotAvailableError(
        "No deep learning backend available. Install with: "
        "pip install genesis-synth[pytorch] or genesis-synth[tensorflow]"
    )
```

Configuration interface:

```python
from genesis import SyntheticGenerator, GeneratorConfig, BackendType

# Auto-detect (default)
gen = SyntheticGenerator(method='ctgan')

# Explicit selection
config = GeneratorConfig(backend=BackendType.PYTORCH)
gen = SyntheticGenerator(config=config)

# Or via string
gen = SyntheticGenerator(method='ctgan', backend='tensorflow')
```

## Consequences

### Positive

- **User flexibility**: works with whichever framework is already installed
- **Lightweight core**: base package is ~50MB, not ~2GB
- **Clear errors**: when no backend available, error message shows install command
- **Future-proof**: easy to add JAX or other backends later
- **Device abstraction**: `get_device()` handles CUDA/MPS/CPU selection uniformly

### Negative

- **Maintenance burden**: some code paths are duplicated per backend
- **Testing complexity**: CI matrix must test both backends
- **Feature parity risk**: new features might not be implemented for both immediately
- **Subtle differences**: PyTorch and TensorFlow have slightly different numerics

### Mitigations

- Generator implementations use a common base class with shared logic
- Integration tests verify output similarity between backends
- PyTorch is the "reference" implementation; TensorFlow matches its behavior
- Backend-specific code is isolated in `genesis/backends/pytorch/` and `genesis/backends/tensorflow/`

## Device Selection

The backend layer also abstracts device selection:

```python
from genesis.backends import get_device

device = get_device("auto")  # Returns "cuda", "mps", or "cpu"
device = get_device("cuda")  # Explicit GPU
device = get_device("cpu")   # Force CPU
```

This handles:
- NVIDIA CUDA detection (PyTorch and TensorFlow)
- Apple Silicon MPS detection (PyTorch only)
- Graceful fallback to CPU
