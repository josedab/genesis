"""Deep learning backend management."""

from typing import Optional

from genesis.core.exceptions import BackendNotAvailableError
from genesis.core.types import BackendType

# Track available backends
_pytorch_available: Optional[bool] = None
_tensorflow_available: Optional[bool] = None


def check_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    global _pytorch_available
    if _pytorch_available is None:
        try:
            import torch  # noqa: F401

            _pytorch_available = True
        except ImportError:
            _pytorch_available = False
    return _pytorch_available


def check_tensorflow_available() -> bool:
    """Check if TensorFlow is available."""
    global _tensorflow_available
    if _tensorflow_available is None:
        try:
            import tensorflow  # noqa: F401

            _tensorflow_available = True
        except ImportError:
            _tensorflow_available = False
    return _tensorflow_available


def get_available_backends() -> list:
    """Get list of available backends."""
    backends = []
    if check_pytorch_available():
        backends.append("pytorch")
    if check_tensorflow_available():
        backends.append("tensorflow")
    return backends


def select_backend(preferred: BackendType = BackendType.AUTO) -> str:
    """Select the best available backend.

    Args:
        preferred: Preferred backend (AUTO will select best available)

    Returns:
        Name of selected backend ('pytorch' or 'tensorflow')

    Raises:
        BackendNotAvailableError: If no backend is available
    """
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
        "any",
        "No deep learning backend available. Install PyTorch or TensorFlow: "
        "pip install genesis-synth[pytorch] or pip install genesis-synth[tensorflow]",
    )


def get_device(preferred: str = "auto") -> str:
    """Get the best available device for computation.

    Args:
        preferred: Preferred device ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Device string
    """
    if preferred != "auto":
        return preferred

    if check_pytorch_available():
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

    if check_tensorflow_available():
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return "GPU:0"

    return "cpu"


__all__ = [
    "check_pytorch_available",
    "check_tensorflow_available",
    "get_available_backends",
    "select_backend",
    "get_device",
]
