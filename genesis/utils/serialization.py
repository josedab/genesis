"""Safe serialization utilities for Genesis models.

This module provides secure serialization alternatives to pickle,
using JSON for metadata and numpy's save format for arrays.

Security Note:
    Pickle can execute arbitrary code during deserialization,
    making it unsafe for loading untrusted model files.
    This module uses JSON + numpy which are safe formats.
"""

import json
import os
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Version for format compatibility checking
SERIALIZATION_VERSION = "1.0.0"


class SerializationError(Exception):
    """Error during model serialization or deserialization."""

    pass


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable format."""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, np.ndarray):
        return {"__ndarray__": True, "data": value.tolist(), "dtype": str(value.dtype)}
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, pd.DataFrame):
        return {
            "__dataframe__": True,
            "columns": list(value.columns),
            "data": value.to_dict(orient="list"),
        }
    elif isinstance(value, pd.Series):
        return {"__series__": True, "name": value.name, "data": value.tolist()}
    elif hasattr(value, "value"):  # Enum
        return {"__enum__": True, "type": type(value).__name__, "value": value.value}
    elif is_dataclass(value):
        return {"__dataclass__": True, "type": type(value).__name__, "data": asdict(value)}
    elif hasattr(value, "to_dict"):
        return {"__custom__": True, "type": type(value).__name__, "data": value.to_dict()}
    else:
        # Try string conversion as last resort
        try:
            return str(value)
        except Exception:
            raise SerializationError(f"Cannot serialize value of type {type(value)}")


def _deserialize_value(value: Any, type_registry: Optional[Dict[str, Type]] = None) -> Any:
    """Convert a JSON value back to its original type."""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, list):
        return [_deserialize_value(v, type_registry) for v in value]
    elif isinstance(value, dict):
        if value.get("__ndarray__"):
            return np.array(value["data"], dtype=value.get("dtype"))
        elif value.get("__dataframe__"):
            return pd.DataFrame(value["data"], columns=value["columns"])
        elif value.get("__series__"):
            return pd.Series(value["data"], name=value.get("name"))
        elif value.get("__enum__") and type_registry:
            enum_type = type_registry.get(value["type"])
            if enum_type:
                return enum_type(value["value"])
            return value["value"]
        elif value.get("__dataclass__") and type_registry:
            dc_type = type_registry.get(value["type"])
            if dc_type:
                return dc_type(**value["data"])
            return value["data"]
        elif value.get("__custom__") and type_registry:
            custom_type = type_registry.get(value["type"])
            if custom_type and hasattr(custom_type, "from_dict"):
                return custom_type.from_dict(value["data"])
            return value["data"]
        else:
            return {k: _deserialize_value(v, type_registry) for k, v in value.items()}
    else:
        return value


class ModelSerializer:
    """Safe model serializer using JSON and numpy formats.

    This class provides secure serialization that avoids the security
    risks associated with pickle. Models are saved as ZIP archives
    containing JSON metadata and numpy arrays.

    Example:
        >>> serializer = ModelSerializer()
        >>> serializer.save(generator, "model.genesis")
        >>> loaded = serializer.load("model.genesis", CTGANGenerator)
    """

    def __init__(self, type_registry: Optional[Dict[str, Type]] = None) -> None:
        """Initialize the serializer.

        Args:
            type_registry: Optional mapping of type names to classes for
                          deserializing custom types (enums, dataclasses).
        """
        self.type_registry = type_registry or {}

    def save(
        self,
        obj: Any,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save an object to a file.

        The object must have `get_serialization_state()` and
        `set_serialization_state()` methods, or be a simple
        dictionary/dataclass.

        Args:
            obj: Object to serialize
            path: Output file path (recommended extension: .genesis)
            metadata: Optional additional metadata to include

        Raises:
            SerializationError: If serialization fails
        """
        path = Path(path)

        try:
            # Get state from object
            if hasattr(obj, "get_serialization_state"):
                state = obj.get_serialization_state()
            elif is_dataclass(obj):
                state = asdict(obj)
            elif isinstance(obj, dict):
                state = obj
            else:
                raise SerializationError(
                    f"Object of type {type(obj).__name__} does not support serialization. "
                    "Implement get_serialization_state() method."
                )

            # Build manifest
            manifest = {
                "version": SERIALIZATION_VERSION,
                "class_name": type(obj).__name__,
                "class_module": type(obj).__module__,
                "metadata": metadata or {},
            }

            # Create temporary directory for building archive
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Save manifest
                with open(tmppath / "manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)

                # Serialize state, extracting large arrays
                arrays = {}
                serialized_state = self._serialize_with_array_extraction(state, arrays, tmppath)

                # Save state
                with open(tmppath / "state.json", "w") as f:
                    json.dump(serialized_state, f, indent=2)

                # Create ZIP archive
                with ZipFile(path, "w") as zf:
                    for file in tmppath.iterdir():
                        zf.write(file, file.name)

            logger.info(f"Saved model to {path}")

        except Exception as e:
            raise SerializationError(f"Failed to save model: {e}") from e

    def load(
        self,
        path: Union[str, Path],
        cls: Optional[Type[T]] = None,
    ) -> T:
        """Load an object from a file.

        Args:
            path: Input file path
            cls: Optional class to instantiate. If provided, the class
                 must have a `from_serialization_state()` class method
                 or `set_serialization_state()` instance method.

        Returns:
            Deserialized object

        Raises:
            SerializationError: If deserialization fails
        """
        path = Path(path)

        if not path.exists():
            raise SerializationError(f"Model file not found: {path}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Extract archive
                with ZipFile(path, "r") as zf:
                    zf.extractall(tmppath)

                # Load manifest
                with open(tmppath / "manifest.json") as f:
                    manifest = json.load(f)

                # Version check
                file_version = manifest.get("version", "0.0.0")
                if file_version != SERIALIZATION_VERSION:
                    logger.warning(
                        f"Model file version {file_version} differs from "
                        f"current version {SERIALIZATION_VERSION}"
                    )

                # Load state
                with open(tmppath / "state.json") as f:
                    serialized_state = json.load(f)

                # Deserialize state, loading arrays
                state = self._deserialize_with_array_loading(serialized_state, tmppath)

                # Create object
                if cls is not None:
                    if hasattr(cls, "from_serialization_state"):
                        obj = cls.from_serialization_state(state)
                    else:
                        obj = cls.__new__(cls)
                        if hasattr(obj, "set_serialization_state"):
                            obj.set_serialization_state(state)
                        else:
                            # Try to set attributes directly
                            for key, value in state.items():
                                setattr(obj, key, value)
                    return obj
                else:
                    return state

        except Exception as e:
            raise SerializationError(f"Failed to load model: {e}") from e

    def _serialize_with_array_extraction(
        self,
        value: Any,
        arrays: Dict[str, np.ndarray],
        tmppath: Path,
        prefix: str = "",
    ) -> Any:
        """Serialize value, extracting large numpy arrays to separate files."""
        if isinstance(value, np.ndarray) and value.size > 1000:
            # Save large arrays separately
            array_name = f"array_{len(arrays)}"
            array_path = tmppath / f"{array_name}.npy"
            np.save(array_path, value)
            arrays[array_name] = value
            return {"__array_file__": True, "name": array_name, "shape": list(value.shape)}
        elif isinstance(value, dict):
            return {
                k: self._serialize_with_array_extraction(v, arrays, tmppath, f"{prefix}.{k}")
                for k, v in value.items()
            }
        elif isinstance(value, (list, tuple)):
            return [
                self._serialize_with_array_extraction(v, arrays, tmppath, f"{prefix}[{i}]")
                for i, v in enumerate(value)
            ]
        else:
            return _serialize_value(value)

    def _deserialize_with_array_loading(
        self,
        value: Any,
        tmppath: Path,
    ) -> Any:
        """Deserialize value, loading numpy arrays from separate files."""
        if isinstance(value, dict):
            if value.get("__array_file__"):
                array_path = tmppath / f"{value['name']}.npy"
                return np.load(array_path)
            else:
                return {
                    k: self._deserialize_with_array_loading(v, tmppath) for k, v in value.items()
                }
        elif isinstance(value, list):
            return [self._deserialize_with_array_loading(v, tmppath) for v in value]
        else:
            return _deserialize_value(value, self.type_registry)


# Module-level convenience functions
_default_serializer = ModelSerializer()


def save_model(
    obj: Any,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model to a file using safe serialization.

    Args:
        obj: Object to serialize (must implement get_serialization_state)
        path: Output file path
        metadata: Optional additional metadata
    """
    _default_serializer.save(obj, path, metadata)


def load_model(
    path: Union[str, Path],
    cls: Optional[Type[T]] = None,
) -> T:
    """Load a model from a file.

    Args:
        path: Input file path
        cls: Optional class to instantiate

    Returns:
        Deserialized object
    """
    return _default_serializer.load(path, cls)
