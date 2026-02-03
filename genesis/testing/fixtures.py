"""pytest fixtures for synthetic test data generation."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pandas as pd

from genesis.testing.generators import SchemaBasedGenerator, TestDataGenerator
from genesis.testing.schemas import SchemaDefinition, SchemaPresets
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Global configuration
_global_seed: Optional[int] = None
_global_generator: Optional[TestDataGenerator] = None


def set_global_seed(seed: int) -> None:
    """Set the global seed for all fixtures."""
    global _global_seed, _global_generator
    _global_seed = seed
    _global_generator = TestDataGenerator(seed=seed)


def get_generator() -> TestDataGenerator:
    """Get the global generator instance."""
    global _global_generator
    if _global_generator is None:
        _global_generator = TestDataGenerator(seed=_global_seed)
    return _global_generator


@dataclass
class SyntheticFixture:
    """Configuration for a synthetic data fixture.

    Attributes:
        schema: Schema definition or dictionary
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        deterministic: If True, use same seed each time
        cache: If True, cache generated data
    """

    schema: Union[SchemaDefinition, Dict[str, Any], Callable]
    n_samples: int = 100
    seed: Optional[int] = None
    deterministic: bool = True
    cache: bool = True
    name: Optional[str] = None

    _cached_data: Optional[pd.DataFrame] = None

    def generate(self) -> pd.DataFrame:
        """Generate the fixture data."""
        if self.cache and self._cached_data is not None:
            return self._cached_data.copy()

        seed = self.seed if self.deterministic else None
        generator = TestDataGenerator(seed=seed)

        if callable(self.schema):
            schema_def = self.schema()
            if isinstance(schema_def, dict):
                data = generator.from_dict(schema_def, self.n_samples)
            elif isinstance(schema_def, SchemaDefinition):
                data = generator.from_schema(schema_def, self.n_samples)
            else:
                raise TypeError(f"Schema function must return dict or SchemaDefinition, got {type(schema_def)}")
        elif isinstance(self.schema, dict):
            data = generator.from_dict(self.schema, self.n_samples)
        elif isinstance(self.schema, SchemaDefinition):
            data = generator.from_schema(self.schema, self.n_samples)
        else:
            raise TypeError(f"Invalid schema type: {type(self.schema)}")

        if self.cache:
            self._cached_data = data

        return data.copy()

    def clear_cache(self) -> None:
        """Clear cached data."""
        self._cached_data = None


def synthetic_fixture(
    n_samples: int = 100,
    seed: Optional[int] = None,
    deterministic: bool = True,
    cache: bool = True,
    scope: str = "function",
) -> Callable:
    """Decorator to create a pytest fixture for synthetic data.

    This decorator transforms a function that returns a schema definition
    into a pytest fixture that generates synthetic data.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        deterministic: If True, always generate same data
        cache: If True, cache data within scope
        scope: pytest fixture scope ('function', 'class', 'module', 'session')

    Returns:
        Decorator function

    Example:
        >>> @synthetic_fixture(n_samples=100, seed=42)
        ... def customer_data():
        ...     return {
        ...         "id": "primary_key",
        ...         "name": "name",
        ...         "email": "email!",
        ...         "age": "int:18-80",
        ...     }
        >>>
        >>> def test_customer_validation(customer_data):
        ...     assert len(customer_data) == 100
        ...     assert customer_data["email"].is_unique
    """

    def decorator(func: Callable) -> Callable:
        fixture_config = SyntheticFixture(
            schema=func,
            n_samples=n_samples,
            seed=seed,
            deterministic=deterministic,
            cache=cache,
            name=func.__name__,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
            return fixture_config.generate()

        # Store configuration for pytest plugin access
        wrapper._synthetic_fixture = fixture_config  # type: ignore

        # Try to apply pytest.fixture if available
        try:
            import pytest

            return pytest.fixture(scope=scope)(wrapper)
        except ImportError:
            # pytest not available, return plain function
            return wrapper

    return decorator


def create_fixture(
    schema: Union[Dict[str, Any], SchemaDefinition, str],
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Create synthetic test data directly (without pytest).

    This is useful for quick testing or non-pytest contexts.

    Args:
        schema: Schema definition (dict, SchemaDefinition, or preset name)
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Generated DataFrame

    Example:
        >>> df = create_fixture({
        ...     "name": "name",
        ...     "age": "int:18-65",
        ...     "email": "email",
        ... }, n_samples=50)
    """
    generator = TestDataGenerator(seed=seed)

    if isinstance(schema, str):
        # Try to get preset
        preset_method = getattr(SchemaPresets, schema, None)
        if preset_method:
            schema_def = preset_method()
        else:
            raise ValueError(f"Unknown preset: {schema}")
    elif isinstance(schema, dict):
        schema_def = SchemaDefinition.from_dict(schema)
    else:
        schema_def = schema

    return generator.from_schema(schema_def, n_samples)


def fixture_from_schema(
    schema: SchemaDefinition,
    n_samples: int = 100,
    seed: Optional[int] = None,
    scope: str = "function",
) -> Callable:
    """Create a pytest fixture from a SchemaDefinition.

    Args:
        schema: Schema definition
        n_samples: Number of samples
        seed: Random seed
        scope: pytest fixture scope

    Returns:
        pytest fixture function

    Example:
        >>> schema = SchemaDefinition(
        ...     name="users",
        ...     columns=[
        ...         ColumnSpec("id", DataType.PRIMARY_KEY),
        ...         ColumnSpec("email", DataType.EMAIL, unique=True),
        ...     ]
        ... )
        >>> user_fixture = fixture_from_schema(schema, n_samples=50)
    """

    def fixture_func() -> pd.DataFrame:
        generator = TestDataGenerator(seed=seed)
        return generator.from_schema(schema, n_samples)

    try:
        import pytest

        return pytest.fixture(scope=scope)(fixture_func)
    except ImportError:
        return fixture_func


def fixture_from_dataclass(
    dataclass_type: Type,
    n_samples: int = 100,
    seed: Optional[int] = None,
    scope: str = "function",
) -> Callable:
    """Create a pytest fixture from a dataclass definition.

    Args:
        dataclass_type: A dataclass type to generate data for
        n_samples: Number of samples
        seed: Random seed
        scope: pytest fixture scope

    Returns:
        pytest fixture function

    Example:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class User:
        ...     id: int
        ...     name: str
        ...     email: str
        ...     age: int
        >>>
        >>> user_fixture = fixture_from_dataclass(User, n_samples=100)
    """
    import dataclasses

    if not dataclasses.is_dataclass(dataclass_type):
        raise TypeError(f"{dataclass_type} is not a dataclass")

    from genesis.testing.schemas import ColumnSpec, DataType

    # Convert dataclass fields to schema
    columns = []
    for field_info in dataclasses.fields(dataclass_type):
        dtype = _python_type_to_data_type(field_info.type)
        columns.append(ColumnSpec(name=field_info.name, dtype=dtype))

    schema = SchemaDefinition(name=dataclass_type.__name__, columns=columns)

    def fixture_func() -> pd.DataFrame:
        generator = TestDataGenerator(seed=seed)
        return generator.from_schema(schema, n_samples)

    try:
        import pytest

        return pytest.fixture(scope=scope)(fixture_func)
    except ImportError:
        return fixture_func


def _python_type_to_data_type(python_type: Any) -> "DataType":
    """Convert Python type annotation to DataType."""
    from genesis.testing.schemas import DataType

    # Handle string annotations
    if isinstance(python_type, str):
        type_str = python_type.lower()
    else:
        type_str = getattr(python_type, "__name__", str(python_type)).lower()

    type_mapping = {
        "int": DataType.INT,
        "float": DataType.FLOAT,
        "str": DataType.STRING,
        "bool": DataType.BOOL,
        "date": DataType.DATE,
        "datetime": DataType.DATETIME,
        "time": DataType.TIME,
    }

    return type_mapping.get(type_str, DataType.STRING)


# Preset fixtures
def user_fixture(n_samples: int = 100, seed: Optional[int] = None) -> Callable:
    """Create a user data fixture."""
    return fixture_from_schema(SchemaPresets.user(), n_samples, seed)


def customer_fixture(n_samples: int = 100, seed: Optional[int] = None) -> Callable:
    """Create a customer data fixture."""
    return fixture_from_schema(SchemaPresets.customer(), n_samples, seed)


def order_fixture(n_samples: int = 100, seed: Optional[int] = None) -> Callable:
    """Create an order data fixture."""
    return fixture_from_schema(SchemaPresets.order(), n_samples, seed)


def transaction_fixture(n_samples: int = 100, seed: Optional[int] = None) -> Callable:
    """Create a transaction data fixture."""
    return fixture_from_schema(SchemaPresets.transaction(), n_samples, seed)


def patient_fixture(n_samples: int = 100, seed: Optional[int] = None) -> Callable:
    """Create a patient data fixture."""
    return fixture_from_schema(SchemaPresets.patient(), n_samples, seed)
