"""Genesis Testing Module - Automated Test Data Generation.

This module provides pytest integration for automatic synthetic test data
generation from schema definitions and ORM models.

Example:
    >>> # In conftest.py
    >>> from genesis.testing import synthetic_fixture
    >>>
    >>> @synthetic_fixture(n_samples=100)
    >>> def customer_data(schema):
    ...     return {
    ...         "name": "string",
    ...         "age": "int:18-65",
    ...         "email": "email",
    ...         "balance": "float:0-10000",
    ...     }
    >>>
    >>> # In test file
    >>> def test_customer_processing(customer_data):
    ...     assert len(customer_data) == 100
    ...     assert all(18 <= age <= 65 for age in customer_data["age"])
"""

from genesis.testing.fixtures import (
    SyntheticFixture,
    create_fixture,
    fixture_from_dataclass,
    fixture_from_schema,
    synthetic_fixture,
)
from genesis.testing.generators import (
    ColumnGenerator,
    SchemaBasedGenerator,
    TestDataGenerator,
)
from genesis.testing.orm import (
    ORMIntrospector,
    fixture_from_django,
    fixture_from_sqlalchemy,
)
from genesis.testing.plugin import (
    configure_genesis_testing,
    genesis_seed,
)
from genesis.testing.schemas import (
    ColumnSpec,
    SchemaDefinition,
    parse_column_spec,
)

__all__ = [
    # Decorators
    "synthetic_fixture",
    "create_fixture",
    # Classes
    "SyntheticFixture",
    "TestDataGenerator",
    "SchemaBasedGenerator",
    "ColumnGenerator",
    "SchemaDefinition",
    "ColumnSpec",
    "ORMIntrospector",
    # ORM helpers
    "fixture_from_sqlalchemy",
    "fixture_from_django",
    "fixture_from_dataclass",
    "fixture_from_schema",
    # Utilities
    "parse_column_spec",
    "configure_genesis_testing",
    "genesis_seed",
]
