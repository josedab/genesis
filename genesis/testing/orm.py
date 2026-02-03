"""ORM introspection for automatic fixture generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

import pandas as pd

from genesis.testing.generators import TestDataGenerator
from genesis.testing.schemas import ColumnSpec, DataType, SchemaDefinition
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ORMIntrospector:
    """Introspect ORM models to generate schemas.

    Supports SQLAlchemy and Django ORM models.
    """

    @staticmethod
    def from_sqlalchemy(model: Type) -> SchemaDefinition:
        """Create schema from SQLAlchemy model.

        Args:
            model: SQLAlchemy model class

        Returns:
            SchemaDefinition

        Example:
            >>> from sqlalchemy import Column, Integer, String
            >>> from sqlalchemy.orm import declarative_base
            >>>
            >>> Base = declarative_base()
            >>>
            >>> class User(Base):
            ...     __tablename__ = 'users'
            ...     id = Column(Integer, primary_key=True)
            ...     name = Column(String(100))
            ...     email = Column(String(255), unique=True)
            >>>
            >>> schema = ORMIntrospector.from_sqlalchemy(User)
        """
        try:
            from sqlalchemy import inspect as sa_inspect
            from sqlalchemy.orm import RelationshipProperty
        except ImportError:
            raise ImportError("SQLAlchemy is required for ORM introspection")

        mapper = sa_inspect(model)
        columns = []
        foreign_keys: Dict[str, str] = {}
        primary_key = None

        for column in mapper.columns:
            col_spec = ORMIntrospector._sqlalchemy_column_to_spec(column)
            columns.append(col_spec)

            if column.primary_key:
                primary_key = column.name

            # Check for foreign keys
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    foreign_keys[column.name] = str(fk.target_fullname)

        return SchemaDefinition(
            name=mapper.class_.__tablename__,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )

    @staticmethod
    def _sqlalchemy_column_to_spec(column: Any) -> ColumnSpec:
        """Convert SQLAlchemy column to ColumnSpec."""
        from sqlalchemy import types as sa_types

        col_type = type(column.type)
        dtype = DataType.STRING  # Default

        # Map SQLAlchemy types to DataType
        type_mapping = {
            sa_types.Integer: DataType.INT,
            sa_types.SmallInteger: DataType.INT,
            sa_types.BigInteger: DataType.INT,
            sa_types.Float: DataType.FLOAT,
            sa_types.Numeric: DataType.FLOAT,
            sa_types.Boolean: DataType.BOOL,
            sa_types.Date: DataType.DATE,
            sa_types.DateTime: DataType.DATETIME,
            sa_types.Time: DataType.TIME,
            sa_types.Text: DataType.TEXT,
            sa_types.String: DataType.STRING,
        }

        for sa_type, data_type in type_mapping.items():
            if issubclass(col_type, sa_type):
                dtype = data_type
                break

        # Check for primary key
        if column.primary_key:
            dtype = DataType.PRIMARY_KEY

        # Check for foreign key
        if column.foreign_keys:
            dtype = DataType.FOREIGN_KEY

        # Build spec
        spec = ColumnSpec(
            name=column.name,
            dtype=dtype,
            nullable=column.nullable if not column.primary_key else False,
            unique=column.unique or column.primary_key,
        )

        # Add string length constraints
        if hasattr(column.type, "length") and column.type.length:
            spec.max_length = column.type.length

        return spec

    @staticmethod
    def from_django(model: Type) -> SchemaDefinition:
        """Create schema from Django model.

        Args:
            model: Django model class

        Returns:
            SchemaDefinition

        Example:
            >>> from django.db import models
            >>>
            >>> class User(models.Model):
            ...     name = models.CharField(max_length=100)
            ...     email = models.EmailField(unique=True)
            ...     age = models.IntegerField()
            >>>
            >>> schema = ORMIntrospector.from_django(User)
        """
        try:
            from django.db import models as django_models
        except ImportError:
            raise ImportError("Django is required for Django ORM introspection")

        meta = model._meta
        columns = []
        foreign_keys: Dict[str, str] = {}
        primary_key = None

        for field in meta.get_fields():
            if not hasattr(field, "column"):
                continue

            col_spec = ORMIntrospector._django_field_to_spec(field)
            if col_spec:
                columns.append(col_spec)

                if field.primary_key:
                    primary_key = field.name

                # Check for foreign keys
                if hasattr(field, "remote_field") and field.remote_field:
                    related_model = field.remote_field.model
                    related_pk = related_model._meta.pk.name
                    foreign_keys[field.name] = f"{related_model._meta.db_table}.{related_pk}"

        return SchemaDefinition(
            name=meta.db_table,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )

    @staticmethod
    def _django_field_to_spec(field: Any) -> Optional[ColumnSpec]:
        """Convert Django field to ColumnSpec."""
        try:
            from django.db import models as django_models
        except ImportError:
            return None

        field_type = type(field)
        dtype = DataType.STRING  # Default

        # Map Django field types to DataType
        type_mapping = {
            django_models.IntegerField: DataType.INT,
            django_models.SmallIntegerField: DataType.INT,
            django_models.BigIntegerField: DataType.INT,
            django_models.PositiveIntegerField: DataType.INT,
            django_models.FloatField: DataType.FLOAT,
            django_models.DecimalField: DataType.FLOAT,
            django_models.BooleanField: DataType.BOOL,
            django_models.NullBooleanField: DataType.BOOL,
            django_models.DateField: DataType.DATE,
            django_models.DateTimeField: DataType.DATETIME,
            django_models.TimeField: DataType.TIME,
            django_models.EmailField: DataType.EMAIL,
            django_models.URLField: DataType.URL,
            django_models.UUIDField: DataType.UUID,
            django_models.TextField: DataType.TEXT,
            django_models.CharField: DataType.STRING,
        }

        for django_type, data_type in type_mapping.items():
            if isinstance(field, django_type):
                dtype = data_type
                break

        # Check for auto field (primary key)
        if isinstance(field, django_models.AutoField):
            dtype = DataType.PRIMARY_KEY

        # Check for foreign key
        if isinstance(field, django_models.ForeignKey):
            dtype = DataType.FOREIGN_KEY

        spec = ColumnSpec(
            name=field.name,
            dtype=dtype,
            nullable=field.null if hasattr(field, "null") else True,
            unique=field.unique if hasattr(field, "unique") else False,
        )

        # Add constraints
        if hasattr(field, "max_length") and field.max_length:
            spec.max_length = field.max_length

        if dtype == DataType.INT and hasattr(field, "validators"):
            # Check for min/max validators
            pass

        return spec


def fixture_from_sqlalchemy(
    model: Type,
    n_samples: int = 100,
    seed: Optional[int] = None,
    scope: str = "function",
) -> Callable:
    """Create a pytest fixture from a SQLAlchemy model.

    Args:
        model: SQLAlchemy model class
        n_samples: Number of samples
        seed: Random seed
        scope: pytest fixture scope

    Returns:
        pytest fixture function

    Example:
        >>> user_fixture = fixture_from_sqlalchemy(User, n_samples=50)
        >>>
        >>> def test_users(user_fixture):
        ...     assert len(user_fixture) == 50
    """
    schema = ORMIntrospector.from_sqlalchemy(model)

    def fixture_func() -> pd.DataFrame:
        generator = TestDataGenerator(seed=seed)
        return generator.from_schema(schema, n_samples)

    try:
        import pytest

        return pytest.fixture(scope=scope)(fixture_func)
    except ImportError:
        return fixture_func


def fixture_from_django(
    model: Type,
    n_samples: int = 100,
    seed: Optional[int] = None,
    scope: str = "function",
) -> Callable:
    """Create a pytest fixture from a Django model.

    Args:
        model: Django model class
        n_samples: Number of samples
        seed: Random seed
        scope: pytest fixture scope

    Returns:
        pytest fixture function
    """
    schema = ORMIntrospector.from_django(model)

    def fixture_func() -> pd.DataFrame:
        generator = TestDataGenerator(seed=seed)
        return generator.from_schema(schema, n_samples)

    try:
        import pytest

        return pytest.fixture(scope=scope)(fixture_func)
    except ImportError:
        return fixture_func


class RelatedFixtureGenerator:
    """Generate related fixtures maintaining referential integrity.

    Useful for generating test data across multiple related tables.

    Example:
        >>> generator = RelatedFixtureGenerator(seed=42)
        >>> generator.add_table("customers", customer_schema, n_samples=100)
        >>> generator.add_table("orders", order_schema, n_samples=500,
        ...                     foreign_keys={"customer_id": "customers.id"})
        >>> data = generator.generate_all()
        >>> customers_df = data["customers"]
        >>> orders_df = data["orders"]
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator."""
        self.seed = seed
        self._tables: Dict[str, Dict[str, Any]] = {}
        self._generation_order: List[str] = []
        self._generated_data: Dict[str, pd.DataFrame] = {}

    def add_table(
        self,
        name: str,
        schema: SchemaDefinition,
        n_samples: int = 100,
        foreign_keys: Optional[Dict[str, str]] = None,
    ) -> "RelatedFixtureGenerator":
        """Add a table to generate.

        Args:
            name: Table name
            schema: Schema definition
            n_samples: Number of rows
            foreign_keys: Foreign key mappings {column: "table.column"}

        Returns:
            Self for chaining
        """
        self._tables[name] = {
            "schema": schema,
            "n_samples": n_samples,
            "foreign_keys": foreign_keys or {},
        }
        self._generation_order.append(name)
        return self

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all tables in dependency order.

        Returns:
            Dictionary of table name to DataFrame
        """
        from genesis.testing.generators import SchemaBasedGenerator

        self._generated_data = {}

        # Sort by dependencies (simple topological sort)
        sorted_tables = self._sort_by_dependencies()

        for table_name in sorted_tables:
            table_info = self._tables[table_name]
            generator = SchemaBasedGenerator(table_info["schema"], self.seed)

            # Pass parent data for foreign keys
            parent_data = {}
            for col, ref in table_info["foreign_keys"].items():
                parent_table, parent_col = ref.split(".")
                if parent_table in self._generated_data:
                    parent_data[parent_table] = self._generated_data[parent_table]

            df = generator.generate(table_info["n_samples"], parent_data)
            self._generated_data[table_name] = df

        return self._generated_data

    def _sort_by_dependencies(self) -> List[str]:
        """Sort tables by foreign key dependencies."""
        # Simple implementation: tables with no foreign keys first
        no_deps = []
        with_deps = []

        for name, info in self._tables.items():
            if info["foreign_keys"]:
                with_deps.append(name)
            else:
                no_deps.append(name)

        return no_deps + with_deps
