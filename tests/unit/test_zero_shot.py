"""Tests for zero-shot schema inference."""

import pytest

from genesis.zero_shot import (
    ColumnDefinition,
    SchemaDefinition,
    ZeroShotDataGenerator,
    ZeroShotSchemaGenerator,
)


class TestColumnDefinition:
    """Tests for ColumnDefinition dataclass."""

    def test_default_values(self) -> None:
        """Test default column definition values."""
        col = ColumnDefinition(
            name="id",
            data_type="integer",
        )

        assert col.name == "id"
        assert col.data_type == "integer"
        assert col.nullable is True
        assert col.unique is False
        assert col.constraints == {}

    def test_with_constraints(self) -> None:
        """Test column definition with constraints."""
        col = ColumnDefinition(
            name="age",
            data_type="integer",
            nullable=False,
            constraints={"min": 0, "max": 150},
        )

        assert col.constraints["min"] == 0
        assert col.constraints["max"] == 150


class TestSchemaDefinition:
    """Tests for SchemaDefinition dataclass."""

    def test_basic_schema(self) -> None:
        """Test basic schema definition."""
        columns = [
            ColumnDefinition("id", "integer", unique=True),
            ColumnDefinition("name", "string"),
        ]

        schema = SchemaDefinition(
            table_name="users",
            columns=columns,
        )

        assert schema.table_name == "users"
        assert len(schema.columns) == 2

    def test_to_dict(self) -> None:
        """Test schema to dictionary conversion."""
        columns = [
            ColumnDefinition("id", "integer"),
            ColumnDefinition("name", "string"),
        ]

        schema = SchemaDefinition(
            table_name="users",
            columns=columns,
        )

        result = schema.to_dict()

        assert result["table_name"] == "users"
        assert "columns" in result
        assert len(result["columns"]) == 2

    def test_get_column_names(self) -> None:
        """Test getting column names."""
        columns = [
            ColumnDefinition("id", "integer"),
            ColumnDefinition("name", "string"),
            ColumnDefinition("email", "string"),
        ]

        schema = SchemaDefinition(
            table_name="users",
            columns=columns,
        )

        names = schema.get_column_names()
        assert names == ["id", "name", "email"]


class TestZeroShotSchemaGenerator:
    """Tests for ZeroShotSchemaGenerator."""

    def test_initialization(self) -> None:
        """Test generator initialization."""
        generator = ZeroShotSchemaGenerator()

        assert generator is not None
        assert len(generator._templates) > 0

    def test_has_domain_templates(self) -> None:
        """Test that domain templates are loaded."""
        generator = ZeroShotSchemaGenerator()

        # Check common domains
        templates = generator._templates
        assert "ecommerce" in templates or "e-commerce" in templates
        assert "healthcare" in templates or "medical" in templates
        assert "finance" in templates or "banking" in templates

    def test_generate_from_template_ecommerce(self) -> None:
        """Test schema generation from ecommerce template."""
        generator = ZeroShotSchemaGenerator()

        schema = generator.generate_from_template("ecommerce", "users")

        assert schema is not None
        assert schema.table_name == "users"
        assert len(schema.columns) > 0

    def test_generate_from_template_healthcare(self) -> None:
        """Test schema generation from healthcare template."""
        generator = ZeroShotSchemaGenerator()

        schema = generator.generate_from_template("healthcare", "patients")

        assert schema is not None
        assert schema.table_name == "patients"

    def test_generate_from_unknown_template(self) -> None:
        """Test handling of unknown template."""
        generator = ZeroShotSchemaGenerator()

        # Should return None or raise for unknown domain/table
        schema = generator.generate_from_template("unknown_domain", "unknown_table")
        assert schema is None

    def test_generate_from_description_simple(self) -> None:
        """Test schema generation from simple description."""
        generator = ZeroShotSchemaGenerator()

        description = "A user table with id, name, email, and age"
        schema = generator.generate_from_description(description)

        assert schema is not None
        assert len(schema.columns) >= 4

        column_names = schema.get_column_names()
        assert "id" in column_names
        assert "name" in column_names
        assert "email" in column_names
        assert "age" in column_names

    def test_generate_from_description_with_types(self) -> None:
        """Test schema generation preserves inferred types."""
        generator = ZeroShotSchemaGenerator()

        description = "Product catalog with price, quantity, and description"
        schema = generator.generate_from_description(description)

        assert schema is not None

        # Check that types are inferred
        price_col = next(
            (c for c in schema.columns if c.name == "price"), None
        )
        if price_col:
            assert price_col.data_type in ["float", "decimal", "number"]

    def test_list_available_templates(self) -> None:
        """Test listing available templates."""
        generator = ZeroShotSchemaGenerator()

        templates = generator.list_templates()

        assert isinstance(templates, dict)
        assert len(templates) > 0


class TestZeroShotDataGenerator:
    """Tests for ZeroShotDataGenerator."""

    def test_initialization(self) -> None:
        """Test generator initialization."""
        generator = ZeroShotDataGenerator()

        assert generator is not None

    def test_generate_from_schema(self) -> None:
        """Test data generation from schema."""
        schema = SchemaDefinition(
            table_name="users",
            columns=[
                ColumnDefinition("id", "integer", unique=True),
                ColumnDefinition("name", "string"),
                ColumnDefinition("age", "integer", constraints={"min": 18, "max": 80}),
            ],
        )

        generator = ZeroShotDataGenerator()
        data = generator.generate(schema, n_rows=100)

        assert len(data) == 100
        assert "id" in data.columns
        assert "name" in data.columns
        assert "age" in data.columns

    def test_generate_respects_unique_constraint(self) -> None:
        """Test that unique constraint is respected."""
        schema = SchemaDefinition(
            table_name="users",
            columns=[
                ColumnDefinition("id", "integer", unique=True),
            ],
        )

        generator = ZeroShotDataGenerator()
        data = generator.generate(schema, n_rows=100)

        assert data["id"].nunique() == 100

    def test_generate_respects_nullable(self) -> None:
        """Test that nullable constraint is respected."""
        schema = SchemaDefinition(
            table_name="data",
            columns=[
                ColumnDefinition("required_field", "string", nullable=False),
            ],
        )

        generator = ZeroShotDataGenerator()
        data = generator.generate(schema, n_rows=100)

        assert data["required_field"].isna().sum() == 0

    def test_generate_from_description(self) -> None:
        """Test end-to-end generation from description."""
        generator = ZeroShotDataGenerator()

        data = generator.generate_from_description(
            "A simple user table with id, name, and email",
            n_rows=50,
        )

        assert len(data) == 50
        assert "id" in data.columns
        assert "name" in data.columns
        assert "email" in data.columns

    def test_generate_from_domain_template(self) -> None:
        """Test generation from domain template."""
        generator = ZeroShotDataGenerator()

        data = generator.generate_from_template(
            domain="ecommerce",
            table="users",
            n_rows=100,
        )

        assert len(data) == 100
        assert len(data.columns) > 0

    def test_generate_different_types(self) -> None:
        """Test generation of different data types."""
        schema = SchemaDefinition(
            table_name="mixed",
            columns=[
                ColumnDefinition("int_col", "integer"),
                ColumnDefinition("float_col", "float"),
                ColumnDefinition("string_col", "string"),
                ColumnDefinition("bool_col", "boolean"),
                ColumnDefinition("date_col", "date"),
            ],
        )

        generator = ZeroShotDataGenerator()
        data = generator.generate(schema, n_rows=50)

        assert len(data) == 50
        assert data["int_col"].dtype in ["int64", "int32", "object"]
        assert data["float_col"].dtype in ["float64", "float32", "object"]
