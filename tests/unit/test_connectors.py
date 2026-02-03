"""Tests for data warehouse connectors."""

import pandas as pd
import pytest

from genesis.connectors import (
    BigQueryConnector,
    ColumnInfo,
    DatabricksConnector,
    SnowflakeConnector,
    TableInfo,
    WarehouseSchemaAdapter,
    get_connector,
)


class TestColumnInfo:
    """Tests for ColumnInfo dataclass."""

    def test_default_values(self) -> None:
        """Test default column info values."""
        col = ColumnInfo(name="id", data_type="INTEGER")

        assert col.name == "id"
        assert col.data_type == "INTEGER"
        assert col.nullable is True
        assert col.primary_key is False
        assert col.comment is None

    def test_custom_values(self) -> None:
        """Test custom column info values."""
        col = ColumnInfo(
            name="customer_id",
            data_type="VARCHAR(255)",
            nullable=False,
            primary_key=True,
            comment="Primary key",
        )

        assert col.name == "customer_id"
        assert col.nullable is False
        assert col.primary_key is True
        assert col.comment == "Primary key"


class TestTableInfo:
    """Tests for TableInfo dataclass."""

    def test_basic_table_info(self) -> None:
        """Test basic table info."""
        cols = [
            ColumnInfo(name="id", data_type="INTEGER", primary_key=True),
            ColumnInfo(name="name", data_type="VARCHAR"),
        ]

        table = TableInfo(
            name="customers",
            schema="public",
            database="mydb",
            columns=cols,
            row_count=1000,
        )

        assert table.name == "customers"
        assert table.schema == "public"
        assert table.database == "mydb"
        assert len(table.columns) == 2
        assert table.row_count == 1000


class TestWarehouseSchemaAdapter:
    """Tests for WarehouseSchemaAdapter."""

    def test_type_mapping(self) -> None:
        """Test data type mapping."""
        cols = [
            ColumnInfo(name="id", data_type="INTEGER"),
            ColumnInfo(name="price", data_type="FLOAT64"),
            ColumnInfo(name="name", data_type="VARCHAR"),
            ColumnInfo(name="is_active", data_type="BOOLEAN"),
            ColumnInfo(name="created", data_type="TIMESTAMP"),
        ]

        table = TableInfo(
            name="products",
            schema="public",
            database="db",
            columns=cols,
        )

        schema = WarehouseSchemaAdapter.to_genesis_schema(table)

        assert schema["table_name"] == "products"
        assert schema["columns"]["id"]["type"] == "integer"
        assert schema["columns"]["price"]["type"] == "float"
        assert schema["columns"]["name"]["type"] == "string"
        assert schema["columns"]["is_active"]["type"] == "boolean"
        assert schema["columns"]["created"]["type"] == "datetime"

    def test_detects_categorical_columns(self) -> None:
        """Test categorical column detection."""
        cols = [
            ColumnInfo(name="status_type", data_type="VARCHAR"),
            ColumnInfo(name="category_code", data_type="VARCHAR"),
            ColumnInfo(name="description", data_type="TEXT"),
        ]

        table = TableInfo(
            name="items",
            schema="public",
            database="db",
            columns=cols,
        )

        schema = WarehouseSchemaAdapter.to_genesis_schema(table)

        assert "status_type" in schema["discrete_columns"]
        assert "category_code" in schema["discrete_columns"]
        assert "description" not in schema["discrete_columns"]


class TestGetConnector:
    """Tests for get_connector factory function."""

    def test_snowflake_connector(self) -> None:
        """Test creating Snowflake connector."""
        connector = get_connector(
            "snowflake",
            account="test_account",
            user="test_user",
            password="test_pass",
            database="test_db",
        )

        assert isinstance(connector, SnowflakeConnector)
        assert connector.account == "test_account"
        assert connector.user == "test_user"

    def test_bigquery_connector(self) -> None:
        """Test creating BigQuery connector."""
        connector = get_connector(
            "bigquery",
            project="test-project",
        )

        assert isinstance(connector, BigQueryConnector)
        assert connector.project == "test-project"

    def test_databricks_connector(self) -> None:
        """Test creating Databricks connector."""
        connector = get_connector(
            "databricks",
            host="test.databricks.com",
            http_path="/sql/test",
        )

        assert isinstance(connector, DatabricksConnector)
        assert connector.host == "test.databricks.com"

    def test_unknown_connector_raises(self) -> None:
        """Test that unknown connector type raises error."""
        with pytest.raises(ValueError, match="Unknown warehouse type"):
            get_connector("unknown_warehouse")

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        connector = get_connector(
            "SNOWFLAKE",
            account="test",
            user="test",
        )
        assert isinstance(connector, SnowflakeConnector)


class TestSnowflakeConnector:
    """Tests for SnowflakeConnector."""

    def test_initialization(self) -> None:
        """Test connector initialization."""
        connector = SnowflakeConnector(
            account="xy12345.us-east-1",
            user="my_user",
            password="my_password",
            database="MY_DB",
            schema="PUBLIC",
            warehouse="COMPUTE_WH",
        )

        assert connector.account == "xy12345.us-east-1"
        assert connector.user == "my_user"
        assert connector.database == "MY_DB"
        assert connector.schema == "PUBLIC"
        assert connector.warehouse == "COMPUTE_WH"
        assert connector._conn is None

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        connector = SnowflakeConnector(
            account="test",
            user="test",
            password="test",
        )

        # Should have __enter__ and __exit__
        assert hasattr(connector, "__enter__")
        assert hasattr(connector, "__exit__")


class TestBigQueryConnector:
    """Tests for BigQueryConnector."""

    def test_initialization(self) -> None:
        """Test connector initialization."""
        connector = BigQueryConnector(
            project="my-project",
            location="US",
        )

        assert connector.project == "my-project"
        assert connector.location == "US"
        assert connector._client is None

    def test_with_credentials_path(self) -> None:
        """Test initialization with credentials path."""
        connector = BigQueryConnector(
            project="my-project",
            credentials_path="/path/to/creds.json",
        )

        assert connector.credentials_path == "/path/to/creds.json"


class TestDatabricksConnector:
    """Tests for DatabricksConnector."""

    def test_initialization(self) -> None:
        """Test connector initialization."""
        connector = DatabricksConnector(
            host="adb-xxx.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/xxx",
            catalog="main",
            schema="default",
        )

        assert connector.host == "adb-xxx.azuredatabricks.net"
        assert connector.catalog == "main"
        assert connector.schema == "default"
        assert connector._conn is None
