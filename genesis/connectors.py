"""Data Warehouse Connectors for Genesis.

This module provides native connectors for popular cloud data warehouses,
enabling seamless integration of synthetic data generation with existing
data infrastructure.

Supported Warehouses:
- Snowflake
- Google BigQuery
- Databricks

Features:
- Schema discovery from warehouses
- Direct write of synthetic data
- Integration with DBT models
- Query-based generation triggers

Example:
    >>> from genesis.connectors import SnowflakeConnector
    >>>
    >>> connector = SnowflakeConnector(
    ...     account="myaccount",
    ...     user="user",
    ...     password="password",
    ...     database="mydb",
    ... )
    >>> schema = connector.discover_schema("customers")
    >>> connector.write_synthetic(synthetic_df, "customers_synthetic")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from genesis.core.exceptions import ConfigurationError, GenesisError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnInfo:
    """Information about a column in a warehouse table."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    comment: Optional[str] = None
    default: Optional[str] = None


@dataclass
class TableInfo:
    """Information about a table in a warehouse."""

    name: str
    schema: str
    database: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    comment: Optional[str] = None


class BaseWarehouseConnector(ABC):
    """Abstract base class for warehouse connectors."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the warehouse."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the warehouse."""
        pass

    @abstractmethod
    def discover_schema(self, table_name: str) -> TableInfo:
        """Discover schema of a table."""
        pass

    @abstractmethod
    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List available tables."""
        pass

    @abstractmethod
    def read_sample(self, table_name: str, n_rows: int = 1000) -> pd.DataFrame:
        """Read a sample of data from a table."""
        pass

    @abstractmethod
    def write_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> int:
        """Write data to the warehouse."""
        pass

    def __enter__(self) -> "BaseWarehouseConnector":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()


class SnowflakeConnector(BaseWarehouseConnector):
    """Snowflake data warehouse connector.

    Example:
        >>> connector = SnowflakeConnector(
        ...     account="xy12345.us-east-1",
        ...     user="my_user",
        ...     password="my_password",
        ...     database="MY_DB",
        ...     schema="PUBLIC",
        ...     warehouse="COMPUTE_WH",
        ... )
        >>> with connector:
        ...     schema = connector.discover_schema("customers")
        ...     sample = connector.read_sample("customers", n_rows=5000)
    """

    def __init__(
        self,
        account: str,
        user: str,
        password: Optional[str] = None,
        database: str = "",
        schema: str = "PUBLIC",
        warehouse: str = "",
        role: Optional[str] = None,
        authenticator: Optional[str] = None,
        private_key_path: Optional[str] = None,
    ) -> None:
        """Initialize Snowflake connector.

        Args:
            account: Snowflake account identifier
            user: Username
            password: Password (or use authenticator)
            database: Default database
            schema: Default schema
            warehouse: Compute warehouse
            role: Role to use
            authenticator: Authentication method (e.g., 'externalbrowser')
            private_key_path: Path to private key file for key-pair auth
        """
        self.account = account
        self.user = user
        self.password = password
        self.database = database
        self.schema = schema
        self.warehouse = warehouse
        self.role = role
        self.authenticator = authenticator
        self.private_key_path = private_key_path
        self._conn = None

    def _check_snowflake(self) -> None:
        """Check if snowflake-connector-python is installed."""
        try:
            import snowflake.connector  # noqa: F401
        except ImportError:
            raise ImportError(
                "snowflake-connector-python required. "
                "Install with: pip install snowflake-connector-python"
            )

    def connect(self) -> None:
        """Connect to Snowflake."""
        self._check_snowflake()
        import snowflake.connector

        connect_params = {
            "account": self.account,
            "user": self.user,
            "database": self.database,
            "schema": self.schema,
            "warehouse": self.warehouse,
        }

        if self.password:
            connect_params["password"] = self.password
        if self.role:
            connect_params["role"] = self.role
        if self.authenticator:
            connect_params["authenticator"] = self.authenticator
        if self.private_key_path:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            with open(self.private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend(),
                )
            connect_params["private_key"] = private_key

        self._conn = snowflake.connector.connect(**connect_params)
        logger.info(f"Connected to Snowflake account {self.account}")

    def disconnect(self) -> None:
        """Disconnect from Snowflake."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Disconnected from Snowflake")

    def discover_schema(self, table_name: str) -> TableInfo:
        """Discover schema of a Snowflake table."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        cursor = self._conn.cursor()
        try:
            # Get column information
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            columns = []
            for row in cursor.fetchall():
                col_name, col_type, kind, null_ok, default, pk, comment = (
                    row[0], row[1], row[2], row[3], row[4], row[5], row[8] if len(row) > 8 else None
                )
                columns.append(
                    ColumnInfo(
                        name=col_name,
                        data_type=col_type,
                        nullable=(null_ok == "Y"),
                        primary_key=(pk == "Y"),
                        default=default,
                        comment=comment,
                    )
                )

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            return TableInfo(
                name=table_name,
                schema=self.schema,
                database=self.database,
                columns=columns,
                row_count=row_count,
            )
        finally:
            cursor.close()

    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List tables in the schema."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        schema = schema or self.schema
        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SHOW TABLES IN SCHEMA {self.database}.{schema}")
            return [row[1] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def read_sample(self, table_name: str, n_rows: int = 1000) -> pd.DataFrame:
        """Read a sample from a Snowflake table."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        query = f"SELECT * FROM {table_name} SAMPLE ({n_rows} ROWS)"
        return pd.read_sql(query, self._conn)

    def read_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        return pd.read_sql(query, self._conn)

    def write_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> int:
        """Write DataFrame to Snowflake table.

        Args:
            data: DataFrame to write
            table_name: Target table name
            if_exists: 'replace', 'append', or 'fail'

        Returns:
            Number of rows written
        """
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        from snowflake.connector.pandas_tools import write_pandas

        # Handle if_exists
        cursor = self._conn.cursor()
        try:
            if if_exists == "replace":
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            elif if_exists == "fail":
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                if cursor.fetchone()[0] > 0:
                    raise GenesisError(f"Table {table_name} already exists")

            success, n_chunks, n_rows, _ = write_pandas(
                self._conn,
                data,
                table_name,
                auto_create_table=True,
            )

            if success:
                logger.info(f"Wrote {n_rows} rows to {table_name}")
                return n_rows
            else:
                raise GenesisError(f"Failed to write to {table_name}")
        finally:
            cursor.close()


class BigQueryConnector(BaseWarehouseConnector):
    """Google BigQuery connector.

    Example:
        >>> connector = BigQueryConnector(
        ...     project="my-project",
        ...     credentials_path="/path/to/credentials.json",
        ... )
        >>> with connector:
        ...     schema = connector.discover_schema("dataset.table")
    """

    def __init__(
        self,
        project: str,
        credentials_path: Optional[str] = None,
        location: str = "US",
    ) -> None:
        """Initialize BigQuery connector.

        Args:
            project: GCP project ID
            credentials_path: Path to service account JSON (uses default if None)
            location: BigQuery location
        """
        self.project = project
        self.credentials_path = credentials_path
        self.location = location
        self._client = None

    def _check_bigquery(self) -> None:
        """Check if google-cloud-bigquery is installed."""
        try:
            from google.cloud import bigquery  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery required. "
                "Install with: pip install google-cloud-bigquery"
            )

    def connect(self) -> None:
        """Connect to BigQuery."""
        self._check_bigquery()
        from google.cloud import bigquery

        if self.credentials_path:
            self._client = bigquery.Client.from_service_account_json(
                self.credentials_path,
                project=self.project,
            )
        else:
            self._client = bigquery.Client(project=self.project)

        logger.info(f"Connected to BigQuery project {self.project}")

    def disconnect(self) -> None:
        """Disconnect from BigQuery."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Disconnected from BigQuery")

    def discover_schema(self, table_name: str) -> TableInfo:
        """Discover schema of a BigQuery table.

        Args:
            table_name: Fully qualified table name (dataset.table)
        """
        if not self._client:
            raise GenesisError("Not connected. Call connect() first.")

        table_ref = f"{self.project}.{table_name}"
        table = self._client.get_table(table_ref)

        columns = []
        for field in table.schema:
            columns.append(
                ColumnInfo(
                    name=field.name,
                    data_type=field.field_type,
                    nullable=(field.mode != "REQUIRED"),
                    comment=field.description,
                )
            )

        return TableInfo(
            name=table_name.split(".")[-1],
            schema=table_name.split(".")[0] if "." in table_name else "",
            database=self.project,
            columns=columns,
            row_count=table.num_rows,
            comment=table.description,
        )

    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List tables in a dataset."""
        if not self._client:
            raise GenesisError("Not connected. Call connect() first.")

        if not schema:
            raise ValueError("schema (dataset) is required for BigQuery")

        tables = self._client.list_tables(schema)
        return [f"{schema}.{t.table_id}" for t in tables]

    def list_datasets(self) -> List[str]:
        """List datasets in the project."""
        if not self._client:
            raise GenesisError("Not connected. Call connect() first.")

        datasets = self._client.list_datasets()
        return [d.dataset_id for d in datasets]

    def read_sample(self, table_name: str, n_rows: int = 1000) -> pd.DataFrame:
        """Read a sample from BigQuery."""
        if not self._client:
            raise GenesisError("Not connected. Call connect() first.")

        query = f"SELECT * FROM `{self.project}.{table_name}` LIMIT {n_rows}"
        return self._client.query(query).to_dataframe()

    def read_query(self, query: str) -> pd.DataFrame:
        """Execute query and return results."""
        if not self._client:
            raise GenesisError("Not connected. Call connect() first.")

        return self._client.query(query).to_dataframe()

    def write_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> int:
        """Write DataFrame to BigQuery.

        Args:
            data: DataFrame to write
            table_name: Target table (dataset.table format)
            if_exists: 'replace', 'append', or 'fail'

        Returns:
            Number of rows written
        """
        if not self._client:
            raise GenesisError("Not connected. Call connect() first.")

        from google.cloud.bigquery import LoadJobConfig, WriteDisposition

        table_ref = f"{self.project}.{table_name}"

        disposition_map = {
            "replace": WriteDisposition.WRITE_TRUNCATE,
            "append": WriteDisposition.WRITE_APPEND,
            "fail": WriteDisposition.WRITE_EMPTY,
        }

        job_config = LoadJobConfig(
            write_disposition=disposition_map.get(if_exists, WriteDisposition.WRITE_TRUNCATE)
        )

        job = self._client.load_table_from_dataframe(
            data, table_ref, job_config=job_config
        )
        job.result()  # Wait for completion

        logger.info(f"Wrote {len(data)} rows to {table_name}")
        return len(data)


class DatabricksConnector(BaseWarehouseConnector):
    """Databricks connector using SQL Warehouse or Unity Catalog.

    Example:
        >>> connector = DatabricksConnector(
        ...     host="adb-xxx.azuredatabricks.net",
        ...     http_path="/sql/1.0/warehouses/xxx",
        ...     access_token="dapi...",
        ... )
        >>> with connector:
        ...     schema = connector.discover_schema("catalog.schema.table")
    """

    def __init__(
        self,
        host: str,
        http_path: str,
        access_token: Optional[str] = None,
        catalog: str = "main",
        schema: str = "default",
    ) -> None:
        """Initialize Databricks connector.

        Args:
            host: Databricks workspace hostname
            http_path: SQL warehouse HTTP path
            access_token: Personal access token (or uses env var)
            catalog: Unity Catalog name
            schema: Default schema
        """
        self.host = host
        self.http_path = http_path
        self.access_token = access_token
        self.catalog = catalog
        self.schema = schema
        self._conn = None

    def _check_databricks(self) -> None:
        """Check if databricks-sql-connector is installed."""
        try:
            from databricks import sql  # noqa: F401
        except ImportError:
            raise ImportError(
                "databricks-sql-connector required. "
                "Install with: pip install databricks-sql-connector"
            )

    def connect(self) -> None:
        """Connect to Databricks SQL Warehouse."""
        self._check_databricks()
        from databricks import sql
        import os

        token = self.access_token or os.environ.get("DATABRICKS_TOKEN")
        if not token:
            raise ConfigurationError(
                "Databricks access token required. "
                "Set via access_token parameter or DATABRICKS_TOKEN env var."
            )

        self._conn = sql.connect(
            server_hostname=self.host,
            http_path=self.http_path,
            access_token=token,
        )
        logger.info(f"Connected to Databricks at {self.host}")

    def disconnect(self) -> None:
        """Disconnect from Databricks."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Disconnected from Databricks")

    def discover_schema(self, table_name: str) -> TableInfo:
        """Discover schema of a Databricks table.

        Args:
            table_name: Table name (can be fully qualified: catalog.schema.table)
        """
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        cursor = self._conn.cursor()
        try:
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            columns = []
            for row in cursor.fetchall():
                if row[0].startswith("#"):
                    continue  # Skip partition info headers
                columns.append(
                    ColumnInfo(
                        name=row[0],
                        data_type=row[1],
                        nullable=True,
                        comment=row[2] if len(row) > 2 else None,
                    )
                )

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            parts = table_name.split(".")
            return TableInfo(
                name=parts[-1],
                schema=parts[-2] if len(parts) > 1 else self.schema,
                database=parts[-3] if len(parts) > 2 else self.catalog,
                columns=columns,
                row_count=row_count,
            )
        finally:
            cursor.close()

    def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List tables in the schema."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        schema = schema or f"{self.catalog}.{self.schema}"
        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SHOW TABLES IN {schema}")
            return [f"{schema}.{row[1]}" for row in cursor.fetchall()]
        finally:
            cursor.close()

    def list_catalogs(self) -> List[str]:
        """List available catalogs."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        cursor = self._conn.cursor()
        try:
            cursor.execute("SHOW CATALOGS")
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def list_schemas(self, catalog: Optional[str] = None) -> List[str]:
        """List schemas in a catalog."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        catalog = catalog or self.catalog
        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SHOW SCHEMAS IN {catalog}")
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def read_sample(self, table_name: str, n_rows: int = 1000) -> pd.DataFrame:
        """Read a sample from a Databricks table."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {n_rows}")
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        finally:
            cursor.close()

    def read_query(self, query: str) -> pd.DataFrame:
        """Execute query and return results."""
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        cursor = self._conn.cursor()
        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        finally:
            cursor.close()

    def write_data(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> int:
        """Write DataFrame to Databricks.

        Note: This creates a temporary view and inserts data via SQL.
        For large datasets, consider using Spark DataFrames directly.

        Args:
            data: DataFrame to write
            table_name: Target table
            if_exists: 'replace', 'append', or 'fail'

        Returns:
            Number of rows written
        """
        if not self._conn:
            raise GenesisError("Not connected. Call connect() first.")

        cursor = self._conn.cursor()
        try:
            if if_exists == "replace":
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Create table with schema from DataFrame
            columns_sql = ", ".join([
                f"{col} STRING" for col in data.columns
            ])
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql})")

            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                values = []
                for _, row in batch.iterrows():
                    row_values = ", ".join([
                        f"'{str(v).replace(chr(39), chr(39)+chr(39))}'" for v in row
                    ])
                    values.append(f"({row_values})")

                if values:
                    cursor.execute(
                        f"INSERT INTO {table_name} VALUES {', '.join(values)}"
                    )

            logger.info(f"Wrote {len(data)} rows to {table_name}")
            return len(data)
        finally:
            cursor.close()


class WarehouseSchemaAdapter:
    """Adapts warehouse schemas to Genesis format."""

    TYPE_MAPPING = {
        # Snowflake types
        "NUMBER": "float",
        "FLOAT": "float",
        "REAL": "float",
        "VARCHAR": "string",
        "CHAR": "string",
        "STRING": "string",
        "TEXT": "string",
        "BOOLEAN": "boolean",
        "DATE": "date",
        "DATETIME": "datetime",
        "TIMESTAMP": "datetime",
        "TIMESTAMP_NTZ": "datetime",
        "TIMESTAMP_LTZ": "datetime",
        "TIMESTAMP_TZ": "datetime",
        # BigQuery types
        "INT64": "integer",
        "FLOAT64": "float",
        "NUMERIC": "float",
        "BIGNUMERIC": "float",
        "BOOL": "boolean",
        "BYTES": "string",
        "GEOGRAPHY": "string",
        "JSON": "string",
        # Databricks types
        "BIGINT": "integer",
        "INT": "integer",
        "SMALLINT": "integer",
        "TINYINT": "integer",
        "DOUBLE": "float",
        "DECIMAL": "float",
        "BINARY": "string",
    }

    @classmethod
    def to_genesis_schema(cls, table_info: TableInfo) -> Dict[str, Any]:
        """Convert warehouse table info to Genesis schema format.

        Args:
            table_info: Table information from warehouse

        Returns:
            Schema dictionary compatible with Genesis generators
        """
        columns = {}
        discrete_columns = []

        for col in table_info.columns:
            # Map type
            base_type = col.data_type.upper().split("(")[0]
            genesis_type = cls.TYPE_MAPPING.get(base_type, "string")

            columns[col.name] = {
                "type": genesis_type,
                "nullable": col.nullable,
            }

            # Detect likely categorical columns
            if genesis_type == "string" and col.name.lower().endswith(("_type", "_status", "_category", "_code")):
                discrete_columns.append(col.name)

        return {
            "table_name": table_info.name,
            "columns": columns,
            "discrete_columns": discrete_columns,
            "row_count": table_info.row_count,
        }


def get_connector(
    warehouse_type: str,
    **kwargs: Any,
) -> BaseWarehouseConnector:
    """Factory function to get appropriate connector.

    Args:
        warehouse_type: One of 'snowflake', 'bigquery', 'databricks'
        **kwargs: Connection parameters

    Returns:
        Appropriate connector instance

    Example:
        >>> connector = get_connector(
        ...     "snowflake",
        ...     account="myaccount",
        ...     user="user",
        ...     password="pass",
        ...     database="db",
        ... )
    """
    connectors = {
        "snowflake": SnowflakeConnector,
        "bigquery": BigQueryConnector,
        "databricks": DatabricksConnector,
    }

    warehouse_type = warehouse_type.lower()
    if warehouse_type not in connectors:
        raise ValueError(
            f"Unknown warehouse type: {warehouse_type}. "
            f"Supported: {list(connectors.keys())}"
        )

    return connectors[warehouse_type](**kwargs)
