# Data Warehouse Connectors

Genesis provides native connectors for popular cloud data warehouses, enabling seamless integration of synthetic data generation with existing data infrastructure.

## Supported Warehouses

| Connector | Backend | Status |
|-----------|---------|--------|
| **SnowflakeConnector** | Snowflake | ✅ Full support |
| **BigQueryConnector** | Google BigQuery | ✅ Full support |
| **DatabricksConnector** | Databricks SQL | ✅ Full support |

## Installation

```bash
# Install with specific connector
pip install genesis-synth[connectors]

# Or individual connectors
pip install snowflake-connector-python  # Snowflake
pip install google-cloud-bigquery       # BigQuery
pip install databricks-sql-connector    # Databricks
```

## Snowflake Connector

Connect to Snowflake data warehouses for schema discovery and data operations.

```python
from genesis.connectors import SnowflakeConnector

connector = SnowflakeConnector(
    account="xy12345.us-east-1",
    user="my_user",
    password="my_password",
    database="MY_DB",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
)

# Use as context manager
with connector:
    # Discover schema
    table_info = connector.discover_schema("customers")
    print(f"Table has {len(table_info.columns)} columns")
    
    # Read sample data
    sample = connector.read_sample("customers", n_rows=5000)
    
    # Train generator on sample
    generator.fit(sample)
    synthetic = generator.generate(10000)
    
    # Write synthetic data back
    connector.write_data(synthetic, "customers_synthetic")
```

### Authentication Options

```python
# Password authentication
connector = SnowflakeConnector(
    account="account",
    user="user",
    password="password",
)

# SSO/Browser authentication
connector = SnowflakeConnector(
    account="account",
    user="user",
    authenticator="externalbrowser",
)

# Key-pair authentication
connector = SnowflakeConnector(
    account="account",
    user="user",
    private_key_path="/path/to/rsa_key.p8",
)
```

## BigQuery Connector

Connect to Google BigQuery for large-scale data operations.

```python
from genesis.connectors import BigQueryConnector

connector = BigQueryConnector(
    project="my-gcp-project",
    credentials_path="/path/to/service-account.json",  # Optional
    location="US",
)

with connector:
    # List datasets
    datasets = connector.list_datasets()
    
    # Discover schema
    schema = connector.discover_schema("dataset.table_name")
    
    # Read data via query
    data = connector.read_query("""
        SELECT * FROM `project.dataset.table`
        WHERE date > '2026-01-01'
        LIMIT 10000
    """)
    
    # Write synthetic data
    connector.write_data(synthetic_df, "dataset.synthetic_table")
```

### Using Default Credentials

```python
# Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
# or default service account in GCP environment
connector = BigQueryConnector(project="my-project")
```

## Databricks Connector

Connect to Databricks SQL Warehouses and Unity Catalog.

```python
from genesis.connectors import DatabricksConnector

connector = DatabricksConnector(
    host="adb-1234567890.1.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",  # Or use DATABRICKS_TOKEN env var
    catalog="main",
    schema="default",
)

with connector:
    # List catalogs and schemas
    catalogs = connector.list_catalogs()
    schemas = connector.list_schemas("main")
    
    # Discover table schema
    info = connector.discover_schema("main.default.customers")
    
    # Read sample
    sample = connector.read_sample("main.default.customers", n_rows=10000)
```

## Schema Discovery

All connectors provide schema discovery for automatic generator configuration:

```python
from genesis.connectors import get_connector, WarehouseSchemaAdapter

# Factory function for any connector
connector = get_connector(
    "snowflake",
    account="...",
    user="...",
    password="...",
)

with connector:
    # Discover schema
    table_info = connector.discover_schema("customers")
    
    # Convert to Genesis format
    genesis_schema = WarehouseSchemaAdapter.to_genesis_schema(table_info)
    
    print(f"Columns: {list(genesis_schema['columns'].keys())}")
    print(f"Discrete columns: {genesis_schema['discrete_columns']}")
```

### TableInfo Structure

```python
@dataclass
class TableInfo:
    name: str              # Table name
    schema: str            # Schema/dataset name
    database: str          # Database/project name
    columns: List[ColumnInfo]
    row_count: Optional[int]
    comment: Optional[str]

@dataclass  
class ColumnInfo:
    name: str              # Column name
    data_type: str         # Native data type
    nullable: bool         # Allows nulls
    primary_key: bool      # Is primary key
    comment: Optional[str] # Column description
```

## Writing Synthetic Data

All connectors support writing synthetic data back to the warehouse:

```python
# Replace existing table
connector.write_data(synthetic_df, "target_table", if_exists="replace")

# Append to existing table
connector.write_data(synthetic_df, "target_table", if_exists="append")

# Fail if table exists
connector.write_data(synthetic_df, "target_table", if_exists="fail")
```

## Complete Workflow Example

```python
from genesis import SyntheticGenerator
from genesis.connectors import SnowflakeConnector, WarehouseSchemaAdapter

# Connect to warehouse
connector = SnowflakeConnector(
    account="myaccount",
    user="genesis_user",
    password="secure_password",
    database="PRODUCTION",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
)

with connector:
    # 1. Discover schema
    table_info = connector.discover_schema("CUSTOMERS")
    schema = WarehouseSchemaAdapter.to_genesis_schema(table_info)
    
    # 2. Sample production data
    sample = connector.read_sample("CUSTOMERS", n_rows=50000)
    
    # 3. Train generator
    generator = SyntheticGenerator(
        method="ctgan",
        discrete_columns=schema["discrete_columns"],
    )
    generator.fit(sample)
    
    # 4. Generate synthetic data
    synthetic = generator.generate(100000)
    
    # 5. Write to test environment
    connector.write_data(
        synthetic,
        "CUSTOMERS_SYNTHETIC",
        if_exists="replace",
    )
    
    print(f"Generated {len(synthetic)} synthetic customers")
```

## Configuration Reference

### SnowflakeConnector

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `account` | str | Yes | Snowflake account identifier |
| `user` | str | Yes | Username |
| `password` | str | No | Password |
| `database` | str | No | Default database |
| `schema` | str | No | Default schema (default: PUBLIC) |
| `warehouse` | str | No | Compute warehouse |
| `role` | str | No | Role to use |
| `authenticator` | str | No | Auth method (externalbrowser, etc.) |
| `private_key_path` | str | No | Path to private key for key-pair auth |

### BigQueryConnector

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project` | str | Yes | GCP project ID |
| `credentials_path` | str | No | Path to service account JSON |
| `location` | str | No | BigQuery location (default: US) |

### DatabricksConnector

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `host` | str | Yes | Workspace hostname |
| `http_path` | str | Yes | SQL warehouse HTTP path |
| `access_token` | str | No | Personal access token |
| `catalog` | str | No | Unity Catalog name (default: main) |
| `schema` | str | No | Default schema (default: default) |
