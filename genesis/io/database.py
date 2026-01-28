"""Database connection utilities."""

from typing import Any, Dict, List

import pandas as pd


def read_sql(
    query: str,
    connection_string: str,
    **kwargs,
) -> pd.DataFrame:
    """Read data from SQL database.

    Args:
        query: SQL query or table name
        connection_string: SQLAlchemy connection string
        **kwargs: Additional pandas arguments

    Returns:
        DataFrame with query results
    """
    try:
        from sqlalchemy import create_engine
    except ImportError as e:
        raise ImportError("sqlalchemy required. Install with: pip install sqlalchemy") from e

    engine = create_engine(connection_string)

    return pd.read_sql(query, engine, **kwargs)


def write_sql(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    if_exists: str = "replace",
    index: bool = False,
    **kwargs,
) -> None:
    """Write DataFrame to SQL database.

    Args:
        df: DataFrame to write
        table_name: Target table name
        connection_string: SQLAlchemy connection string
        if_exists: Action if table exists ('fail', 'replace', 'append')
        index: Whether to write row index
        **kwargs: Additional pandas arguments
    """
    try:
        from sqlalchemy import create_engine
    except ImportError as e:
        raise ImportError("sqlalchemy required. Install with: pip install sqlalchemy") from e

    engine = create_engine(connection_string)

    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=index,
        **kwargs,
    )


def list_tables(connection_string: str) -> List[str]:
    """List tables in database.

    Args:
        connection_string: SQLAlchemy connection string

    Returns:
        List of table names
    """
    try:
        from sqlalchemy import create_engine, inspect
    except ImportError as e:
        raise ImportError("sqlalchemy required. Install with: pip install sqlalchemy") from e

    engine = create_engine(connection_string)
    inspector = inspect(engine)

    return inspector.get_table_names()


def get_table_schema(
    table_name: str,
    connection_string: str,
) -> Dict[str, Any]:
    """Get schema for a database table.

    Args:
        table_name: Table name
        connection_string: SQLAlchemy connection string

    Returns:
        Dictionary with table schema
    """
    try:
        from sqlalchemy import create_engine, inspect
    except ImportError as e:
        raise ImportError("sqlalchemy required. Install with: pip install sqlalchemy") from e

    engine = create_engine(connection_string)
    inspector = inspect(engine)

    columns = inspector.get_columns(table_name)
    pk = inspector.get_pk_constraint(table_name)
    fks = inspector.get_foreign_keys(table_name)

    return {
        "table_name": table_name,
        "columns": [
            {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
            }
            for col in columns
        ],
        "primary_key": pk.get("constrained_columns", []),
        "foreign_keys": [
            {
                "constrained_columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"],
            }
            for fk in fks
        ],
    }
