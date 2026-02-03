"""Zero-Shot Schema Inference for Synthetic Data Generation.

This module enables generating synthetic data schemas from natural language
descriptions without any real data input. Uses LLMs to understand domain
requirements and generate appropriate schemas.

Features:
- Natural language schema generation
- Domain-aware templates (healthcare, finance, e-commerce, etc.)
- Relationship inference
- Constraint generation
- Sample data generation

Example:
    >>> from genesis.zero_shot import ZeroShotSchemaGenerator
    >>>
    >>> generator = ZeroShotSchemaGenerator()
    >>> schema = generator.generate_schema(
    ...     "An e-commerce platform with users, products, orders, and reviews"
    ... )
    >>> synthetic_data = generator.generate_data(schema, n_samples=1000)
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.exceptions import ConfigurationError, GenesisError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ColumnDataType(Enum):
    """Data types for schema columns."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    UUID = "uuid"
    CATEGORICAL = "categorical"
    TEXT = "text"
    URL = "url"
    IP_ADDRESS = "ip_address"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"


class RelationshipType(Enum):
    """Types of relationships between tables."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_MANY = "many_to_many"


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""

    name: str
    data_type: ColumnDataType
    nullable: bool = True
    unique: bool = False
    primary_key: bool = False
    foreign_key: Optional[str] = None  # "table.column" format
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    pattern: Optional[str] = None  # Regex pattern
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "nullable": self.nullable,
            "unique": self.unique,
            "primary_key": self.primary_key,
            "foreign_key": self.foreign_key,
            "default": self.default,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "categories": self.categories,
            "pattern": self.pattern,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSchema":
        data = dict(data)
        data["data_type"] = ColumnDataType(data["data_type"])
        return cls(**data)


@dataclass
class TableSchema:
    """Schema definition for a table."""

    name: str
    columns: List[ColumnSchema] = field(default_factory=list)
    description: Optional[str] = None
    row_count_estimate: Optional[int] = None

    def add_column(self, column: ColumnSchema) -> "TableSchema":
        self.columns.append(column)
        return self

    @property
    def primary_key_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.primary_key]

    @property
    def foreign_key_columns(self) -> List[Tuple[str, str]]:
        return [(c.name, c.foreign_key) for c in self.columns if c.foreign_key]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "description": self.description,
            "row_count_estimate": self.row_count_estimate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSchema":
        return cls(
            name=data["name"],
            columns=[ColumnSchema.from_dict(c) for c in data.get("columns", [])],
            description=data.get("description"),
            row_count_estimate=data.get("row_count_estimate"),
        )


@dataclass
class Relationship:
    """Relationship between tables."""

    name: str
    source_table: str
    target_table: str
    source_column: str
    target_column: str
    relationship_type: RelationshipType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_table": self.source_table,
            "target_table": self.target_table,
            "source_column": self.source_column,
            "target_column": self.target_column,
            "relationship_type": self.relationship_type.value,
        }


@dataclass
class DatabaseSchema:
    """Complete database schema with multiple tables and relationships."""

    name: str
    tables: List[TableSchema] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    description: Optional[str] = None
    domain: Optional[str] = None

    def add_table(self, table: TableSchema) -> "DatabaseSchema":
        self.tables.append(table)
        return self

    def add_relationship(self, relationship: Relationship) -> "DatabaseSchema":
        self.relationships.append(relationship)
        return self

    def get_table(self, name: str) -> Optional[TableSchema]:
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tables": [t.to_dict() for t in self.tables],
            "relationships": [r.to_dict() for r in self.relationships],
            "description": self.description,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseSchema":
        relationships = []
        for r in data.get("relationships", []):
            relationships.append(
                Relationship(
                    name=r["name"],
                    source_table=r["source_table"],
                    target_table=r["target_table"],
                    source_column=r["source_column"],
                    target_column=r["target_column"],
                    relationship_type=RelationshipType(r["relationship_type"]),
                )
            )
        return cls(
            name=data["name"],
            tables=[TableSchema.from_dict(t) for t in data.get("tables", [])],
            relationships=relationships,
            description=data.get("description"),
            domain=data.get("domain"),
        )

    def to_json(self, path: str) -> None:
        """Save schema to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DatabaseSchema":
        """Load schema from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Domain-specific templates
DOMAIN_TEMPLATES: Dict[str, DatabaseSchema] = {}


def _init_templates() -> None:
    """Initialize domain templates."""
    global DOMAIN_TEMPLATES

    # E-commerce template
    ecommerce = DatabaseSchema(
        name="ecommerce",
        domain="e-commerce",
        description="E-commerce platform with users, products, orders",
    )
    ecommerce.add_table(
        TableSchema(
            name="users",
            description="Customer accounts",
            columns=[
                ColumnSchema("user_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("email", ColumnDataType.EMAIL, unique=True),
                ColumnSchema("name", ColumnDataType.NAME),
                ColumnSchema("created_at", ColumnDataType.DATETIME),
                ColumnSchema("is_active", ColumnDataType.BOOLEAN, default=True),
            ],
        )
    )
    ecommerce.add_table(
        TableSchema(
            name="products",
            description="Product catalog",
            columns=[
                ColumnSchema("product_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("name", ColumnDataType.STRING),
                ColumnSchema("description", ColumnDataType.TEXT, nullable=True),
                ColumnSchema("price", ColumnDataType.CURRENCY, min_value=0),
                ColumnSchema(
                    "category",
                    ColumnDataType.CATEGORICAL,
                    categories=["Electronics", "Clothing", "Home", "Books", "Sports"],
                ),
                ColumnSchema("stock_quantity", ColumnDataType.INTEGER, min_value=0),
            ],
        )
    )
    ecommerce.add_table(
        TableSchema(
            name="orders",
            description="Customer orders",
            columns=[
                ColumnSchema("order_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("user_id", ColumnDataType.UUID, foreign_key="users.user_id"),
                ColumnSchema("order_date", ColumnDataType.DATETIME),
                ColumnSchema("total_amount", ColumnDataType.CURRENCY, min_value=0),
                ColumnSchema(
                    "status",
                    ColumnDataType.CATEGORICAL,
                    categories=["pending", "processing", "shipped", "delivered", "cancelled"],
                ),
            ],
        )
    )
    ecommerce.add_table(
        TableSchema(
            name="order_items",
            description="Items within orders",
            columns=[
                ColumnSchema("item_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("order_id", ColumnDataType.UUID, foreign_key="orders.order_id"),
                ColumnSchema("product_id", ColumnDataType.UUID, foreign_key="products.product_id"),
                ColumnSchema("quantity", ColumnDataType.INTEGER, min_value=1),
                ColumnSchema("unit_price", ColumnDataType.CURRENCY, min_value=0),
            ],
        )
    )
    ecommerce.add_relationship(
        Relationship("user_orders", "orders", "users", "user_id", "user_id", RelationshipType.MANY_TO_MANY)
    )
    ecommerce.add_relationship(
        Relationship("order_products", "order_items", "products", "product_id", "product_id", RelationshipType.MANY_TO_MANY)
    )
    DOMAIN_TEMPLATES["ecommerce"] = ecommerce
    DOMAIN_TEMPLATES["e-commerce"] = ecommerce

    # Healthcare template
    healthcare = DatabaseSchema(
        name="healthcare",
        domain="healthcare",
        description="Healthcare system with patients, providers, and records",
    )
    healthcare.add_table(
        TableSchema(
            name="patients",
            description="Patient demographics",
            columns=[
                ColumnSchema("patient_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("mrn", ColumnDataType.STRING, unique=True, description="Medical Record Number"),
                ColumnSchema("first_name", ColumnDataType.NAME),
                ColumnSchema("last_name", ColumnDataType.NAME),
                ColumnSchema("date_of_birth", ColumnDataType.DATE),
                ColumnSchema("gender", ColumnDataType.CATEGORICAL, categories=["M", "F", "O"]),
                ColumnSchema("blood_type", ColumnDataType.CATEGORICAL, categories=["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
                ColumnSchema("phone", ColumnDataType.PHONE, nullable=True),
                ColumnSchema("email", ColumnDataType.EMAIL, nullable=True),
            ],
        )
    )
    healthcare.add_table(
        TableSchema(
            name="providers",
            description="Healthcare providers",
            columns=[
                ColumnSchema("provider_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("npi", ColumnDataType.STRING, unique=True, description="National Provider Identifier"),
                ColumnSchema("name", ColumnDataType.NAME),
                ColumnSchema("specialty", ColumnDataType.CATEGORICAL, categories=["Primary Care", "Cardiology", "Oncology", "Neurology", "Pediatrics", "Surgery"]),
                ColumnSchema("email", ColumnDataType.EMAIL),
            ],
        )
    )
    healthcare.add_table(
        TableSchema(
            name="encounters",
            description="Patient encounters/visits",
            columns=[
                ColumnSchema("encounter_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("patient_id", ColumnDataType.UUID, foreign_key="patients.patient_id"),
                ColumnSchema("provider_id", ColumnDataType.UUID, foreign_key="providers.provider_id"),
                ColumnSchema("encounter_date", ColumnDataType.DATETIME),
                ColumnSchema("encounter_type", ColumnDataType.CATEGORICAL, categories=["office_visit", "emergency", "inpatient", "telehealth"]),
                ColumnSchema("chief_complaint", ColumnDataType.TEXT),
                ColumnSchema("diagnosis_code", ColumnDataType.STRING, nullable=True),
            ],
        )
    )
    healthcare.add_table(
        TableSchema(
            name="lab_results",
            description="Laboratory test results",
            columns=[
                ColumnSchema("result_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("encounter_id", ColumnDataType.UUID, foreign_key="encounters.encounter_id"),
                ColumnSchema("test_name", ColumnDataType.STRING),
                ColumnSchema("test_code", ColumnDataType.STRING),
                ColumnSchema("result_value", ColumnDataType.FLOAT),
                ColumnSchema("unit", ColumnDataType.STRING),
                ColumnSchema("reference_range", ColumnDataType.STRING, nullable=True),
                ColumnSchema("result_date", ColumnDataType.DATETIME),
            ],
        )
    )
    DOMAIN_TEMPLATES["healthcare"] = healthcare
    DOMAIN_TEMPLATES["medical"] = healthcare

    # Finance template
    finance = DatabaseSchema(
        name="finance",
        domain="finance",
        description="Financial services with accounts and transactions",
    )
    finance.add_table(
        TableSchema(
            name="customers",
            description="Bank customers",
            columns=[
                ColumnSchema("customer_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("name", ColumnDataType.NAME),
                ColumnSchema("email", ColumnDataType.EMAIL, unique=True),
                ColumnSchema("phone", ColumnDataType.PHONE),
                ColumnSchema("date_of_birth", ColumnDataType.DATE),
                ColumnSchema("credit_score", ColumnDataType.INTEGER, min_value=300, max_value=850),
                ColumnSchema("customer_since", ColumnDataType.DATE),
            ],
        )
    )
    finance.add_table(
        TableSchema(
            name="accounts",
            description="Financial accounts",
            columns=[
                ColumnSchema("account_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("customer_id", ColumnDataType.UUID, foreign_key="customers.customer_id"),
                ColumnSchema("account_type", ColumnDataType.CATEGORICAL, categories=["checking", "savings", "credit", "investment"]),
                ColumnSchema("account_number", ColumnDataType.STRING, unique=True),
                ColumnSchema("balance", ColumnDataType.CURRENCY),
                ColumnSchema("opened_date", ColumnDataType.DATE),
                ColumnSchema("status", ColumnDataType.CATEGORICAL, categories=["active", "dormant", "closed"]),
            ],
        )
    )
    finance.add_table(
        TableSchema(
            name="transactions",
            description="Financial transactions",
            columns=[
                ColumnSchema("transaction_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("account_id", ColumnDataType.UUID, foreign_key="accounts.account_id"),
                ColumnSchema("transaction_date", ColumnDataType.DATETIME),
                ColumnSchema("amount", ColumnDataType.CURRENCY),
                ColumnSchema("transaction_type", ColumnDataType.CATEGORICAL, categories=["deposit", "withdrawal", "transfer", "payment", "fee"]),
                ColumnSchema("description", ColumnDataType.STRING, nullable=True),
                ColumnSchema("merchant_name", ColumnDataType.STRING, nullable=True),
                ColumnSchema("is_fraud", ColumnDataType.BOOLEAN, default=False),
            ],
        )
    )
    DOMAIN_TEMPLATES["finance"] = finance
    DOMAIN_TEMPLATES["banking"] = finance

    # HR/Employee template
    hr = DatabaseSchema(
        name="hr",
        domain="human_resources",
        description="HR system with employees and departments",
    )
    hr.add_table(
        TableSchema(
            name="departments",
            description="Company departments",
            columns=[
                ColumnSchema("department_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("name", ColumnDataType.STRING),
                ColumnSchema("budget", ColumnDataType.CURRENCY, min_value=0),
                ColumnSchema("location", ColumnDataType.STRING),
            ],
        )
    )
    hr.add_table(
        TableSchema(
            name="employees",
            description="Employee records",
            columns=[
                ColumnSchema("employee_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("department_id", ColumnDataType.UUID, foreign_key="departments.department_id"),
                ColumnSchema("first_name", ColumnDataType.NAME),
                ColumnSchema("last_name", ColumnDataType.NAME),
                ColumnSchema("email", ColumnDataType.EMAIL, unique=True),
                ColumnSchema("hire_date", ColumnDataType.DATE),
                ColumnSchema("job_title", ColumnDataType.STRING),
                ColumnSchema("salary", ColumnDataType.CURRENCY, min_value=0),
                ColumnSchema("manager_id", ColumnDataType.UUID, foreign_key="employees.employee_id", nullable=True),
            ],
        )
    )
    DOMAIN_TEMPLATES["hr"] = hr
    DOMAIN_TEMPLATES["human_resources"] = hr
    DOMAIN_TEMPLATES["employees"] = hr


# Initialize templates
_init_templates()


class ZeroShotSchemaGenerator:
    """Generate schemas from natural language descriptions.

    Example:
        >>> generator = ZeroShotSchemaGenerator()
        >>> schema = generator.generate_schema(
        ...     "A social media platform with users, posts, comments, and likes"
        ... )
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4",
    ) -> None:
        """Initialize schema generator.

        Args:
            llm_provider: LLM provider (openai, anthropic, or local)
            api_key: API key for LLM provider
            model: Model name to use
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        self._client = None

    def _init_llm(self) -> None:
        """Initialize LLM client."""
        if self.llm_provider == "openai":
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        elif self.llm_provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        else:
            # Use rule-based fallback
            self._client = None

    def generate_schema(
        self,
        description: str,
        use_templates: bool = True,
        domain_hint: Optional[str] = None,
    ) -> DatabaseSchema:
        """Generate a database schema from natural language description.

        Args:
            description: Natural language description of the data
            use_templates: Whether to use domain templates as starting points
            domain_hint: Optional hint for which domain template to use

        Returns:
            Generated database schema
        """
        # Check for domain template match
        if use_templates:
            domain = domain_hint or self._detect_domain(description)
            if domain and domain in DOMAIN_TEMPLATES:
                logger.info(f"Using '{domain}' template as base")
                base_schema = DOMAIN_TEMPLATES[domain]
                return self._customize_schema(base_schema, description)

        # Use LLM or rule-based generation
        if self._client is None:
            self._init_llm()

        if self._client is not None:
            return self._generate_with_llm(description)
        else:
            return self._generate_rule_based(description)

    def _detect_domain(self, description: str) -> Optional[str]:
        """Detect domain from description keywords."""
        description_lower = description.lower()

        domain_keywords = {
            "ecommerce": ["e-commerce", "ecommerce", "shop", "cart", "product", "order", "customer"],
            "healthcare": ["patient", "doctor", "medical", "health", "hospital", "diagnosis", "lab"],
            "finance": ["bank", "account", "transaction", "payment", "credit", "debit", "finance"],
            "hr": ["employee", "department", "salary", "hire", "hr", "human resource"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in description_lower for kw in keywords):
                return domain

        return None

    def _customize_schema(self, base_schema: DatabaseSchema, description: str) -> DatabaseSchema:
        """Customize a template schema based on description."""
        # For now, return the template as-is
        # Could be enhanced to add/remove tables based on description
        return DatabaseSchema(
            name=base_schema.name + "_customized",
            tables=base_schema.tables.copy(),
            relationships=base_schema.relationships.copy(),
            description=description,
            domain=base_schema.domain,
        )

    def _generate_with_llm(self, description: str) -> DatabaseSchema:
        """Generate schema using LLM."""
        prompt = f"""Generate a database schema for the following description:

{description}

Return the schema as a JSON object with this structure:
{{
    "name": "schema_name",
    "tables": [
        {{
            "name": "table_name",
            "description": "table description",
            "columns": [
                {{
                    "name": "column_name",
                    "data_type": "one of: integer, float, string, boolean, date, datetime, email, phone, name, uuid, categorical, text, currency",
                    "nullable": true/false,
                    "unique": true/false,
                    "primary_key": true/false,
                    "foreign_key": "other_table.column" or null,
                    "categories": ["cat1", "cat2"] for categorical columns
                }}
            ]
        }}
    ],
    "relationships": [
        {{
            "name": "relationship_name",
            "source_table": "table1",
            "target_table": "table2",
            "source_column": "fk_column",
            "target_column": "pk_column",
            "relationship_type": "one_to_many"
        }}
    ]
}}

Only return valid JSON, no other text."""

        try:
            if self.llm_provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = response.choices[0].message.content
            elif self.llm_provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
            else:
                return self._generate_rule_based(description)

            # Parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                schema_dict = json.loads(json_match.group())
                return DatabaseSchema.from_dict(schema_dict)

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, falling back to rule-based")

        return self._generate_rule_based(description)

    def _generate_rule_based(self, description: str) -> DatabaseSchema:
        """Generate schema using rule-based approach."""
        # Extract potential table names from description
        words = re.findall(r"\b[a-zA-Z]+\b", description.lower())

        # Common entity keywords
        entity_keywords = {
            "user": ["user", "users", "customer", "customers", "person", "people", "member", "members"],
            "product": ["product", "products", "item", "items", "good", "goods"],
            "order": ["order", "orders", "purchase", "purchases"],
            "transaction": ["transaction", "transactions", "payment", "payments"],
            "post": ["post", "posts", "article", "articles", "content"],
            "comment": ["comment", "comments", "review", "reviews"],
            "category": ["category", "categories", "type", "types"],
            "location": ["location", "locations", "address", "addresses"],
        }

        detected_entities = []
        for entity, keywords in entity_keywords.items():
            if any(kw in words for kw in keywords):
                detected_entities.append(entity)

        # Default to basic user/item schema if nothing detected
        if not detected_entities:
            detected_entities = ["user", "item"]

        # Build schema
        schema = DatabaseSchema(
            name="generated_schema",
            description=description,
        )

        for entity in detected_entities:
            table = self._create_table_for_entity(entity)
            schema.add_table(table)

        # Add relationships
        if "user" in detected_entities:
            for entity in detected_entities:
                if entity != "user":
                    schema.add_relationship(
                        Relationship(
                            name=f"user_{entity}s",
                            source_table=f"{entity}s",
                            target_table="users",
                            source_column="user_id",
                            target_column="user_id",
                            relationship_type=RelationshipType.MANY_TO_MANY,
                        )
                    )

        return schema

    def _create_table_for_entity(self, entity: str) -> TableSchema:
        """Create a table schema for a common entity type."""
        table_name = f"{entity}s" if not entity.endswith("s") else entity

        # Common columns for each entity type
        entity_columns = {
            "user": [
                ColumnSchema("user_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("email", ColumnDataType.EMAIL, unique=True),
                ColumnSchema("name", ColumnDataType.NAME),
                ColumnSchema("created_at", ColumnDataType.DATETIME),
                ColumnSchema("is_active", ColumnDataType.BOOLEAN, default=True),
            ],
            "product": [
                ColumnSchema("product_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("name", ColumnDataType.STRING),
                ColumnSchema("description", ColumnDataType.TEXT, nullable=True),
                ColumnSchema("price", ColumnDataType.CURRENCY, min_value=0),
                ColumnSchema("category", ColumnDataType.CATEGORICAL, categories=["A", "B", "C"]),
            ],
            "order": [
                ColumnSchema("order_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("user_id", ColumnDataType.UUID, foreign_key="users.user_id"),
                ColumnSchema("order_date", ColumnDataType.DATETIME),
                ColumnSchema("total", ColumnDataType.CURRENCY, min_value=0),
                ColumnSchema("status", ColumnDataType.CATEGORICAL, categories=["pending", "completed", "cancelled"]),
            ],
            "transaction": [
                ColumnSchema("transaction_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("user_id", ColumnDataType.UUID, foreign_key="users.user_id"),
                ColumnSchema("amount", ColumnDataType.CURRENCY),
                ColumnSchema("transaction_date", ColumnDataType.DATETIME),
                ColumnSchema("type", ColumnDataType.CATEGORICAL, categories=["credit", "debit"]),
            ],
            "post": [
                ColumnSchema("post_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("user_id", ColumnDataType.UUID, foreign_key="users.user_id"),
                ColumnSchema("title", ColumnDataType.STRING),
                ColumnSchema("content", ColumnDataType.TEXT),
                ColumnSchema("created_at", ColumnDataType.DATETIME),
            ],
            "comment": [
                ColumnSchema("comment_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("user_id", ColumnDataType.UUID, foreign_key="users.user_id"),
                ColumnSchema("post_id", ColumnDataType.UUID, foreign_key="posts.post_id", nullable=True),
                ColumnSchema("content", ColumnDataType.TEXT),
                ColumnSchema("created_at", ColumnDataType.DATETIME),
            ],
            "category": [
                ColumnSchema("category_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("name", ColumnDataType.STRING, unique=True),
                ColumnSchema("description", ColumnDataType.TEXT, nullable=True),
            ],
            "location": [
                ColumnSchema("location_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("name", ColumnDataType.STRING),
                ColumnSchema("address", ColumnDataType.ADDRESS),
                ColumnSchema("latitude", ColumnDataType.FLOAT, min_value=-90, max_value=90),
                ColumnSchema("longitude", ColumnDataType.FLOAT, min_value=-180, max_value=180),
            ],
        }

        columns = entity_columns.get(
            entity,
            [
                ColumnSchema(f"{entity}_id", ColumnDataType.UUID, primary_key=True, unique=True),
                ColumnSchema("name", ColumnDataType.STRING),
                ColumnSchema("created_at", ColumnDataType.DATETIME),
            ],
        )

        return TableSchema(name=table_name, columns=columns)

    def get_available_templates(self) -> List[str]:
        """Get list of available domain templates."""
        return list(set(DOMAIN_TEMPLATES.keys()))

    def get_template(self, domain: str) -> Optional[DatabaseSchema]:
        """Get a specific domain template."""
        return DOMAIN_TEMPLATES.get(domain)


class ZeroShotDataGenerator:
    """Generate synthetic data from a schema without real data.

    Example:
        >>> schema = ZeroShotSchemaGenerator().generate_schema("e-commerce")
        >>> generator = ZeroShotDataGenerator()
        >>> data = generator.generate(schema, n_samples=1000)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize data generator.

        Args:
            seed: Random seed
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        schema: DatabaseSchema,
        n_samples: Optional[int] = None,
        table_sizes: Optional[Dict[str, int]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data from schema.

        Args:
            schema: Database schema
            n_samples: Default number of samples per table
            table_sizes: Override sizes for specific tables

        Returns:
            Dictionary of table_name -> DataFrame
        """
        n_samples = n_samples or 100
        table_sizes = table_sizes or {}

        # Generate primary tables first, then dependent tables
        tables_data: Dict[str, pd.DataFrame] = {}
        generated_order = self._get_generation_order(schema)

        for table_name in generated_order:
            table = schema.get_table(table_name)
            if table is None:
                continue

            size = table_sizes.get(table_name, table.row_count_estimate or n_samples)
            tables_data[table_name] = self._generate_table(table, size, tables_data)

        return tables_data

    def _get_generation_order(self, schema: DatabaseSchema) -> List[str]:
        """Determine order to generate tables (dependencies first)."""
        # Build dependency graph
        dependencies: Dict[str, List[str]] = {t.name: [] for t in schema.tables}

        for table in schema.tables:
            for col in table.columns:
                if col.foreign_key:
                    ref_table = col.foreign_key.split(".")[0]
                    if ref_table != table.name:  # Ignore self-references
                        dependencies[table.name].append(ref_table)

        # Topological sort
        order = []
        visited = set()

        def visit(table_name: str) -> None:
            if table_name in visited:
                return
            visited.add(table_name)
            for dep in dependencies.get(table_name, []):
                visit(dep)
            order.append(table_name)

        for table in schema.tables:
            visit(table.name)

        return order

    def _generate_table(
        self,
        table: TableSchema,
        n_rows: int,
        existing_tables: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate data for a single table."""
        data: Dict[str, Any] = {}

        for column in table.columns:
            if column.foreign_key:
                # Reference existing table
                ref_table, ref_col = column.foreign_key.split(".")
                if ref_table in existing_tables:
                    ref_values = existing_tables[ref_table][ref_col].values
                    data[column.name] = self._rng.choice(ref_values, n_rows, replace=True)
                else:
                    data[column.name] = self._generate_column(column, n_rows)
            else:
                data[column.name] = self._generate_column(column, n_rows)

        return pd.DataFrame(data)

    def _generate_column(self, column: ColumnSchema, n_rows: int) -> np.ndarray:
        """Generate data for a single column."""
        dtype = column.data_type

        if dtype == ColumnDataType.UUID:
            import uuid
            return np.array([str(uuid.uuid4()) for _ in range(n_rows)])

        elif dtype == ColumnDataType.INTEGER:
            min_val = int(column.min_value or 0)
            max_val = int(column.max_value or 1000000)
            return self._rng.integers(min_val, max_val, n_rows)

        elif dtype == ColumnDataType.FLOAT:
            min_val = column.min_value or 0.0
            max_val = column.max_value or 1000.0
            return self._rng.uniform(min_val, max_val, n_rows)

        elif dtype == ColumnDataType.CURRENCY:
            min_val = column.min_value or 0.0
            max_val = column.max_value or 10000.0
            return np.round(self._rng.uniform(min_val, max_val, n_rows), 2)

        elif dtype == ColumnDataType.BOOLEAN:
            return self._rng.choice([True, False], n_rows)

        elif dtype == ColumnDataType.CATEGORICAL:
            categories = column.categories or ["A", "B", "C"]
            return self._rng.choice(categories, n_rows)

        elif dtype == ColumnDataType.DATE:
            import datetime
            start = datetime.date(2020, 1, 1)
            days = self._rng.integers(0, 1500, n_rows)
            return np.array([start + datetime.timedelta(days=int(d)) for d in days])

        elif dtype == ColumnDataType.DATETIME:
            import datetime
            start = datetime.datetime(2020, 1, 1)
            seconds = self._rng.integers(0, 150000000, n_rows)
            return np.array([start + datetime.timedelta(seconds=int(s)) for s in seconds])

        elif dtype == ColumnDataType.EMAIL:
            domains = ["gmail.com", "yahoo.com", "outlook.com", "example.com"]
            names = [f"user{i}" for i in range(n_rows)]
            return np.array([f"{name}@{self._rng.choice(domains)}" for name in names])

        elif dtype == ColumnDataType.PHONE:
            return np.array([
                f"+1-{self._rng.integers(200,999)}-{self._rng.integers(100,999)}-{self._rng.integers(1000,9999)}"
                for _ in range(n_rows)
            ])

        elif dtype == ColumnDataType.NAME:
            first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller", "Wilson"]
            return np.array([
                f"{self._rng.choice(first_names)} {self._rng.choice(last_names)}"
                for _ in range(n_rows)
            ])

        elif dtype == ColumnDataType.ADDRESS:
            streets = ["Main St", "Oak Ave", "Park Rd", "Maple Dr", "Cedar Ln"]
            cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
            return np.array([
                f"{self._rng.integers(100,9999)} {self._rng.choice(streets)}, {self._rng.choice(cities)}"
                for _ in range(n_rows)
            ])

        elif dtype in (ColumnDataType.STRING, ColumnDataType.TEXT):
            return np.array([f"text_{i}" for i in range(n_rows)])

        elif dtype == ColumnDataType.URL:
            domains = ["example.com", "test.org", "sample.net"]
            return np.array([
                f"https://www.{self._rng.choice(domains)}/page{self._rng.integers(1,1000)}"
                for _ in range(n_rows)
            ])

        elif dtype == ColumnDataType.IP_ADDRESS:
            return np.array([
                f"{self._rng.integers(1,255)}.{self._rng.integers(0,255)}.{self._rng.integers(0,255)}.{self._rng.integers(0,255)}"
                for _ in range(n_rows)
            ])

        elif dtype == ColumnDataType.PERCENTAGE:
            return np.round(self._rng.uniform(0, 100, n_rows), 2)

        else:
            return np.array([f"value_{i}" for i in range(n_rows)])


def zero_shot_generate(
    description: str,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience function for zero-shot data generation.

    Args:
        description: Natural language description of the data
        n_samples: Number of samples per table
        seed: Random seed

    Returns:
        Dictionary of table_name -> DataFrame

    Example:
        >>> data = zero_shot_generate(
        ...     "An e-commerce platform with users, products, and orders",
        ...     n_samples=1000,
        ... )
        >>> print(data["users"].head())
    """
    schema_gen = ZeroShotSchemaGenerator()
    schema = schema_gen.generate_schema(description)

    data_gen = ZeroShotDataGenerator(seed=seed)
    return data_gen.generate(schema, n_samples=n_samples)
