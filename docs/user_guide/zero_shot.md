# Zero-Shot Schema Inference

Genesis provides zero-shot schema inference capabilities that can generate realistic synthetic data from natural language descriptions or minimal schema specifications, without requiring training data.

## Overview

| Component | Purpose |
|-----------|---------|
| **ZeroShotSchemaGenerator** | Infer schemas from descriptions |
| **ZeroShotDataGenerator** | Generate data without training |
| **DomainKnowledge** | Pre-built domain templates |
| **SchemaEnricher** | Enhance schemas with constraints |

## Schema Inference

Generate schemas from natural language descriptions:

```python
from genesis.zero_shot import ZeroShotSchemaGenerator

generator = ZeroShotSchemaGenerator()

# Infer schema from description
schema = generator.infer_schema(
    description="E-commerce customer database with purchase history",
    domain="ecommerce",
)

print("Inferred Schema:")
for column in schema.columns:
    print(f"  {column.name}: {column.dtype}")
    print(f"    Description: {column.description}")
    if column.constraints:
        print(f"    Constraints: {column.constraints}")
```

### Output Schema

```python
# Schema structure
InferredSchema(
    columns=[
        Column(
            name="customer_id",
            dtype="string",
            description="Unique customer identifier",
            constraints={"pattern": "CUST-[0-9]{8}"},
        ),
        Column(
            name="email",
            dtype="string",
            description="Customer email address",
            constraints={"format": "email", "unique": True},
        ),
        Column(
            name="registration_date",
            dtype="datetime",
            description="Account creation date",
            constraints={"min": "2020-01-01"},
        ),
        Column(
            name="total_purchases",
            dtype="integer",
            description="Total number of purchases",
            constraints={"min": 0},
        ),
        Column(
            name="lifetime_value",
            dtype="float",
            description="Customer lifetime value in USD",
            constraints={"min": 0, "max": 1000000},
        ),
    ],
    relationships=[
        Relationship(
            from_column="customer_id",
            to_table="orders",
            to_column="customer_id",
            type="one-to-many",
        ),
    ],
)
```

### Supported Domains

| Domain | Description | Example Entities |
|--------|-------------|------------------|
| `ecommerce` | Online retail | customers, orders, products, reviews |
| `healthcare` | Medical records | patients, visits, diagnoses, prescriptions |
| `finance` | Financial services | accounts, transactions, loans, investments |
| `hr` | Human resources | employees, departments, payroll, benefits |
| `iot` | IoT/sensors | devices, readings, alerts, locations |
| `generic` | General purpose | Auto-inferred based on description |

## Zero-Shot Data Generation

Generate synthetic data without training data:

```python
from genesis.zero_shot import ZeroShotDataGenerator

generator = ZeroShotDataGenerator()

# Generate from description
data = generator.generate(
    description="Customer support tickets with priority and resolution times",
    num_rows=1000,
)

print(data.head())
#    ticket_id  customer_id  priority     category  created_at  resolved_at  resolution_hours
# 0  TKT-00001     C-12345      high       billing  2026-01-15   2026-01-15              2.5
# 1  TKT-00002     C-67890    medium       support  2026-01-15   2026-01-16             18.3
# 2  TKT-00003     C-11111       low  feature_request  2026-01-15         NaT              NaN
```

### With Schema

```python
# Generate from explicit schema
schema = {
    "columns": [
        {"name": "product_id", "type": "string", "pattern": "PRD-[A-Z]{2}[0-9]{4}"},
        {"name": "product_name", "type": "string", "category": "product_name"},
        {"name": "price", "type": "float", "min": 0.99, "max": 9999.99},
        {"name": "category", "type": "categorical", "values": ["Electronics", "Clothing", "Home", "Sports"]},
        {"name": "in_stock", "type": "boolean", "true_ratio": 0.85},
    ]
}

data = generator.generate(
    schema=schema,
    num_rows=5000,
)
```

### With Relationships

```python
# Generate related tables
data = generator.generate_related(
    tables={
        "customers": {
            "description": "Customer information",
            "num_rows": 1000,
        },
        "orders": {
            "description": "Customer orders",
            "num_rows": 5000,
            "foreign_keys": [
                {"column": "customer_id", "references": "customers.customer_id"}
            ],
        },
        "order_items": {
            "description": "Items in each order",
            "num_rows": 15000,
            "foreign_keys": [
                {"column": "order_id", "references": "orders.order_id"}
            ],
        },
    }
)

customers = data["customers"]
orders = data["orders"]
order_items = data["order_items"]
```

## Domain Knowledge

Use pre-built domain templates for common use cases:

```python
from genesis.zero_shot import DomainKnowledge

# Load domain knowledge
domain = DomainKnowledge.load("ecommerce")

# View available entities
print("Available entities:")
for entity in domain.entities:
    print(f"  {entity.name}: {entity.description}")

# Generate using domain template
generator = ZeroShotDataGenerator(domain_knowledge=domain)

data = generator.generate(
    entity="customer",
    num_rows=10000,
    locale="en_US",
)
```

### Custom Domain Knowledge

```python
from genesis.zero_shot import DomainKnowledge, Entity, Attribute

# Define custom domain
domain = DomainKnowledge(
    name="insurance",
    entities=[
        Entity(
            name="policy",
            attributes=[
                Attribute("policy_number", "string", pattern="POL-[0-9]{10}"),
                Attribute("policy_type", "categorical", values=["auto", "home", "life", "health"]),
                Attribute("premium", "float", min=50, max=5000),
                Attribute("start_date", "date"),
                Attribute("end_date", "date"),
                Attribute("status", "categorical", values=["active", "expired", "cancelled"]),
            ],
        ),
        Entity(
            name="claim",
            attributes=[
                Attribute("claim_id", "string", pattern="CLM-[0-9]{8}"),
                Attribute("policy_number", "string", foreign_key="policy.policy_number"),
                Attribute("claim_amount", "float", min=100, max=500000),
                Attribute("claim_date", "date"),
                Attribute("status", "categorical", values=["pending", "approved", "denied"]),
            ],
        ),
    ],
)

# Save for reuse
domain.save("insurance_domain.json")

# Load later
domain = DomainKnowledge.load("insurance_domain.json")
```

## Schema Enrichment

Enhance minimal schemas with realistic constraints:

```python
from genesis.zero_shot import SchemaEnricher

enricher = SchemaEnricher()

# Minimal schema
minimal_schema = {
    "columns": [
        {"name": "email", "type": "string"},
        {"name": "age", "type": "integer"},
        {"name": "salary", "type": "float"},
    ]
}

# Enrich with constraints
enriched = enricher.enrich(minimal_schema)

print("Enriched Schema:")
for col in enriched.columns:
    print(f"  {col.name}:")
    print(f"    Type: {col.dtype}")
    print(f"    Constraints: {col.constraints}")
    print(f"    Generator: {col.generator_hint}")
```

### Enrichment Results

```python
# email column enriched
Column(
    name="email",
    dtype="string",
    constraints={
        "format": "email",
        "unique": True,
        "not_null": True,
    },
    generator_hint="faker.email",
)

# age column enriched
Column(
    name="age",
    dtype="integer",
    constraints={
        "min": 0,
        "max": 120,
        "distribution": "normal",
        "mean": 35,
        "std": 15,
    },
    generator_hint="age_distribution",
)

# salary column enriched
Column(
    name="salary",
    dtype="float",
    constraints={
        "min": 0,
        "max": 10000000,
        "distribution": "lognormal",
    },
    generator_hint="salary_distribution",
)
```

## Complete Example

```python
from genesis.zero_shot import (
    ZeroShotSchemaGenerator,
    ZeroShotDataGenerator,
    DomainKnowledge,
)

# Use healthcare domain
domain = DomainKnowledge.load("healthcare")

# Initialize generators
schema_gen = ZeroShotSchemaGenerator(domain_knowledge=domain)
data_gen = ZeroShotDataGenerator(domain_knowledge=domain)

# Infer schema from description
description = """
Hospital patient records including:
- Patient demographics
- Visit history with diagnoses
- Prescribed medications
- Insurance information
"""

schema = schema_gen.infer_schema(description)

print("=== Inferred Schema ===")
for table_name, table_schema in schema.tables.items():
    print(f"\nTable: {table_name}")
    for col in table_schema.columns:
        print(f"  {col.name}: {col.dtype}")

# Generate data
print("\n=== Generating Data ===")
data = data_gen.generate_related(
    schema=schema,
    row_counts={
        "patients": 10000,
        "visits": 50000,
        "diagnoses": 75000,
        "medications": 100000,
    },
)

# Verify referential integrity
print("\n=== Data Summary ===")
for table_name, df in data.items():
    print(f"{table_name}: {len(df)} rows, {len(df.columns)} columns")

# Validate foreign keys
patients_with_visits = data["visits"]["patient_id"].nunique()
total_patients = len(data["patients"])
print(f"\nPatients with visits: {patients_with_visits}/{total_patients}")

# Save data
for table_name, df in data.items():
    df.to_csv(f"{table_name}.csv", index=False)
    print(f"Saved {table_name}.csv")
```

## Configuration Reference

### ZeroShotSchemaGenerator

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain_knowledge` | DomainKnowledge | Pre-built domain templates |
| `language_model` | str | LLM for schema inference |
| `inference_mode` | str | Inference strategy (rule-based, ml, hybrid) |
| `locale` | str | Locale for generated values |

### ZeroShotDataGenerator

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain_knowledge` | DomainKnowledge | Pre-built domain templates |
| `faker_locale` | str | Faker locale for realistic values |
| `seed` | int | Random seed for reproducibility |
| `validate_constraints` | bool | Validate generated data |

### DomainKnowledge

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Domain name |
| `entities` | List[Entity] | Domain entities |
| `relationships` | List[Relationship] | Entity relationships |
| `business_rules` | List[Rule] | Domain-specific rules |

### SchemaEnricher

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain_knowledge` | DomainKnowledge | Domain context |
| `infer_constraints` | bool | Infer value constraints |
| `infer_distributions` | bool | Infer statistical distributions |
| `infer_relationships` | bool | Infer foreign keys |

## Best Practices

1. **Start with domain knowledge** - Use pre-built domains when available
2. **Provide detailed descriptions** - More context yields better schemas
3. **Validate inferred schemas** - Review and adjust before generating large datasets
4. **Use relationships** - Generate related tables together for referential integrity
5. **Iterate** - Refine descriptions based on initial results
