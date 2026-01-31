# LLM-Powered Schema Inference

Genesis can use Large Language Models to automatically infer rich column schemas from column names and sample data, generating realistic synthetic data without manual configuration.

## Overview

LLM schema inference analyzes column names and sample values to determine:
- Semantic data types (email, phone, address, etc.)
- Value patterns and constraints
- Realistic generation strategies

```python
from genesis.llm_inference import LLMSchemaInferrer

inferrer = LLMSchemaInferrer(api_key="your-api-key")
schema = inferrer.infer(df)

# Schema now contains semantic types for each column
for col_name, col_schema in schema.items():
    print(f"{col_name}: {col_schema.semantic_type}")
```

## Supported Semantic Types

Genesis recognizes 30+ semantic types:

| Category | Types |
|----------|-------|
| **Personal** | first_name, last_name, full_name, email, phone, ssn, age, birth_date, gender |
| **Location** | address, street, city, state, country, zipcode, latitude, longitude |
| **Financial** | credit_card, account_number, routing_number, iban, amount, price, currency |
| **Temporal** | date, datetime, timestamp, time, year, month, day, duration |
| **Technical** | uuid, ip_address, mac_address, url, domain, user_agent |
| **Business** | company_name, job_title, department, employee_id, product_id |
| **Text** | text, paragraph, sentence, word, description, comment |

## Quick Start

### With OpenAI

```python
from genesis.llm_inference import LLMSchemaInferrer
import os

inferrer = LLMSchemaInferrer(
    provider="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini"  # or gpt-4o for better accuracy
)

schema = inferrer.infer(df)
```

### With Anthropic

```python
inferrer = LLMSchemaInferrer(
    provider="anthropic",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model="claude-3-haiku-20240307"
)

schema = inferrer.infer(df)
```

### Without LLM (Rule-based)

For environments without API access, use rule-based inference:

```python
from genesis.llm_inference import RuleBasedInferrer

inferrer = RuleBasedInferrer()
schema = inferrer.infer(df)
```

## Using Inferred Schemas

```python
from genesis.llm_inference import LLMSchemaInferrer
from genesis import SyntheticGenerator

# Infer schema
inferrer = LLMSchemaInferrer(api_key=api_key)
schema = inferrer.infer(df)

# Use schema for generation
generator = SyntheticGenerator(method="ctgan")
generator.fit(df, schema=schema)
synthetic = generator.generate(1000)
```

## Schema Output

```python
schema = inferrer.infer(df)

# Each column has rich metadata
for col_name, col_schema in schema.items():
    print(f"""
Column: {col_name}
  Semantic Type: {col_schema.semantic_type}
  Data Type: {col_schema.data_type}
  Nullable: {col_schema.nullable}
  Unique: {col_schema.unique}
  Pattern: {col_schema.pattern}
  Constraints: {col_schema.constraints}
""")
```

## Rule-Based Patterns

The rule-based inferrer recognizes patterns without API calls:

```python
from genesis.llm_inference import RuleBasedInferrer

inferrer = RuleBasedInferrer()

# Recognizes 30+ patterns from column names
df = pd.DataFrame({
    "user_email": ["alice@example.com"],
    "phone_number": ["555-123-4567"],
    "ip_address": ["192.168.1.1"],
    "created_at": ["2024-01-15"],
    "account_balance": [1234.56],
})

schema = inferrer.infer(df)
# user_email -> email
# phone_number -> phone
# ip_address -> ip_address
# created_at -> datetime
# account_balance -> amount
```

## Pattern Recognition Rules

| Column Name Pattern | Inferred Type |
|--------------------|---------------|
| `*email*`, `*mail*` | email |
| `*phone*`, `*mobile*`, `*tel*` | phone |
| `*ssn*`, `*social_security*` | ssn |
| `*first_name*`, `*fname*` | first_name |
| `*last_name*`, `*lname*`, `*surname*` | last_name |
| `*address*`, `*street*` | address |
| `*city*` | city |
| `*state*`, `*province*` | state |
| `*zip*`, `*postal*` | zipcode |
| `*country*` | country |
| `*lat*`, `*latitude*` | latitude |
| `*lon*`, `*lng*`, `*longitude*` | longitude |
| `*ip_addr*`, `*ip_address*` | ip_address |
| `*url*`, `*link*`, `*website*` | url |
| `*uuid*`, `*guid*` | uuid |
| `*date*`, `*_at`, `*_on` | datetime |
| `*price*`, `*amount*`, `*cost*`, `*total*` | amount |
| `*credit_card*`, `*cc_number*` | credit_card |
| `*company*`, `*organization*` | company_name |
| `*job*`, `*title*`, `*position*` | job_title |
| `*age*` | age |
| `*gender*`, `*sex*` | gender |

## Batch Inference

For multiple datasets:

```python
from genesis.llm_inference import LLMSchemaInferrer

inferrer = LLMSchemaInferrer(api_key=api_key)

datasets = {
    "customers": customers_df,
    "orders": orders_df,
    "products": products_df
}

schemas = {}
for name, df in datasets.items():
    schemas[name] = inferrer.infer(df)
    print(f"Inferred {len(schemas[name])} columns for {name}")
```

## Customization

### Adding Custom Patterns

```python
from genesis.llm_inference import RuleBasedInferrer

inferrer = RuleBasedInferrer()

# Add custom pattern
inferrer.add_pattern(
    pattern=r".*mrn.*",  # Medical record number
    semantic_type="medical_record_number",
    data_type="string"
)

schema = inferrer.infer(df)
```

### Override Inferred Types

```python
schema = inferrer.infer(df)

# Override a specific column
schema["transaction_id"].semantic_type = "uuid"
schema["notes"].semantic_type = "paragraph"

# Use modified schema
generator.fit(df, schema=schema)
```

## Pipeline Integration

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("raw_data.csv")
    .add_node("infer", "infer_schema", {
        "provider": "openai",
        "model": "gpt-4o-mini"
    })
    .add_node("generate", "synthesize", {
        "n_samples": 10000
    })
    .sink("synthetic_data.csv")
    .build()
)

pipeline.execute()
```

## Best Practices

1. **Use LLM for complex schemas**: LLM inference handles ambiguous cases better
2. **Fall back to rules**: Use rule-based for cost savings and offline environments
3. **Review inferences**: Always review inferred types before generating sensitive data
4. **Cache schemas**: Save inferred schemas to avoid repeated API calls
5. **Provide context**: Better column names lead to better inferences

## Cost Optimization

```python
# Use smaller models for large datasets
inferrer = LLMSchemaInferrer(
    api_key=api_key,
    model="gpt-4o-mini",  # Cheaper than gpt-4o
    max_samples=5         # Only send 5 sample rows
)

# Cache results
schema = inferrer.infer(df)
schema.save("schema_cache.json")

# Load later
from genesis.llm_inference import Schema
schema = Schema.load("schema_cache.json")
```
