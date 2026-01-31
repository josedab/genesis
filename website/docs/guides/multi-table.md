---
sidebar_position: 5
title: Multi-Table
---

# Multi-Table Generation

Generate synthetic data for relational databases while preserving foreign key relationships and referential integrity.

## Quick Start

```python
from genesis import MultiTableGenerator

# Define tables and relationships
tables = {
    'customers': customers_df,
    'orders': orders_df,
    'order_items': items_df
}

relationships = [
    ('orders', 'customer_id', 'customers', 'id'),
    ('order_items', 'order_id', 'orders', 'id')
]

# Generate
generator = MultiTableGenerator()
generator.fit(tables, relationships)
synthetic = generator.generate(scale=1.0)  # Same size as original
```

## Defining Relationships

### One-to-Many

```python
# customers -> orders (one customer, many orders)
relationships = [
    ('orders', 'customer_id', 'customers', 'id')
]
```

### Many-to-Many

```python
# Through junction table
relationships = [
    ('student_courses', 'student_id', 'students', 'id'),
    ('student_courses', 'course_id', 'courses', 'id')
]
```

### Self-Referential

```python
# employees -> managers (both in employees table)
relationships = [
    ('employees', 'manager_id', 'employees', 'id')
]
```

### Complex Schemas

```python
relationships = [
    ('orders', 'customer_id', 'customers', 'id'),
    ('orders', 'product_id', 'products', 'id'),
    ('order_items', 'order_id', 'orders', 'id'),
    ('order_items', 'product_id', 'products', 'id'),
    ('products', 'category_id', 'categories', 'id'),
    ('customers', 'region_id', 'regions', 'id')
]
```

## Scaling

### Scale All Tables

```python
# 2x the original data
synthetic = generator.generate(scale=2.0)
```

### Scale Specific Tables

```python
# Different scale per table
synthetic = generator.generate(
    scale={
        'customers': 1.5,    # 50% more customers
        'orders': 2.0,       # 2x orders
        'order_items': 2.0   # 2x items
    }
)
```

### Fixed Row Counts

```python
synthetic = generator.generate(
    n_rows={
        'customers': 1000,
        'orders': 5000,
        'order_items': 15000
    }
)
```

## Preserving Relationships

### Cardinality Distribution

Genesis preserves relationship cardinality:

```python
# If original: avg 3.5 orders per customer
# Synthetic will have similar distribution

real_orders_per_customer = orders_df.groupby('customer_id').size()
syn_orders_per_customer = synthetic['orders'].groupby('customer_id').size()

print(f"Real avg: {real_orders_per_customer.mean():.1f}")
print(f"Synthetic avg: {syn_orders_per_customer.mean():.1f}")
```

### Referential Integrity

All foreign keys are valid:

```python
# All order customer_ids exist in customers
assert synthetic['orders']['customer_id'].isin(
    synthetic['customers']['id']
).all()
```

## Table-Specific Configuration

```python
generator = MultiTableGenerator(
    table_config={
        'customers': {
            'method': 'ctgan',
            'discrete_columns': ['segment', 'region']
        },
        'orders': {
            'method': 'tvae',
            'discrete_columns': ['status']
        },
        'order_items': {
            'method': 'gaussian_copula'
        }
    }
)
```

## Constraints Across Tables

```python
from genesis import Constraint

generator.fit(
    tables,
    relationships,
    constraints={
        'customers': [
            Constraint.unique('email'),
            Constraint.positive('age')
        ],
        'orders': [
            Constraint.positive('total_amount'),
            Constraint.not_null('customer_id')
        ],
        'order_items': [
            Constraint.positive('quantity'),
            Constraint.positive('unit_price')
        ]
    }
)
```

## Cross-Table Consistency

Ensure logical consistency:

```python
from genesis import CrossTableConstraint

# Order total = sum of items
class OrderTotalConstraint(CrossTableConstraint):
    def validate(self, tables):
        items = tables['order_items']
        orders = tables['orders']
        
        item_totals = items.groupby('order_id')['amount'].sum()
        return (orders.set_index('id')['total_amount'] == item_totals).all()

generator.fit(
    tables,
    relationships,
    cross_constraints=[OrderTotalConstraint()]
)
```

## Primary Key Generation

### Auto-Increment (Default)

```python
# IDs generated sequentially
generator.fit(tables, relationships, pk_strategy='auto_increment')
```

### UUID

```python
generator.fit(tables, relationships, pk_strategy='uuid')
```

### Custom

```python
def custom_pk_generator(table_name, n_rows):
    prefix = table_name[:3].upper()
    return [f"{prefix}-{i:06d}" for i in range(n_rows)]

generator.fit(tables, relationships, pk_generator=custom_pk_generator)
```

## Evaluation

```python
from genesis.evaluation import MultiTableMetrics

metrics = MultiTableMetrics(
    real_tables=tables,
    synthetic_tables=synthetic,
    relationships=relationships
)

# Overall quality
print(f"Overall Score: {metrics.overall_score():.1%}")

# Per-table quality
for table, score in metrics.table_scores().items():
    print(f"  {table}: {score:.1%}")

# Relationship preservation
for rel, score in metrics.relationship_scores().items():
    print(f"  {rel}: {score:.1%}")

# Cardinality match
print(f"Cardinality Match: {metrics.cardinality_score():.1%}")
```

## Complete Example

```python
import pandas as pd
from genesis import MultiTableGenerator
from genesis.evaluation import MultiTableMetrics

# Load tables
customers = pd.read_csv('customers.csv')
orders = pd.read_csv('orders.csv')
order_items = pd.read_csv('order_items.csv')
products = pd.read_csv('products.csv')

tables = {
    'customers': customers,
    'orders': orders,
    'order_items': order_items,
    'products': products
}

relationships = [
    ('orders', 'customer_id', 'customers', 'customer_id'),
    ('order_items', 'order_id', 'orders', 'order_id'),
    ('order_items', 'product_id', 'products', 'product_id')
]

# Create generator
generator = MultiTableGenerator(
    table_config={
        'customers': {'method': 'ctgan'},
        'orders': {'method': 'ctgan'},
        'order_items': {'method': 'gaussian_copula'},
        'products': {'method': 'ctgan'}
    }
)

# Fit
generator.fit(
    tables,
    relationships,
    discrete_columns={
        'customers': ['segment', 'country'],
        'orders': ['status', 'payment_method'],
        'products': ['category']
    }
)

# Generate 2x scale
synthetic = generator.generate(scale=2.0)

# Evaluate
metrics = MultiTableMetrics(tables, synthetic, relationships)
print(f"Quality: {metrics.overall_score():.1%}")

# Save
for name, df in synthetic.items():
    df.to_csv(f'synthetic_{name}.csv', index=False)
```

## Database Integration

### From SQL Database

```python
from genesis.io import load_from_database

tables, relationships = load_from_database(
    connection_string='postgresql://user:pass@localhost/mydb',
    schema='public',
    tables=['customers', 'orders', 'order_items']
)

generator.fit(tables, relationships)
```

### To SQL Database

```python
from genesis.io import save_to_database

save_to_database(
    synthetic,
    connection_string='postgresql://user:pass@localhost/synth_db',
    schema='public',
    if_exists='replace'
)
```

## Best Practices

1. **Start with parent tables** - Generate in dependency order
2. **Verify referential integrity** - Check foreign keys after generation
3. **Match cardinality distributions** - Key quality metric
4. **Use consistent constraints** - Across related columns
5. **Test with small scale first** - Before generating large datasets

## Troubleshooting

### Orphan records
- Check relationship definitions
- Verify parent table generates first

### Cardinality mismatch
- Increase training epochs
- Check if relationships are learned correctly

### Performance issues
- Generate smaller batches
- Use simpler models for large tables

## Next Steps

- **[Pipelines](/docs/guides/pipelines)** - Multi-table generation workflows
- **[Versioning](/docs/guides/versioning)** - Track multi-table datasets
- **[Constraints](/docs/concepts/constraints)** - Cross-table constraints
