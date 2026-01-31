---
sidebar_position: 1
title: Tabular Data
---

# Tabular Data Generation

Generate realistic tabular data that preserves statistical properties and relationships.

## Quick Start

```python
from genesis import SyntheticGenerator
import pandas as pd

# Load your data
df = pd.read_csv('customers.csv')

# Generate synthetic data
generator = SyntheticGenerator(method='ctgan')
generator.fit(df, discrete_columns=['status', 'region'])
synthetic = generator.generate(1000)
```

## Specifying Column Types

Genesis auto-detects column types, but you can be explicit:

```python
generator.fit(
    df,
    discrete_columns=['category', 'status', 'region'],  # Categorical
    # Numeric columns are auto-detected
)
```

## Handling Different Data Types

### Numeric Columns

```python
# Integers and floats are handled automatically
df = pd.DataFrame({
    'age': [25, 30, 35],           # Integer
    'salary': [50000.0, 60000.0, 70000.0],  # Float
    'score': [0.85, 0.92, 0.78]    # Decimal
})

generator.fit(df)
synthetic = generator.generate(100)
```

### Categorical Columns

```python
# Mark discrete columns explicitly
generator.fit(
    df,
    discrete_columns=['city', 'department', 'status']
)
```

### Boolean Columns

```python
# Booleans are treated as categorical
df['is_active'] = [True, False, True, False]
generator.fit(df, discrete_columns=['is_active'])
```

### Datetime Columns

```python
# Convert datetime to numeric for better generation
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['signup_timestamp'] = df['signup_date'].astype(int) / 10**9

generator.fit(df.drop('signup_date', axis=1))
synthetic = generator.generate(100)

# Convert back
synthetic['signup_date'] = pd.to_datetime(synthetic['signup_timestamp'], unit='s')
```

## Adding Constraints

```python
from genesis import Constraint

generator.fit(
    df,
    discrete_columns=['status'],
    constraints=[
        Constraint.positive('age'),
        Constraint.range('age', 18, 100),
        Constraint.range('rating', 1, 5),
        Constraint.unique('customer_id')
    ]
)
```

## Handling Missing Values

```python
# Genesis handles NaN values automatically
df_with_nulls = pd.DataFrame({
    'name': ['Alice', None, 'Charlie'],
    'age': [25, 30, None],
    'city': ['NYC', 'LA', 'NYC']
})

generator.fit(df_with_nulls, discrete_columns=['city'])
synthetic = generator.generate(100)

# Synthetic data will have similar null patterns
print(synthetic.isna().sum())
```

## Preserving Correlations

CTGAN and TVAE automatically preserve correlations:

```python
import matplotlib.pyplot as plt

# Check correlation preservation
real_corr = df.corr()
syn_corr = synthetic.corr()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(real_corr, cmap='coolwarm')
axes[0].set_title('Real Data Correlations')
axes[1].imshow(syn_corr, cmap='coolwarm')
axes[1].set_title('Synthetic Data Correlations')
plt.show()
```

## High-Cardinality Categories

For categories with many unique values:

```python
# Use CTGAN for high-cardinality
generator = SyntheticGenerator(
    method='ctgan',
    config={
        'epochs': 500,  # More training for complex data
    }
)

# Or let AutoML decide
from genesis import auto_synthesize
synthetic = auto_synthesize(df, n_samples=1000)
```

## Large Datasets

For datasets with many rows:

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(
    method='ctgan',
    device='cuda',  # Use GPU
    batch_size=10000
)

generator.fit(large_df)
synthetic = generator.generate(1_000_000)
```

## Evaluating Quality

```python
from genesis import QualityEvaluator

evaluator = QualityEvaluator(df, synthetic)
report = evaluator.evaluate()

print(f"Overall: {report.overall_score:.1%}")
print(f"Fidelity: {report.fidelity_score:.1%}")

# Per-column quality
for col, metrics in report.column_metrics.items():
    print(f"{col}: {metrics['similarity']:.2f}")
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator, Constraint, QualityEvaluator

# Load data
df = pd.read_csv('ecommerce_customers.csv')

# Prepare
discrete_cols = ['region', 'subscription_type', 'churn']

# Create generator with constraints
generator = SyntheticGenerator(
    method='ctgan',
    config={'epochs': 300}
)

generator.fit(
    df,
    discrete_columns=discrete_cols,
    constraints=[
        Constraint.positive('age'),
        Constraint.range('age', 18, 100),
        Constraint.positive('total_spend'),
        Constraint.unique('customer_id')
    ]
)

# Generate
synthetic = generator.generate(len(df))

# Evaluate
report = QualityEvaluator(df, synthetic).evaluate(target_column='churn')
print(report.summary())

# Save
synthetic.to_csv('synthetic_customers.csv', index=False)
```

## Best Practices

1. **Specify discrete columns** - Don't rely solely on auto-detection
2. **Use constraints** - Enforce business rules
3. **Evaluate quality** - Always check output
4. **Handle dates carefully** - Convert to timestamps
5. **Use AutoML for unfamiliar data** - Let Genesis choose the method

## Next Steps

- **[Conditional Generation](/docs/guides/conditional-generation)** - Generate specific scenarios
- **[Multi-Table](/docs/guides/multi-table)** - Relational data with foreign keys
- **[Constraints](/docs/concepts/constraints)** - Advanced constraint usage
