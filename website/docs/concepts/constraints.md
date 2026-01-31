---
sidebar_position: 5
title: Constraints
---

# Constraints

Constraints let you enforce business rules and data quality requirements on generated synthetic data.

## Why Use Constraints?

Without constraints, synthetic data might violate business rules:

```python
# Without constraints - might generate invalid data
synthetic = generator.generate(1000)
print(synthetic['age'].min())  # Could be -5 (invalid!)
print(synthetic['rating'].max())  # Could be 7 (should be 1-5)
```

With constraints:

```python
from genesis import Constraint

generator.fit(
    data,
    constraints=[
        Constraint.positive('age'),
        Constraint.range('rating', 1, 5)
    ]
)
synthetic = generator.generate(1000)
print(synthetic['age'].min())  # Always >= 0
print(synthetic['rating'].max())  # Always <= 5
```

## Built-in Constraints

### Positive Values

Ensure values are greater than zero:

```python
from genesis import Constraint

# Age must be positive
Constraint.positive('age')

# Strict: > 0 (default)
Constraint.positive('amount', strict=True)

# Non-strict: >= 0
Constraint.positive('quantity', strict=False)
```

### Range

Keep values within bounds:

```python
# Age between 0 and 120
Constraint.range('age', min=0, max=120)

# Rating from 1 to 5
Constraint.range('rating', min=1, max=5)

# Only minimum
Constraint.range('price', min=0)

# Only maximum
Constraint.range('discount_pct', max=100)
```

### Uniqueness

Ensure no duplicate values:

```python
# Each customer_id should be unique
Constraint.unique('customer_id')

# Combination must be unique
Constraint.unique(['order_id', 'product_id'])
```

### Categorical

Restrict to specific categories:

```python
# Status must be one of these values
Constraint.categorical('status', ['active', 'inactive', 'pending'])

# Country codes
Constraint.categorical('country', ['US', 'UK', 'CA', 'DE', 'FR'])
```

### Regex Pattern

Match a regular expression:

```python
# Email format
Constraint.regex('email', r'^[\w.-]+@[\w.-]+\.\w+$')

# Phone format
Constraint.regex('phone', r'^\d{3}-\d{3}-\d{4}$')

# Zip code
Constraint.regex('zip', r'^\d{5}(-\d{4})?$')
```

### Not Null

Ensure no missing values:

```python
# Customer ID cannot be null
Constraint.not_null('customer_id')

# Multiple columns
Constraint.not_null(['name', 'email', 'phone'])
```

## Applying Constraints

### During Fit

```python
from genesis import SyntheticGenerator, Constraint

generator = SyntheticGenerator(method='ctgan')
generator.fit(
    data,
    discrete_columns=['category', 'status'],
    constraints=[
        Constraint.positive('age'),
        Constraint.range('age', 0, 120),
        Constraint.range('rating', 1, 5),
        Constraint.unique('customer_id'),
        Constraint.not_null('email')
    ]
)
```

### Using ConstraintSet

```python
from genesis import ConstraintSet, Constraint

# Define reusable constraint sets
customer_constraints = ConstraintSet([
    Constraint.positive('age'),
    Constraint.range('age', 18, 120),
    Constraint.unique('customer_id'),
    Constraint.regex('email', r'^[\w.-]+@[\w.-]+\.\w+$'),
    Constraint.categorical('status', ['active', 'inactive'])
])

generator.fit(data, constraints=customer_constraints)
```

## Constraint Validation

Check if data satisfies constraints:

```python
from genesis import Constraint

constraint = Constraint.range('age', 0, 120)

# Validate existing data
is_valid = constraint.validate(synthetic_data)
print(f"Valid: {is_valid}")

# Get detailed validation
result = constraint.validate_detailed(synthetic_data)
print(f"Valid rows: {result['valid_count']}")
print(f"Invalid rows: {result['invalid_count']}")
print(f"Invalid indices: {result['invalid_indices']}")
```

## Constraint Transformation

Transform data to satisfy constraints:

```python
# Make values positive
constraint = Constraint.positive('age')
fixed_data = constraint.transform(synthetic_data)

# Clip to range
constraint = Constraint.range('rating', 1, 5)
fixed_data = constraint.transform(synthetic_data)
```

## Custom Constraints

Create your own constraint logic:

```python
from genesis.core.constraints import BaseConstraint

class IncrementingConstraint(BaseConstraint):
    """Ensure values are incrementing."""
    
    def __init__(self, column: str):
        super().__init__(column, 'incrementing')
    
    def validate(self, data):
        values = data[self.column].dropna()
        return (values.diff().dropna() >= 0).all()
    
    def transform(self, data):
        result = data.copy()
        result[self.column] = result[self.column].sort_values().values
        return result

# Use it
generator.fit(data, constraints=[
    IncrementingConstraint('timestamp')
])
```

## Conditional Constraints

Apply constraints based on conditions:

```python
from genesis import Constraint

# If premium=True, min_balance should be >= 10000
class ConditionalConstraint(BaseConstraint):
    def validate(self, data):
        premium = data[data['is_premium'] == True]
        return (premium['min_balance'] >= 10000).all()
    
    def transform(self, data):
        result = data.copy()
        mask = result['is_premium'] == True
        result.loc[mask & (result['min_balance'] < 10000), 'min_balance'] = 10000
        return result
```

## Cross-Column Constraints

Enforce relationships between columns:

```python
class DateOrderConstraint(BaseConstraint):
    """Ensure end_date >= start_date."""
    
    def __init__(self):
        super().__init__('end_date', 'date_order')
        self.start_col = 'start_date'
        self.end_col = 'end_date'
    
    def validate(self, data):
        return (data[self.end_col] >= data[self.start_col]).all()
    
    def transform(self, data):
        result = data.copy()
        invalid = result[self.end_col] < result[self.start_col]
        result.loc[invalid, self.end_col] = result.loc[invalid, self.start_col]
        return result
```

## Common Constraint Patterns

### Customer Data

```python
customer_constraints = [
    Constraint.positive('age'),
    Constraint.range('age', 18, 120),
    Constraint.unique('customer_id'),
    Constraint.regex('email', r'^[\w.-]+@[\w.-]+\.\w+$'),
    Constraint.categorical('status', ['active', 'inactive', 'suspended']),
    Constraint.not_null(['customer_id', 'email'])
]
```

### Financial Data

```python
transaction_constraints = [
    Constraint.positive('amount'),
    Constraint.range('amount', 0.01, 1_000_000),
    Constraint.unique('transaction_id'),
    Constraint.categorical('currency', ['USD', 'EUR', 'GBP']),
    Constraint.not_null(['transaction_id', 'amount', 'timestamp'])
]
```

### Healthcare Data

```python
patient_constraints = [
    Constraint.positive('age'),
    Constraint.range('age', 0, 120),
    Constraint.range('heart_rate', 30, 200),
    Constraint.range('blood_pressure_systolic', 70, 200),
    Constraint.unique('patient_id'),
    Constraint.not_null(['patient_id', 'dob'])
]
```

## Constraint Enforcement Strategies

### Reject and Resample

Generate more samples and filter:

```python
# Generate extra samples, filter invalid ones
synthetic = generator.generate(n_samples * 1.2)
for constraint in constraints:
    synthetic = synthetic[constraint.validate_mask(synthetic)]
synthetic = synthetic.head(n_samples)
```

### Transform After Generation

Fix invalid values:

```python
synthetic = generator.generate(n_samples)
for constraint in constraints:
    synthetic = constraint.transform(synthetic)
```

### During Training (Built-in)

Most effective - learned during fit:

```python
generator.fit(data, constraints=[...])  # Constraints learned
synthetic = generator.generate(n_samples)  # Naturally satisfied
```

## Debugging Constraints

```python
from genesis import Constraint

# Check which constraints fail
constraints = [
    Constraint.positive('age'),
    Constraint.range('rating', 1, 5),
    Constraint.unique('id')
]

for constraint in constraints:
    if not constraint.validate(synthetic_data):
        print(f"FAILED: {constraint}")
        result = constraint.validate_detailed(synthetic_data)
        print(f"  Invalid rows: {result['invalid_count']}")
```

## Best Practices

1. **Define constraints early** - Before fitting the generator
2. **Validate both real and synthetic data** - Ensure constraints match reality
3. **Use the strictest constraints needed** - Don't over-constrain
4. **Test constraint combinations** - Some constraints may conflict
5. **Document business rules** - Map constraints to requirements

## Next Steps

- **[Tabular Data Guide](/docs/guides/tabular-data)** - Put constraints to use
- **[Quality Evaluation](/docs/concepts/evaluation)** - Validate constraint satisfaction
- **[API Reference](/docs/api/reference)** - Full constraint API
