---
sidebar_position: 101
title: Troubleshooting
---

# Troubleshooting

Common issues and their solutions.

## Installation Issues

### pip install fails

**Error:** `Failed building wheel for sdv`

```bash
# Install build dependencies first
pip install wheel setuptools numpy

# Then install genesis
pip install genesis-synth
```

**Error:** `No matching distribution found`

```bash
# Check Python version (requires 3.8+)
python --version

# Use specific Python version
python3.10 -m pip install genesis-synth
```

### CUDA/GPU issues

**Error:** `CUDA not available`

```bash
# Check CUDA installation
nvidia-smi

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

**Error:** `CUDA out of memory`

```python
# Solution 1: Reduce batch size
generator = SyntheticGenerator(
    method='ctgan',
    config={'batch_size': 256}  # Default is 500
)

# Solution 2: Use CPU
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cpu'}
)

# Solution 3: Clear cache
import torch
torch.cuda.empty_cache()
```

## Generation Issues

### Poor quality synthetic data

**Symptom:** Low quality scores, unrealistic values

```python
# Check 1: Verify discrete columns are specified
generator.fit(
    data,
    discrete_columns=['status', 'category', 'region']  # Important!
)

# Check 2: Increase training epochs
generator = SyntheticGenerator(
    method='ctgan',
    config={'epochs': 500}  # Default is 300
)

# Check 3: Try different method
generator = SyntheticGenerator(method='tvae')  # Often better for complex data
```

### Synthetic data has wrong distributions

**Symptom:** Categories have wrong proportions

```python
# Check 1: Ensure enough training data
# Minimum 500-1000 rows recommended
if len(data) < 500:
    print("Warning: Small dataset may produce poor results")

# Check 2: Check for rare categories
for col in discrete_columns:
    counts = data[col].value_counts()
    rare = counts[counts < 10]
    if len(rare) > 0:
        print(f"Warning: {col} has rare categories: {rare.index.tolist()}")
```

### Mode collapse (all samples similar)

**Symptom:** Generated samples lack diversity

```python
# Solution 1: Increase PAC size
generator = SyntheticGenerator(
    method='ctgan',
    config={'pac': 20}  # Default is 10
)

# Solution 2: Increase discriminator steps
generator = SyntheticGenerator(
    method='ctgan',
    config={'discriminator_steps': 5}  # Default is 1
)

# Solution 3: Use different random seed
generator = SyntheticGenerator(
    method='ctgan',
    random_state=42
)
```

### NaN values in output

**Symptom:** Generated data contains NaN

```python
# Check 1: Original data has NaN (Genesis preserves null patterns)
print(data.isna().sum())

# Check 2: Drop NaN before training
clean_data = data.dropna()
generator.fit(clean_data)

# Check 3: Fill NaN with appropriate values
data_filled = data.fillna(data.median())
generator.fit(data_filled)
```

## Constraint Issues

### Constraints not satisfied

**Symptom:** Generated data violates constraints

```python
# Check 1: Verify constraint definition
from genesis import Constraint

constraints = [
    Constraint.positive('age'),
    Constraint.range('age', 18, 100),  # Both are needed
]

# Check 2: Use transform after generation
synthetic = generator.generate(1000)
for constraint in constraints:
    synthetic = constraint.transform(synthetic)

# Check 3: Generate more and filter
synthetic = generator.generate(2000)  # Generate extra
for constraint in constraints:
    synthetic = synthetic[constraint.validate_mask(synthetic)]
synthetic = synthetic.head(1000)  # Take what we need
```

### Conflicting constraints

**Symptom:** Cannot satisfy all constraints

```python
# Example of conflicting constraints
constraints = [
    Constraint.range('value', 0, 100),
    Constraint.range('value', 50, 150)  # Conflicts!
]

# Solution: Use non-overlapping constraints
constraints = [
    Constraint.range('value', 50, 100)  # Intersection
]
```

## Privacy Issues

### Privacy audit fails

**Symptom:** Low privacy score

```python
# Solution 1: Add differential privacy
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {
            'epsilon': 1.0,  # Lower = more private
            'delta': 1e-5
        }
    }
)

# Solution 2: Add k-anonymity
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'k_anonymity': {
            'k': 5,
            'quasi_identifiers': ['age', 'zip_code', 'gender']
        }
    }
)

# Solution 3: Suppress outliers
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'suppress_outliers': True,
        'min_category_count': 10
    }
)
```

### Membership inference attack succeeds

**Symptom:** High attack accuracy

```python
# Strengthen privacy settings
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {'epsilon': 0.5},  # Stricter
        'suppress_outliers': True
    }
)

# Verify after generation
from genesis import run_privacy_audit
report = run_privacy_audit(real, synthetic)
print(f"Privacy score: {report.overall_score:.1%}")
```

## Performance Issues

### Training is slow

```python
# Solution 1: Use GPU
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cuda'}
)

# Solution 2: Reduce epochs
generator = SyntheticGenerator(
    method='ctgan',
    config={'epochs': 100}  # Fewer epochs
)

# Solution 3: Sample training data
sample = data.sample(n=10000, random_state=42)
generator.fit(sample)

# Solution 4: Use faster method
generator = SyntheticGenerator(method='gaussian_copula')
```

### Generation is slow

```python
# Solution 1: Generate in batches
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(
    method='ctgan',
    device='cuda',
    batch_size=10000
)

# Solution 2: Use simpler method for large volumes
generator = SyntheticGenerator(method='gaussian_copula')
```

### Out of memory

```python
# Solution 1: Reduce batch size
config = {'batch_size': 128}

# Solution 2: Use mixed precision
config = {'mixed_precision': True}

# Solution 3: Process in chunks
chunks = []
for i in range(10):
    chunk = generator.generate(1000)
    chunks.append(chunk)
synthetic = pd.concat(chunks, ignore_index=True)
```

## Time Series Issues

### Poor temporal patterns

**Symptom:** Generated sequences lack realistic trends

```python
# Solution 1: Increase sequence length
generator = TimeSeriesGenerator()
generator.fit(data, sequence_length=100)  # Capture more history

# Solution 2: Increase model capacity
generator = TimeSeriesGenerator(
    config={
        'hidden_dim': 128,
        'n_layers': 3
    }
)
```

### All sequences identical

```python
# Solution 1: Increase temperature/variance
generator = TimeSeriesGenerator(
    config={'temperature': 1.2}
)

# Solution 2: Add noise to training
data_noisy = data + np.random.normal(0, 0.01, data.shape)
generator.fit(data_noisy)
```

## Multi-Table Issues

### Foreign key violations

**Symptom:** Child table references non-existent parent

```python
# Check 1: Verify relationship definition
relationships = [
    ('orders', 'customer_id', 'customers', 'id'),
    #  child    child_col      parent       parent_col
]

# Check 2: Verify column names match exactly
print(orders.columns)  # Should include 'customer_id'
print(customers.columns)  # Should include 'id'
```

### Cardinality mismatch

**Symptom:** Wrong number of child records per parent

```python
# Check original cardinality
original = orders.groupby('customer_id').size().describe()
synthetic_card = syn_orders.groupby('customer_id').size().describe()
print("Original:", original)
print("Synthetic:", synthetic_card)

# Solution: Increase training
generator = MultiTableGenerator(
    table_config={
        'orders': {'method': 'ctgan', 'epochs': 500}
    }
)
```

## Common Error Messages

### `ValueError: Unknown method 'xyz'`

```python
# Check available methods
from genesis import list_methods
print(list_methods())
# ['ctgan', 'tvae', 'gaussian_copula', 'copulagan']
```

### `RuntimeError: Generator not fitted`

```python
# Must call fit() before generate()
generator = SyntheticGenerator(method='ctgan')
generator.fit(data)  # Don't forget this!
synthetic = generator.generate(1000)
```

### `TypeError: cannot convert 'xxx' to numeric`

```python
# Check data types
print(data.dtypes)

# Convert problematic columns
data['col'] = pd.to_numeric(data['col'], errors='coerce')

# Or specify as discrete
generator.fit(data, discrete_columns=['col'])
```

## Getting Help

### Debug logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for Genesis only
logging.getLogger('genesis').setLevel(logging.DEBUG)
```

### Reporting issues

When reporting issues, include:

1. Genesis version: `genesis --version`
2. Python version: `python --version`
3. OS and version
4. Minimal reproducible example
5. Full error traceback

[Open an issue on GitHub](https://github.com/genesis/genesis/issues)

## Next Steps

- **[Examples](/docs/examples)** - Working code examples
- **[API Reference](/docs/api/reference)** - Complete API documentation
- **[Contributing](/docs/contributing)** - Help improve Genesis
