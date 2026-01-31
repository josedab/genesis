---
sidebar_position: 2
title: From SDV
---

# Migrating from SDV to Genesis

A complete guide to migrating your Synthetic Data Vault (SDV) code to Genesis.

## Overview

SDV and Genesis share similar concepts but have different APIs. This guide covers:
- API differences and equivalents
- Metadata handling
- Multi-table migration
- Feature parity

**Migration time:** 15-30 minutes for most projects.

---

## Installation

```bash
# Remove SDV (optional)
pip uninstall sdv

# Install Genesis
pip install genesis-synth[pytorch]
```

---

## Basic Migration

### Single Table Generation

```python
# ❌ SDV (before)
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Create and configure metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.update_column('category', sdtype='categorical')
metadata.update_column('amount', sdtype='numerical')

# Create synthesizer
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    verbose=True
)

# Train and generate
synthesizer.fit(data)
synthetic = synthesizer.sample(num_rows=1000)
```

```python
# ✅ Genesis (after)
from genesis import SyntheticGenerator

# Create generator (metadata auto-detected)
generator = SyntheticGenerator(
    method='ctgan',
    config={
        'epochs': 300,
        'batch_size': 500
    }
)

# Train and generate
generator.fit(data, discrete_columns=['category'])
synthetic = generator.generate(n_samples=1000)
```

**Key differences:**
- No separate metadata object needed
- Column types auto-detected (specify `discrete_columns` for categorical)
- `sample()` → `generate()`
- `num_rows` → `n_samples`

### Using AutoML (Recommended)

Genesis AutoML makes migration even simpler:

```python
# ✅ Genesis with AutoML - one line!
from genesis import auto_synthesize

synthetic = auto_synthesize(data, n_samples=1000)
```

---

## Metadata Migration

### SDV Metadata Files

If you have existing SDV metadata JSON files:

```python
# Load SDV metadata and convert
from genesis.compat import load_sdv_metadata

# Load your SDV metadata
metadata = load_sdv_metadata('metadata.json')

# Use with Genesis
generator = SyntheticGenerator(method='ctgan')
generator.fit(
    data,
    discrete_columns=metadata['categorical_columns'],
    constraints=metadata['constraints']
)
```

### Manual Metadata Conversion

| SDV Concept | Genesis Equivalent |
|-------------|-------------------|
| `sdtype='categorical'` | `discrete_columns=['col']` |
| `sdtype='numerical'` | Auto-detected |
| `sdtype='datetime'` | Auto-detected |
| `sdtype='boolean'` | `discrete_columns=['col']` |
| `sdtype='id'` | `Constraint.unique('col')` |

```python
# SDV metadata
metadata.update_column('status', sdtype='categorical')
metadata.update_column('customer_id', sdtype='id')
metadata.set_primary_key('customer_id')

# Genesis equivalent
generator.fit(
    data,
    discrete_columns=['status'],
    constraints=[Constraint.unique('customer_id')]
)
```

---

## Synthesizer Migration

### CTGAN

```python
# SDV
from sdv.single_table import CTGANSynthesizer
synth = CTGANSynthesizer(metadata, epochs=300)

# Genesis
from genesis import SyntheticGenerator
gen = SyntheticGenerator(method='ctgan', config={'epochs': 300})
```

### TVAE

```python
# SDV
from sdv.single_table import TVAESynthesizer
synth = TVAESynthesizer(metadata, epochs=300)

# Genesis
gen = SyntheticGenerator(method='tvae', config={'epochs': 300})
```

### Gaussian Copula

```python
# SDV
from sdv.single_table import GaussianCopulaSynthesizer
synth = GaussianCopulaSynthesizer(metadata)

# Genesis
gen = SyntheticGenerator(method='gaussian_copula')
```

### CopulaGAN

```python
# SDV
from sdv.single_table import CopulaGANSynthesizer
synth = CopulaGANSynthesizer(metadata)

# Genesis (use CTGAN with copula preprocessing)
gen = SyntheticGenerator(
    method='ctgan',
    config={'use_copula_transform': True}
)
```

---

## Multi-Table Migration

### SDV HMA (Hierarchical Modeling)

```python
# ❌ SDV (before)
from sdv.multi_table import HMASynthesizer
from sdv.metadata import MultiTableMetadata

metadata = MultiTableMetadata()
metadata.detect_from_dataframes(tables)
metadata.update_column('customers', 'customer_id', sdtype='id')
metadata.set_primary_key('customers', 'customer_id')
metadata.add_relationship(
    parent_table_name='customers',
    child_table_name='orders',
    parent_primary_key='customer_id',
    child_foreign_key='customer_id'
)

synth = HMASynthesizer(metadata)
synth.fit(tables)
synthetic_tables = synth.sample(scale=1.0)
```

```python
# ✅ Genesis (after)
from genesis import MultiTableGenerator

generator = MultiTableGenerator()
generator.fit(
    tables={
        'customers': customers_df,
        'orders': orders_df
    },
    relationships=[
        ('orders', 'customer_id', 'customers', 'customer_id')
    ]
)
synthetic_tables = generator.generate(scale=1.0)
```

**Key differences:**
- Relationships defined as tuples: `(child_table, child_col, parent_table, parent_col)`
- No separate metadata object
- Primary keys auto-detected

---

## Constraints Migration

### SDV Constraints

```python
# SDV
from sdv.constraints import Positive, Between, Unique

synth.add_constraints([
    Positive(column_name='amount'),
    Between(column_name='age', low=0, high=120),
    Unique(column_names=['customer_id'])
])
```

```python
# Genesis
from genesis import Constraint

generator.fit(
    data,
    constraints=[
        Constraint.positive('amount'),
        Constraint.range('age', 0, 120),
        Constraint.unique('customer_id')
    ]
)
```

### Constraint Mapping

| SDV Constraint | Genesis Constraint |
|----------------|-------------------|
| `Positive` | `Constraint.positive(col)` |
| `Negative` | `Constraint.negative(col)` |
| `Between` | `Constraint.range(col, min, max)` |
| `Unique` | `Constraint.unique(col)` |
| `FixedCombinations` | `Constraint.combinations([cols])` |
| `Inequality` | `Constraint.greater_than(col1, col2)` |
| `OneHotEncoding` | `Constraint.one_hot([cols])` |

---

## Evaluation Migration

### SDV Evaluation

```python
# SDV
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

quality_report = evaluate_quality(real, synthetic, metadata)
diagnostic = run_diagnostic(real, synthetic, metadata)
```

```python
# Genesis
from genesis import QualityEvaluator

evaluator = QualityEvaluator(real, synthetic)
report = evaluator.evaluate()

print(report.overall_score)      # Overall quality
print(report.fidelity_score)     # Column shapes + correlations
print(report.utility_score)      # ML utility
print(report.privacy_score)      # Privacy metrics

# Export report
report.to_html('quality_report.html')
```

---

## Saving and Loading Models

### SDV

```python
# SDV
synth.save('model.pkl')
synth = CTGANSynthesizer.load('model.pkl')
```

### Genesis

```python
# Genesis
generator.save('model.pkl')
generator = SyntheticGenerator.load('model.pkl')
```

---

## Conditional Sampling

### SDV

```python
# SDV
from sdv.sampling import Condition

conditions = [
    Condition({'status': 'active'}, num_rows=500),
    Condition({'status': 'inactive'}, num_rows=500)
]
synthetic = synth.sample_from_conditions(conditions)
```

### Genesis

```python
# Genesis
from genesis import ConditionalGenerator

gen = ConditionalGenerator(method='ctgan')
gen.fit(data, discrete_columns=['status'])

# Generate with conditions
active = gen.generate(500, conditions={'status': 'active'})
inactive = gen.generate(500, conditions={'status': 'inactive'})
synthetic = pd.concat([active, inactive])
```

---

## Features Only in Genesis

After migrating, you gain access to Genesis-exclusive features:

### AutoML

```python
from genesis import auto_synthesize

# Automatically picks best method
synthetic = auto_synthesize(data, n_samples=1000, mode='quality')
```

### Privacy Attack Testing

```python
from genesis import run_privacy_audit

report = run_privacy_audit(real, synthetic)
print(f"Safe to release: {report.is_safe}")
```

### Dataset Versioning

```python
from genesis.versioning import DatasetRepository

repo = DatasetRepository('./data_versions')
repo.save(synthetic, message="Production v1", tags=['production'])
```

### Drift Detection

```python
from genesis import detect_drift

report = detect_drift(baseline_data, new_data)
if report.has_drift:
    print(f"Drift detected: {report.drifted_columns}")
```

### Pipeline API

```python
from genesis.pipeline import Pipeline, steps

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    steps.fit_generator('ctgan'),
    steps.generate(1000),
    steps.evaluate(),
    steps.save_csv('output.csv')
])
pipeline.run()
```

---

## Common Issues

### "Missing metadata" errors

Genesis auto-detects column types. Just specify categorical columns:

```python
generator.fit(data, discrete_columns=['cat1', 'cat2', 'cat3'])
```

### Different output columns

SDV may rename columns. Genesis preserves original names:

```python
# Verify columns match
assert set(synthetic.columns) == set(real.columns)
```

### Quality differences

Different random seeds produce different results. For reproducibility:

```python
generator = SyntheticGenerator(method='ctgan', random_state=42)
```

---

## Migration Checklist

- [ ] Install Genesis: `pip install genesis-synth[pytorch]`
- [ ] Update imports
- [ ] Replace `SingleTableMetadata` with `discrete_columns` parameter
- [ ] Replace `sample()` with `generate()`
- [ ] Update constraints syntax
- [ ] Update evaluation code
- [ ] Test quality on sample data
- [ ] Update CI/CD pipelines

---

## Need Help?

- [GitHub Discussions](https://github.com/genesis-synth/genesis/discussions) - Ask migration questions
- [API Reference](/docs/api/reference) - Full API documentation
- [Examples](/docs/examples) - Working code examples
