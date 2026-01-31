# ADR 002: Sklearn-Style fit/generate API

## Status

Accepted

## Context

We need to define the primary API for synthetic data generation. Options considered:

1. **Functional API**: `generate_synthetic(data, method='ctgan')`
2. **Builder Pattern**: `Generator().with_method('ctgan').with_epochs(300).build().generate(data)`
3. **Sklearn-Style**: `generator.fit(data); generator.generate(n_samples)`

Key requirements:
- Intuitive for data scientists
- Supports configuration and customization
- Enables quality evaluation workflows
- Allows model persistence (save/load)

## Decision

Adopt the **sklearn-style fit/generate pattern**:

```python
from genesis import SyntheticGenerator

# Create generator with configuration
generator = SyntheticGenerator(method='ctgan', epochs=300)

# Fit to real data (learns distribution)
generator.fit(real_data, discrete_columns=['gender', 'city'])

# Generate synthetic samples
synthetic_data = generator.generate(n_samples=10000)

# Optional: evaluate quality
report = generator.quality_report()
```

### Key API Methods

| Method | Purpose |
|--------|---------|
| `fit(data, ...)` | Learn data distribution |
| `generate(n_samples)` | Create synthetic samples |
| `fit_generate(data, n_samples)` | Convenience method |
| `quality_report()` | Evaluate synthetic data |
| `save(path)` / `load(path)` | Model persistence |

## Consequences

### Positive

- Familiar to ML practitioners (matches sklearn, keras, transformers)
- Clear separation between training and generation
- Supports model persistence naturally
- Enables caching of trained models
- Easy to integrate into existing ML pipelines

### Negative

- Requires two-step process for simple cases
- Stateful object may be confusing for some users
- Need to handle "not fitted" errors

### Neutral

- `fit_generate()` convenience method addresses the two-step issue
- Pattern is well-established and documented
