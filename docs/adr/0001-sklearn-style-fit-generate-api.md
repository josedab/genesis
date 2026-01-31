# ADR-0001: Sklearn-Style Fit/Generate API Pattern

## Status

Accepted

## Context

When designing the Genesis API, we needed to choose an interface pattern that would be intuitive for our target users: data scientists and ML engineers. Several options were considered:

1. **Builder pattern**: `Generator().with_method('ctgan').with_privacy(epsilon=1.0).build().run(data)`
2. **Declarative/config-first**: `generate(config_file='config.yaml')`
3. **Sklearn-style fit/predict**: `generator.fit(data); synthetic = generator.generate(n)`
4. **Functional API**: `synthetic = generate(data, method='ctgan', n_samples=1000)`

Our target users work daily with scikit-learn, pandas, and similar libraries. They expect:
- Stateful objects that learn from data
- Separation between training and inference
- Method chaining support
- Familiar parameter names

## Decision

We adopt the **sklearn-style fit/generate pattern** as the primary API:

```python
from genesis import SyntheticGenerator

# Initialize with configuration
generator = SyntheticGenerator(method='ctgan', epochs=300)

# Fit to training data (learns distributions)
generator.fit(real_data, discrete_columns=['gender', 'city'])

# Generate synthetic samples (can be called multiple times)
synthetic_data = generator.generate(n_samples=10000)

# Access learned properties
schema = generator.schema
is_ready = generator.is_fitted
```

Key design choices:
- `fit()` returns `self` to enable method chaining
- `generate()` mirrors `predict()` but returns a DataFrame
- Generators are picklable for persistence
- A `fit_generate()` convenience method combines both steps
- `is_fitted` property for state introspection

## Consequences

### Positive

- **Zero learning curve** for sklearn users—the API feels immediately familiar
- **Clear separation of concerns**: training (expensive) vs. generation (cheap)
- **Reproducibility**: fitted generators can be saved/loaded for consistent generation
- **Flexibility**: same generator can produce multiple synthetic datasets
- **Composability**: generators work with sklearn pipelines and cross-validation

### Negative

- **Stateful complexity**: users must understand that generators have state
- **Two-step process**: simplest use case requires two calls, not one
- **Memory overhead**: fitted generators hold learned parameters in memory

### Neutral

- We provide `SyntheticGenerator.fit_generate()` as a one-liner for simple cases
- The functional API exists as `genesis.generate()` for quick scripts
- This pattern is consistent across all generator types (tabular, time series, text, image)

## Examples

```python
# Basic usage
gen = SyntheticGenerator(method='gaussian_copula')
gen.fit(df)
synthetic = gen.generate(5000)

# Method chaining
synthetic = (
    SyntheticGenerator(method='tvae')
    .fit(df, discrete_columns=['category'])
    .generate(1000)
)

# Persistence
gen.save('trained_model.pkl')
loaded = SyntheticGenerator.load('trained_model.pkl')
more_data = loaded.generate(500)

# Quality evaluation (extends the pattern)
report = gen.quality_report()
print(report.summary())
```
