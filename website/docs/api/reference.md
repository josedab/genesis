---
sidebar_position: 1
title: API Reference
---

# API Reference

Complete reference for Genesis Python API.

## Core Classes

### SyntheticGenerator

The main class for generating synthetic data.

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='ctgan',           # Generation method
    config=None,              # Method-specific configuration
    privacy=None,             # Privacy settings
    random_state=None         # Random seed for reproducibility
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `'ctgan'` | Generation method: `'ctgan'`, `'tvae'`, `'gaussian_copula'`, `'copulagan'` |
| `config` | dict | `None` | Method-specific configuration |
| `privacy` | dict | `None` | Privacy settings (differential privacy, k-anonymity) |
| `random_state` | int | `None` | Random seed for reproducibility |

#### Methods

##### fit()

Train the generator on data.

```python
generator.fit(
    data,                     # pandas DataFrame
    discrete_columns=None,    # List of categorical column names
    constraints=None          # List of Constraint objects
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame | Training data |
| `discrete_columns` | list | Categorical column names |
| `constraints` | list | Data constraints to enforce |

##### generate()

Generate synthetic data.

```python
synthetic = generator.generate(
    n_samples,                # Number of samples to generate
    conditions=None           # Optional conditions for conditional generation
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_samples` | int | Number of samples to generate |
| `conditions` | dict | Conditions for generation |

**Returns**: `pandas.DataFrame`

##### save() / load()

```python
# Save trained generator
generator.save('model.pkl')

# Load generator
generator = SyntheticGenerator.load('model.pkl')
```

### QualityEvaluator

Evaluate synthetic data quality.

```python
from genesis import QualityEvaluator

evaluator = QualityEvaluator(
    real_data,                # Original DataFrame
    synthetic_data,           # Generated DataFrame
    target_column=None        # Optional target for ML utility
)
```

#### Methods

##### evaluate()

```python
report = evaluator.evaluate()

print(report.overall_score)      # 0-1 overall quality
print(report.fidelity_score)     # Statistical fidelity
print(report.utility_score)      # ML utility (if target specified)
print(report.privacy_score)      # Privacy score
```

##### per_column_metrics()

```python
metrics = evaluator.per_column_metrics()
for col, m in metrics.items():
    print(f"{col}: {m['similarity']:.2f}")
```

### Constraint

Define data constraints.

```python
from genesis import Constraint

# Built-in constraints
Constraint.positive('column_name')
Constraint.range('column_name', min=0, max=100)
Constraint.unique('column_name')
Constraint.categorical('column_name', ['A', 'B', 'C'])
Constraint.regex('column_name', pattern)
Constraint.not_null('column_name')
```

---

## Convenience Functions

### auto_synthesize()

Automatically generate synthetic data with optimal settings.

```python
from genesis import auto_synthesize

synthetic = auto_synthesize(
    data,                     # Input DataFrame
    n_samples=None,           # Number of samples (default: same as input)
    discrete_columns=None,    # Categorical columns
    constraints=None,         # Data constraints
    privacy=None,             # Privacy settings
    mode='balanced',          # 'fast', 'balanced', or 'quality'
    return_report=False       # Return quality report
)
```

### augment_imbalanced()

Balance an imbalanced dataset.

```python
from genesis import augment_imbalanced

balanced = augment_imbalanced(
    data,                     # Input DataFrame
    target_column,            # Column to balance
    ratio=1.0,                # Target ratio (minority/majority)
    strategy='oversample',    # 'oversample', 'undersample', 'hybrid'
    discrete_columns=None,
    return_report=False
)
```

### run_privacy_audit()

Audit synthetic data privacy.

```python
from genesis import run_privacy_audit

report = run_privacy_audit(
    real_data,                # Original data
    synthetic_data,           # Generated data
    sensitive_columns=None,   # Columns with sensitive info
    quasi_identifiers=None    # Columns that could identify individuals
)

print(report.overall_score)
print(report.is_safe)
print(report.recommendations)
```

### detect_drift()

Detect statistical drift between datasets.

```python
from genesis import detect_drift

report = detect_drift(
    baseline,                 # Reference DataFrame
    current,                  # New DataFrame
    numeric_threshold=0.1,    # KS statistic threshold
    categorical_threshold=0.1 # JS divergence threshold
)

print(report.has_drift)
print(report.drift_score)
print(report.drifted_columns())
```

---

## Generators

### TimeSeriesGenerator

```python
from genesis import TimeSeriesGenerator

generator = TimeSeriesGenerator(config=None)
generator.fit(data, sequence_length=20)
sequences = generator.generate(n_sequences=10)
```

### TextGenerator

```python
from genesis import TextGenerator

generator = TextGenerator(config={'model': 'lstm', 'temperature': 0.8})
generator.fit(texts)  # List of strings
synthetic_texts = generator.generate(100)
```

### MultiTableGenerator

```python
from genesis import MultiTableGenerator

generator = MultiTableGenerator(table_config=None)
generator.fit(tables, relationships)
synthetic_tables = generator.generate(scale=1.0)
```

### ConditionalGenerator

```python
from genesis import ConditionalGenerator

generator = ConditionalGenerator(method='ctgan')
generator.fit(data, discrete_columns=['category'])
synthetic = generator.generate(
    n_samples=100,
    conditions={'category': 'A', 'value': ('>', 100)}
)
```

---

## AutoML

### AutoMLSynthesizer

```python
from genesis.automl import AutoMLSynthesizer

auto = AutoMLSynthesizer()

# Analyze data and get recommendation
recommendation = auto.analyze(data)
print(recommendation.method)
print(recommendation.confidence)
print(recommendation.reason)

# Fit and generate
auto.fit(data)
synthetic = auto.generate(n_samples)
```

---

## Privacy

### Differential Privacy

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {
            'epsilon': 1.0,    # Privacy budget (lower = more private)
            'delta': 1e-5      # Probability of privacy breach
        }
    }
)
```

### K-Anonymity

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'k_anonymity': {
            'k': 5,
            'quasi_identifiers': ['age', 'zip_code', 'gender']
        }
    }
)
```

---

## Versioning

### DatasetRepository

```python
from genesis.versioning import DatasetRepository

repo = DatasetRepository('./datasets')

# Save version
version_id = repo.save(df, message="Initial version", tags=['v1.0'])

# Load version
df = repo.load(version_id)
df = repo.load(tag='production')
df = repo.load_latest()

# List versions
versions = repo.list_versions()
versions = repo.list_versions(tag='production')

# Compare
diff = repo.compare(version1, version2)

# Tag
repo.tag(version_id, 'production')
```

---

## Pipeline

### Pipeline

```python
from genesis.pipeline import Pipeline, steps

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    steps.fit_generator('ctgan'),
    steps.generate(1000),
    steps.evaluate(),
    steps.save_csv('output.csv')
])

result = pipeline.run()
print(result.success)
print(result.metrics)
```

### From YAML

```python
pipeline = Pipeline.from_yaml('pipeline.yaml')
result = pipeline.run()
```

---

## Domain Generators

```python
from genesis.domains import (
    NameGenerator,
    EmailGenerator,
    PhoneGenerator,
    AddressGenerator,
    DateGenerator
)

# Generate domain-specific data
names = NameGenerator(locale='en_US').generate(100)
emails = EmailGenerator().generate(100)
phones = PhoneGenerator(locale='en_US').generate(100)
addresses = AddressGenerator(locale='en_US').generate(100)
dates = DateGenerator().generate(100, start_date='2020-01-01')
```

---

## Drift Detection

### DriftDetector

```python
from genesis.drift import DriftDetector

detector = DriftDetector(
    numeric_threshold=0.1,
    categorical_threshold=0.1
)

report = detector.detect(baseline, current)
print(report.has_drift)
print(report.drift_score)
print(report.column_metrics)
```

---

## Exceptions

```python
from genesis.exceptions import (
    GenesisError,           # Base exception
    FittingError,           # Error during fit()
    GenerationError,        # Error during generate()
    ValidationError,        # Constraint or schema validation failed
    PrivacyError,           # Privacy threshold not met
    ConfigurationError      # Invalid configuration
)
```

---

## Type Hints

Genesis provides full type hints:

```python
from genesis import SyntheticGenerator
from genesis.types import GeneratorConfig, PrivacyConfig

config: GeneratorConfig = {
    'epochs': 300,
    'batch_size': 500
}

privacy: PrivacyConfig = {
    'differential_privacy': {'epsilon': 1.0}
}

generator = SyntheticGenerator(method='ctgan', config=config, privacy=privacy)
```
