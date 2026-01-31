---
sidebar_position: 5
title: v1.4.0 Features
---

# v1.4.0 Features API Reference

API reference for features introduced in Genesis v1.4.0.

## AutoML Synthesis

Automatic method selection and hyperparameter optimization.

### AutoMLSynthesizer

```python
from genesis.automl import AutoMLSynthesizer

auto = AutoMLSynthesizer(
    methods=['ctgan', 'tvae', 'gaussian_copula'],
    scoring='quality',         # 'quality', 'privacy', 'balanced'
    time_budget=300            # seconds
)

# Analyze and recommend
recommendation = auto.analyze(data)
print(recommendation.method)       # 'ctgan'
print(recommendation.confidence)   # 0.87
print(recommendation.reason)       # 'Complex distributions...'

# Fit with recommended method
auto.fit(data, discrete_columns=['status'])

# Generate
synthetic = auto.generate(n_samples=1000)
```

### auto_synthesize()

Convenience function for one-shot generation.

```python
from genesis import auto_synthesize

synthetic = auto_synthesize(
    data,
    n_samples=1000,
    mode='balanced',              # 'fast', 'balanced', 'quality'
    discrete_columns=['status'],
    constraints=[...],
    privacy={'epsilon': 1.0}
)

# With report
synthetic, report = auto_synthesize(data, return_report=True)
print(report.method)
print(report.quality_score)
```

---

## Data Augmentation

Balance imbalanced datasets.

### augment_imbalanced()

```python
from genesis import augment_imbalanced

balanced = augment_imbalanced(
    data,
    target_column='fraud',
    ratio=1.0,                    # Target ratio minority/majority
    strategy='oversample',        # 'oversample', 'undersample', 'hybrid'
    discrete_columns=['category'],
    quality_threshold=0.8
)

# With report
balanced, report = augment_imbalanced(
    data,
    target_column='fraud',
    return_report=True
)

print(report.n_generated)
print(report.quality_score)
```

### AugmentationPipeline

```python
from genesis.augmentation import AugmentationPipeline

pipeline = AugmentationPipeline(
    target_column='fraud',
    strategy='oversample',
    generator_config={'method': 'ctgan', 'epochs': 300}
)

pipeline.fit(data)
balanced = pipeline.transform(data)

# Metrics
print(pipeline.metrics)
```

---

## Privacy Attack Testing

Evaluate synthetic data privacy.

### run_privacy_audit()

```python
from genesis import run_privacy_audit

report = run_privacy_audit(
    real_data,
    synthetic_data,
    sensitive_columns=['income', 'health'],
    quasi_identifiers=['age', 'gender', 'zip']
)

print(report.overall_score)       # 0.0-1.0
print(report.is_safe)             # True/False
print(report.recommendations)     # List of suggestions
```

### Individual Attacks

```python
from genesis.privacy_attacks import (
    MembershipInferenceAttack,
    AttributeInferenceAttack,
    SinglingOutAttack,
    LinkageAttack
)

# Membership inference
mia = MembershipInferenceAttack()
result = mia.evaluate(real_data, synthetic_data)
print(result.accuracy)            # Attack success rate
print(result.risk_level)          # 'low', 'medium', 'high', 'critical'

# Attribute inference
aia = AttributeInferenceAttack(
    sensitive_column='income',
    known_columns=['age', 'education']
)
result = aia.evaluate(real_data, synthetic_data)

# Singling out
soa = SinglingOutAttack(
    quasi_identifiers=['zip', 'age', 'gender']
)
result = soa.evaluate(real_data, synthetic_data)
print(result.risk)                # Fraction identifiable

# Linkage attack
la = LinkageAttack(
    linking_columns=['name', 'dob', 'zip']
)
result = la.evaluate(real_data, synthetic_data)
```

---

## LLM-Powered Inference

Generate context-aware synthetic records.

### LLMInference

```python
from genesis.llm_inference import LLMInference

llm = LLMInference(
    model='gpt-3.5-turbo',
    api_key='your-api-key'
)

# Generate with context
synthetic = llm.generate(
    n_samples=100,
    schema={'name': 'str', 'bio': 'str', 'skills': 'list'},
    context="Generate profiles for software engineers at a startup"
)

# Enhance existing data
enhanced = llm.enhance(
    data,
    columns_to_generate=['description', 'summary'],
    context="E-commerce product data"
)
```

### Local LLMs

```python
llm = LLMInference(
    model='llama2',
    backend='ollama',
    base_url='http://localhost:11434'
)
```

---

## Drift Detection

Detect statistical changes between datasets.

### detect_drift()

```python
from genesis import detect_drift

report = detect_drift(
    baseline,                     # Reference DataFrame
    current,                      # New DataFrame
    numeric_threshold=0.1,        # KS statistic threshold
    categorical_threshold=0.1    # JS divergence threshold
)

print(report.has_drift)           # True/False
print(report.drift_score)         # 0.0-1.0
print(report.drifted_columns())   # List of columns
```

### DriftDetector

```python
from genesis.drift import DriftDetector

detector = DriftDetector(
    numeric_threshold=0.1,
    categorical_threshold=0.1,
    pvalue_threshold=0.05
)

report = detector.detect(baseline, current)

# Per-column metrics
for col, metrics in report.column_metrics.items():
    print(f"{col}:")
    print(f"  Score: {metrics['score']:.3f}")
    print(f"  Drift: {metrics['has_drift']}")
```

### DriftMonitor

Continuous monitoring.

```python
from genesis.drift import DriftMonitor

monitor = DriftMonitor(
    baseline=training_data,
    threshold=0.15
)

# Check batches
for batch in data_stream:
    report = monitor.check(batch)
    if report.alert:
        notify(f"Drift detected: {report.summary()}")
```

---

## Dataset Versioning

Track and manage dataset versions.

### DatasetRepository

```python
from genesis.versioning import DatasetRepository

repo = DatasetRepository('./datasets')

# Save version
version_id = repo.save(
    df,
    message="Initial synthetic data",
    tags=['v1.0', 'production'],
    metadata={'quality': 0.95}
)

# Load versions
df = repo.load(version_id)
df = repo.load(tag='production')
df = repo.load_latest()

# List and compare
versions = repo.list_versions(tag='production')
diff = repo.compare(v1_id, v2_id)

# Tagging
repo.tag(version_id, 'production')
repo.untag(version_id, 'experimental')
```

---

## GPU Acceleration

Optimized GPU training and generation.

### BatchedGenerator

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(
    method='ctgan',
    device='cuda',
    batch_size=10000,
    mixed_precision=True
)

generator.fit(large_data)
synthetic = generator.generate(1_000_000)
```

### Multi-GPU Training

```python
from genesis.gpu import DistributedGenerator

generator = DistributedGenerator(
    method='ctgan',
    devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
    strategy='data_parallel'
)
```

### Memory Optimization

```python
from genesis.gpu import optimize_memory

# Automatic memory optimization
with optimize_memory():
    generator.fit(very_large_data)
    synthetic = generator.generate(n_samples)
```

---

## Domain Generators

Pre-built generators for common data types.

### Available Generators

```python
from genesis.domains import (
    NameGenerator,
    EmailGenerator,
    PhoneGenerator,
    AddressGenerator,
    DateGenerator,
    SSNGenerator,
    CreditCardGenerator,
    CompanyGenerator
)

# Names
names = NameGenerator(locale='en_US').generate(1000)

# Emails (linked to names)
emails = EmailGenerator().generate_from_names(names)

# Addresses
addresses = AddressGenerator(locale='en_US').generate(1000, format='dict')

# Phones
phones = PhoneGenerator(locale='en_US').generate(1000, format='e164')

# Dates
dates = DateGenerator().generate(1000, start_date='2020-01-01')
```

### CompositeGenerator

```python
from genesis.domains import CompositeGenerator

gen = CompositeGenerator({
    'name': NameGenerator(locale='en_US'),
    'email': EmailGenerator(link_to='name'),
    'phone': PhoneGenerator(locale='en_US'),
    'address': AddressGenerator(locale='en_US')
})

records = gen.generate(10000)  # DataFrame
```

---

## Pipeline Builder

Create reproducible generation workflows.

### Pipeline

```python
from genesis.pipeline import Pipeline, steps

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    steps.fit_generator('ctgan', discrete_columns=['status']),
    steps.generate(1000),
    steps.evaluate(),
    steps.privacy_audit(threshold=0.9),
    steps.save_csv('output.csv')
])

result = pipeline.run()
print(result.success)
print(result.metrics)
```

### YAML Configuration

```yaml
name: production_pipeline
steps:
  - load_csv: data.csv
  - fit_generator:
      method: ctgan
      epochs: 300
  - generate:
      n_samples: 10000
  - evaluate:
      target_column: churn
  - save_csv: output.csv
```

```python
pipeline = Pipeline.from_yaml('pipeline.yaml')
result = pipeline.run()
```

### Custom Steps

```python
from genesis.pipeline import Step

class MyStep(Step):
    def run(self, context):
        context.data = my_transform(context.data)
        return context

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    MyStep(),
    steps.save_csv('output.csv')
])
```

---

## v1.4.0 Changelog

### New Features
- **AutoML synthesis** with automatic method selection
- **Data augmentation** for imbalanced datasets
- **Privacy attack testing** (MIA, attribute inference, singling out)
- **LLM-powered inference** for context-aware generation
- **Drift detection** with continuous monitoring
- **Dataset versioning** with content-addressable storage
- **Domain generators** for names, emails, addresses
- **Pipeline builder** with YAML support

### Improvements
- 2x faster generation on GPU
- Better memory efficiency for large datasets
- Improved constraint handling
- Enhanced evaluation metrics

### API Changes
- New `auto_synthesize()` convenience function
- New `augment_imbalanced()` function
- New `run_privacy_audit()` function
- New `detect_drift()` function
- Pipeline API for workflows

### Bug Fixes
- Fixed k-anonymity with single quasi-identifier
- Fixed scipy boolean array handling
- Fixed numpy bool assertion compatibility
- Improved categorical handling in evaluation

See [CHANGELOG.md](https://github.com/genesis/genesis/blob/main/CHANGELOG.md) for complete details.
