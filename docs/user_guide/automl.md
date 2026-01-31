# AutoML Synthetic Data Generation

Genesis provides an AutoML system that automatically selects the best generation method based on your data characteristics.

## Overview

The AutoML system analyzes your dataset's meta-features (size, column types, correlations, etc.) and recommends the optimal generation method without manual configuration.

```python
from genesis import auto_synthesize

# One-line automatic synthesis
synthetic_df = auto_synthesize(df, n_samples=1000)
```

## Components

| Component | Purpose |
|-----------|---------|
| **MetaFeatureExtractor** | Analyzes dataset characteristics |
| **MethodSelector** | Recommends generation methods |
| **AutoMLSynthesizer** | End-to-end automatic synthesis |

## Quick Start

### Basic Usage

```python
from genesis.automl import AutoMLSynthesizer, auto_synthesize

# Using the convenience function
synthetic = auto_synthesize(df, n_samples=1000)

# Using the class for more control
automl = AutoMLSynthesizer()
automl.fit(df)
synthetic = automl.generate(1000)

# Access the selected method
print(f"Selected: {automl.selected_method}")
print(f"Confidence: {automl.selection_confidence:.2%}")
```

### With Preferences

```python
from genesis.automl import AutoMLSynthesizer

# Prefer faster methods
automl = AutoMLSynthesizer(prefer_speed=True)
automl.fit(df)
synthetic = automl.generate(1000)

# Prefer higher quality
automl = AutoMLSynthesizer(prefer_quality=True)
automl.fit(df)
synthetic = automl.generate(1000)
```

## Meta-Feature Extraction

The system extracts these features from your data:

```python
from genesis.automl import MetaFeatureExtractor

extractor = MetaFeatureExtractor()
features = extractor.extract(df)

print(f"Rows: {features.n_rows}")
print(f"Columns: {features.n_columns}")
print(f"Numeric ratio: {features.numeric_ratio:.2%}")
print(f"Categorical ratio: {features.categorical_ratio:.2%}")
print(f"Missing ratio: {features.missing_ratio:.2%}")
print(f"High cardinality: {features.has_high_cardinality}")
print(f"Temporal columns: {features.has_temporal}")
print(f"Max correlation: {features.max_correlation:.3f}")
```

## Method Selection

```python
from genesis.automl import MethodSelector, MetaFeatureExtractor
from genesis import GenerationMethod

extractor = MetaFeatureExtractor()
features = extractor.extract(df)

selector = MethodSelector()
result = selector.select(features)

print(f"Recommended: {result.recommended_method}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Reason: {result.reason}")

# View all recommendations ranked
for rec in result.all_recommendations[:5]:
    print(f"  {rec.method}: {rec.confidence:.2%} - {rec.reason}")
```

## Method Selection Rules

| Data Characteristic | Recommended Method |
|--------------------|-------------------|
| Large dataset (>100K rows) | CTGAN |
| Small dataset (<1K rows) | Gaussian Copula |
| High cardinality categories | CTGAN |
| Strong correlations | TVAE |
| Many numeric columns | Gaussian Copula |
| Time series data | TimeGAN |
| Privacy required | DP-CTGAN |

## Customization

### Custom Method Constraints

```python
from genesis.automl import AutoMLSynthesizer
from genesis import GenerationMethod

# Exclude certain methods
automl = AutoMLSynthesizer(
    exclude_methods=[GenerationMethod.CTGAN]  # Don't use CTGAN
)

# Force a specific method (bypasses auto-selection)
automl = AutoMLSynthesizer(
    force_method=GenerationMethod.TVAE
)
```

### Training Configuration

```python
automl = AutoMLSynthesizer(
    max_epochs=500,
    batch_size=256,
    verbose=True
)
automl.fit(df)
```

## Pipeline Integration

```python
from genesis.automl import AutoMLSynthesizer
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("customers.csv")
    .add_node("automl", "synthesize", {
        "method": "auto",
        "n_samples": 10000,
        "prefer_quality": True
    })
    .sink("synthetic_customers.csv")
    .build()
)

pipeline.execute()
```

## Best Practices

1. **Trust the defaults**: The AutoML system makes good choices for most datasets
2. **Use preferences wisely**: Only set `prefer_speed=True` when generation time matters more than quality
3. **Check confidence**: Low confidence (<60%) suggests reviewing the recommendation
4. **Validate results**: Always evaluate synthetic data quality regardless of method

## CLI Usage

```bash
# Auto-select method and generate
genesis automl -i data.csv -o synthetic.csv -n 1000

# Prefer speed
genesis automl -i data.csv -o synthetic.csv --prefer-speed

# Just get recommendation without generating
genesis automl -i data.csv --recommend-only
```
