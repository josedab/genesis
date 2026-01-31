# API Reference

Complete API documentation for Genesis.

## Core Classes

### SyntheticGenerator

Main entry point for synthetic data generation.

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='auto',           # 'auto', 'ctgan', 'tvae', 'gaussian_copula'
    config=GeneratorConfig(),
    privacy=PrivacyConfig(),
)
```

**Methods:**
- `fit(data, discrete_columns=None, constraints=None)` - Fit on real data
- `generate(n_samples)` - Generate synthetic samples
- `quality_report()` - Get quality report
- `save(path)` - Save model
- `load(path)` - Load model

### GeneratorConfig

Configuration for generators.

```python
from genesis import GeneratorConfig

config = GeneratorConfig(
    epochs=300,              # Training epochs
    batch_size=500,          # Batch size
    learning_rate=0.0002,    # Learning rate
    random_seed=42,          # Random seed
    verbose=True,            # Print progress
)
```

### PrivacyConfig

Privacy settings.

```python
from genesis import PrivacyConfig

config = PrivacyConfig(
    privacy_level='medium',                # 'low', 'medium', 'high'
    enable_differential_privacy=False,     # DP-SGD
    epsilon=1.0,                           # Privacy budget
    delta=1e-5,                            # DP delta
    k_anonymity=None,                      # K-anonymity requirement
    l_diversity=None,                      # L-diversity requirement
    suppress_rare_categories=False,        # Remove rare categories
    rare_category_threshold=0.01,          # Threshold for rare
)
```

### Constraint

Data constraints.

```python
from genesis import Constraint

# Factory methods
Constraint.positive(column)              # Value > 0
Constraint.range(column, min, max)       # min <= value <= max
Constraint.unique(column)                # All unique values
Constraint.one_of(column, values)        # Value in list
Constraint.regex(column, pattern)        # Matches pattern
```

## Generators

### CTGANGenerator

```python
from genesis.generators.tabular import CTGANGenerator

generator = CTGANGenerator(
    embedding_dim=128,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    epochs=300,
    batch_size=500,
    pac=10,  # PacGAN parameter
)
```

### TVAEGenerator

```python
from genesis.generators.tabular import TVAEGenerator

generator = TVAEGenerator(
    embedding_dim=128,
    encoder_dim=(128, 128),
    decoder_dim=(128, 128),
    latent_dim=128,
    epochs=300,
)
```

### GaussianCopulaGenerator

```python
from genesis.generators.tabular import GaussianCopulaGenerator

generator = GaussianCopulaGenerator(
    default_distribution='norm',  # 'norm', 'beta', 'gamma'
)
```

### TimeGANGenerator

```python
from genesis.generators.timeseries import TimeGANGenerator

generator = TimeGANGenerator(
    seq_len=24,
    hidden_dim=24,
    n_layers=3,
    n_epochs=100,
)
```

### LLMTextGenerator

```python
from genesis.generators.text import LLMTextGenerator

generator = LLMTextGenerator(
    backend='openai',           # 'openai' or 'huggingface'
    model='gpt-3.5-turbo',
    temperature=0.7,
    max_tokens=500,
    privacy_filter=True,
)
```

## Evaluation

### QualityEvaluator

```python
from genesis import QualityEvaluator

evaluator = QualityEvaluator(real_data, synthetic_data)
report = evaluator.evaluate(target_column=None)
```

### QualityReport

```python
report.overall_score      # 0-100 overall score
report.fidelity_score     # 0-1 statistical fidelity
report.utility_score      # 0-1 ML utility
report.privacy_score      # 0-1 privacy score

report.summary()          # Text summary
report.to_dict()          # Dictionary
report.to_json()          # JSON string
report.to_html()          # HTML report
report.save_html(path)    # Save HTML
report.save_json(path)    # Save JSON
```

## Multi-Table

### MultiTableGenerator

```python
from genesis.multitable import MultiTableGenerator, RelationalSchema

schema = RelationalSchema.from_dataframes(
    tables,
    foreign_keys=[...],
    primary_keys={...},
)

generator = MultiTableGenerator()
generator.fit_tables(tables, schema)
synthetic_tables = generator.generate_tables(n_samples={...})
```

## CLI Commands

```bash
genesis generate --help
genesis evaluate --help
genesis analyze --help
genesis report --help
```
