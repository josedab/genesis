---
sidebar_position: 2
title: Configuration
---

# Configuration Reference

Complete reference for all Genesis configuration options.

## Generator Configuration

### CTGAN

Conditional Tabular GAN - best for complex distributions.

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='ctgan',
    config={
        'epochs': 300,              # Training epochs
        'batch_size': 500,          # Batch size
        'generator_dim': (256, 256),  # Generator architecture
        'discriminator_dim': (256, 256),  # Discriminator architecture
        'generator_lr': 2e-4,       # Generator learning rate
        'discriminator_lr': 2e-4,   # Discriminator learning rate
        'discriminator_steps': 1,   # D steps per G step
        'log_frequency': True,      # Log training progress
        'pac': 10                   # PAC size for mode coverage
    }
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 300 | Number of training epochs |
| `batch_size` | int | 500 | Training batch size |
| `generator_dim` | tuple | (256, 256) | Generator hidden layer sizes |
| `discriminator_dim` | tuple | (256, 256) | Discriminator hidden layer sizes |
| `generator_lr` | float | 2e-4 | Generator learning rate |
| `discriminator_lr` | float | 2e-4 | Discriminator learning rate |
| `discriminator_steps` | int | 1 | Discriminator updates per generator update |
| `log_frequency` | bool | True | Whether to log training progress |
| `pac` | int | 10 | PAC size for mode coverage |

### TVAE

Tabular Variational Autoencoder - good for high-dimensional data.

```python
generator = SyntheticGenerator(
    method='tvae',
    config={
        'epochs': 300,
        'batch_size': 500,
        'compress_dims': (128, 128),
        'decompress_dims': (128, 128),
        'embedding_dim': 128,
        'l2scale': 1e-5,
        'loss_factor': 2
    }
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 300 | Training epochs |
| `batch_size` | int | 500 | Training batch size |
| `compress_dims` | tuple | (128, 128) | Encoder hidden layers |
| `decompress_dims` | tuple | (128, 128) | Decoder hidden layers |
| `embedding_dim` | int | 128 | Latent space dimension |
| `l2scale` | float | 1e-5 | L2 regularization |
| `loss_factor` | int | 2 | Loss scaling factor |

### Gaussian Copula

Statistical method - fast and interpretable.

```python
generator = SyntheticGenerator(
    method='gaussian_copula',
    config={
        'default_distribution': 'parametric',
        'categorical_transformer': 'label_encoding'
    }
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_distribution` | str | 'parametric' | 'parametric' or 'empirical' |
| `categorical_transformer` | str | 'label_encoding' | Categorical encoding method |

### CopulaGAN

Combines copulas with GANs.

```python
generator = SyntheticGenerator(
    method='copulagan',
    config={
        'epochs': 300,
        'batch_size': 500,
        'generator_dim': (256, 256),
        'discriminator_dim': (256, 256)
    }
)
```

---

## Privacy Configuration

### Differential Privacy

```python
privacy = {
    'differential_privacy': {
        'epsilon': 1.0,      # Privacy budget
        'delta': 1e-5,       # Breach probability
        'max_grad_norm': 1.0,  # Gradient clipping
        'noise_multiplier': None  # Auto-calculated if None
    }
}

generator = SyntheticGenerator(method='ctgan', privacy=privacy)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon` | float | 1.0 | Privacy budget (lower = more private) |
| `delta` | float | 1e-5 | Probability of privacy breach |
| `max_grad_norm` | float | 1.0 | Gradient clipping threshold |
| `noise_multiplier` | float | None | Noise scale (auto-calculated if None) |

**Epsilon Guidelines:**
- `ε < 1`: Strong privacy, significant utility loss
- `ε = 1-3`: Good balance of privacy/utility
- `ε > 3`: Weak privacy, high utility

### K-Anonymity

```python
privacy = {
    'k_anonymity': {
        'k': 5,
        'quasi_identifiers': ['age', 'zip_code', 'gender'],
        'sensitive_columns': ['income', 'health_status'],
        'suppression_limit': 0.1
    }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 5 | Minimum group size |
| `quasi_identifiers` | list | None | Columns for grouping |
| `sensitive_columns` | list | None | Sensitive attributes |
| `suppression_limit` | float | 0.1 | Max fraction to suppress |

### Combined Privacy

```python
privacy = {
    'differential_privacy': {'epsilon': 1.0},
    'k_anonymity': {'k': 5, 'quasi_identifiers': ['age', 'zip']},
    'suppress_outliers': True,
    'min_category_count': 10
}
```

---

## Time Series Configuration

```python
from genesis import TimeSeriesGenerator

generator = TimeSeriesGenerator(
    config={
        'hidden_dim': 64,
        'n_layers': 2,
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'model_type': 'lstm'  # 'lstm', 'gru', 'transformer'
    }
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_dim` | int | 64 | Hidden layer dimension |
| `n_layers` | int | 2 | Number of recurrent layers |
| `epochs` | int | 200 | Training epochs |
| `batch_size` | int | 128 | Training batch size |
| `learning_rate` | float | 1e-3 | Learning rate |
| `model_type` | str | 'lstm' | RNN type: 'lstm', 'gru', 'transformer' |

---

## Text Configuration

```python
from genesis import TextGenerator

generator = TextGenerator(
    config={
        'model': 'lstm',          # 'lstm', 'gru', 'markov', 'gpt2'
        'max_length': 200,        # Maximum generation length
        'min_length': 10,         # Minimum generation length
        'temperature': 0.8,       # Sampling temperature
        'hidden_dim': 256,        # LSTM hidden dimension
        'embedding_dim': 128,     # Word embedding dimension
        'epochs': 100             # Training epochs
    }
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | 'lstm' | Model type |
| `max_length` | int | 200 | Max tokens to generate |
| `min_length` | int | 10 | Min tokens to generate |
| `temperature` | float | 0.8 | Sampling randomness (0.1-1.5) |
| `hidden_dim` | int | 256 | LSTM hidden size |
| `embedding_dim` | int | 128 | Word embedding size |
| `epochs` | int | 100 | Training epochs |

---

## AutoML Configuration

```python
from genesis import auto_synthesize

synthetic = auto_synthesize(
    data,
    n_samples=1000,
    mode='balanced',          # 'fast', 'balanced', 'quality'
    tune_hyperparameters=False,
    tuning_budget=60,         # Seconds for tuning
    methods=['ctgan', 'tvae', 'gaussian_copula']  # Methods to consider
)
```

| Mode | Description | Time |
|------|-------------|------|
| `fast` | Quick generation, lower quality | ~1 min |
| `balanced` | Good quality/speed tradeoff | ~5 min |
| `quality` | Maximum quality | ~15+ min |

---

## Pipeline Configuration

### YAML Configuration

```yaml
name: my_pipeline
description: Production data generation

settings:
  log_level: info
  on_error: stop
  retry_failed: 3

steps:
  - load_csv:
      path: data.csv
  
  - fit_generator:
      method: ctgan
      discrete_columns:
        - category
        - status
      config:
        epochs: 300
        batch_size: 500
  
  - generate:
      n_samples: 10000
  
  - evaluate:
      target_column: churn
  
  - privacy_audit:
      sensitive_columns:
        - income
      threshold: 0.9
  
  - save_csv:
      path: output.csv
```

---

## Drift Detection Configuration

```python
from genesis.drift import DriftDetector

detector = DriftDetector(
    numeric_threshold=0.1,        # KS statistic threshold
    categorical_threshold=0.1,    # JS divergence threshold
    pvalue_threshold=0.05,        # Statistical significance
    columns=None,                 # Columns to monitor (None = all)
    ignore_columns=['id']         # Columns to ignore
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_threshold` | float | 0.1 | KS statistic threshold |
| `categorical_threshold` | float | 0.1 | JS divergence threshold |
| `pvalue_threshold` | float | 0.05 | Significance level |
| `columns` | list | None | Columns to monitor |
| `ignore_columns` | list | None | Columns to skip |

---

## Domain Generator Configuration

```python
from genesis.domains import NameGenerator

gen = NameGenerator(
    locale='en_US',
    seed=42
)

# Available locales
locales = [
    'en_US', 'en_GB', 'en_CA', 'en_AU',  # English
    'de_DE', 'de_AT', 'de_CH',            # German
    'fr_FR', 'fr_CA',                      # French
    'es_ES', 'es_MX',                      # Spanish
    'it_IT', 'pt_BR', 'nl_NL',            # Other European
    'ja_JP', 'zh_CN', 'ko_KR',            # Asian
    # ... 50+ locales supported
]
```

---

## Environment Variables

```bash
# Set default random seed
export GENESIS_RANDOM_SEED=42

# Configure logging
export GENESIS_LOG_LEVEL=INFO

# GPU settings
export GENESIS_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

# Cache directory
export GENESIS_CACHE_DIR=/path/to/cache

# Privacy defaults
export GENESIS_DEFAULT_EPSILON=1.0
export GENESIS_DEFAULT_K=5
```

---

## Configuration Files

### genesis.yaml

```yaml
# Default configuration file
defaults:
  method: ctgan
  epochs: 300
  batch_size: 500
  random_seed: 42

privacy:
  differential_privacy:
    epsilon: 1.0
  k_anonymity:
    k: 5

logging:
  level: INFO
  file: genesis.log

paths:
  cache: ~/.genesis/cache
  models: ~/.genesis/models
```

Load configuration:

```python
from genesis import SyntheticGenerator

# Auto-loads from ./genesis.yaml or ~/.genesis/genesis.yaml
generator = SyntheticGenerator()

# Or specify path
generator = SyntheticGenerator.from_config('path/to/genesis.yaml')
```
