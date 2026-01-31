# Tabular Data Synthesis

Genesis provides three main methods for tabular data synthesis.

## Method Overview

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| **Gaussian Copula** | Fast | Good | Small/medium datasets, prototyping |
| **CTGAN** | Slow | Excellent | Complex distributions, large datasets |
| **TVAE** | Medium | Very Good | Balanced needs |

## Gaussian Copula

Statistical method that models marginal distributions and correlations.

```python
from genesis.generators.tabular import GaussianCopulaGenerator

generator = GaussianCopulaGenerator()
generator.fit(data, discrete_columns=['gender', 'city'])
synthetic = generator.generate(n_samples=1000)
```

### Pros
- Fast training
- No GPU required
- Works well with normally distributed data

### Cons
- May miss complex non-linear relationships
- Less effective with highly skewed data

## CTGAN

Deep learning method using Generative Adversarial Networks.

```python
from genesis.generators.tabular import CTGANGenerator

generator = CTGANGenerator(
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
)
generator.fit(data, discrete_columns=['gender', 'city'])
synthetic = generator.generate(n_samples=1000)
```

### Key Parameters
- `epochs`: Training iterations (default: 300)
- `batch_size`: Samples per batch (default: 500)
- `generator_dim`: Generator layer sizes
- `discriminator_dim`: Discriminator layer sizes

### Pros
- Excellent for complex data
- Handles multi-modal distributions
- Good for imbalanced classes

### Cons
- Slower training
- Requires parameter tuning
- Benefits from GPU

## TVAE

Variational Autoencoder approach.

```python
from genesis.generators.tabular import TVAEGenerator

generator = TVAEGenerator(
    epochs=300,
    batch_size=500,
    latent_dim=128,
)
generator.fit(data, discrete_columns=['gender', 'city'])
synthetic = generator.generate(n_samples=1000)
```

### Pros
- More stable training than GAN
- Good reconstruction quality
- Useful latent space

### Cons
- May produce blurry outputs
- Less sharp distributions

## Auto Selection

Let Genesis choose the best method:

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(method='auto')
generator.fit(data)
print(f"Selected: {generator.selected_method}")
```

Selection criteria:
- Dataset size
- Column types
- Distribution characteristics

## Working with Constraints

```python
from genesis import Constraint

constraints = [
    Constraint.positive('price'),
    Constraint.range('age', 0, 120),
    Constraint.unique('id'),
]

generator.fit(data, constraints=constraints)
```

## Handling Missing Values

Genesis handles missing values automatically:

```python
# Data with NaN values
data_with_missing = data.copy()
data_with_missing.loc[0:10, 'income'] = np.nan

generator.fit(data_with_missing, discrete_columns=['city'])
synthetic = generator.generate(1000)  # No missing values
```

## Conditional Generation

Generate data with specific conditions:

```python
# Generate only male records
synthetic_male = generator.sample_conditional(
    n_samples=100,
    conditions={'gender': 'Male'}
)
```
