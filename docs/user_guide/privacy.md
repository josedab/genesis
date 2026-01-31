# Privacy Configuration

Configure privacy settings for synthetic data generation.

## Privacy Levels

Use preset privacy levels:

```python
from genesis import PrivacyConfig

# Low privacy (maximum utility)
config = PrivacyConfig(privacy_level='low')

# Medium privacy (balanced)
config = PrivacyConfig(privacy_level='medium')

# High privacy (maximum protection)
config = PrivacyConfig(privacy_level='high')
```

## Differential Privacy

Add mathematical privacy guarantees:

```python
config = PrivacyConfig(
    enable_differential_privacy=True,
    epsilon=1.0,    # Privacy budget (lower = more private)
    delta=1e-5,     # Failure probability
)
```

### Understanding Epsilon
- `epsilon=10`: Weak privacy, high utility
- `epsilon=1`: Moderate privacy
- `epsilon=0.1`: Strong privacy, lower utility

## K-Anonymity

Ensure each record is indistinguishable from k-1 others:

```python
config = PrivacyConfig(
    k_anonymity=5,  # Min group size
    quasi_identifiers=['age', 'zipcode', 'gender'],
)
```

## L-Diversity

Ensure diversity in sensitive attributes:

```python
config = PrivacyConfig(
    l_diversity=3,  # Min distinct values
    sensitive_columns=['diagnosis'],
)
```

## Rare Category Suppression

Remove rare categories that could identify individuals:

```python
config = PrivacyConfig(
    suppress_rare_categories=True,
    rare_category_threshold=0.01,  # 1% threshold
)
```

## Complete Example

```python
from genesis import SyntheticGenerator, PrivacyConfig

privacy = PrivacyConfig(
    enable_differential_privacy=True,
    epsilon=0.5,
    k_anonymity=10,
    l_diversity=3,
    suppress_rare_categories=True,
    sensitive_columns=['diagnosis'],
    quasi_identifiers=['age', 'gender', 'zipcode'],
)

generator = SyntheticGenerator(method='ctgan', privacy=privacy)
generator.fit(data, discrete_columns=['gender', 'diagnosis'])
synthetic = generator.generate(n_samples=1000)
```

## Verifying Privacy

```python
from genesis.privacy.anonymity import check_k_anonymity, check_l_diversity

# Check k-anonymity
result = check_k_anonymity(synthetic, ['age', 'gender'], k=5)
print(f"Satisfies 5-anonymity: {result['satisfies_k']}")

# Check l-diversity
result = check_l_diversity(synthetic, ['age'], 'diagnosis', l=3)
print(f"Satisfies 3-diversity: {result['satisfies_l']}")
```
