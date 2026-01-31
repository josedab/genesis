---
sidebar_position: 7
title: Data Augmentation
---

# Data Augmentation

Balance imbalanced datasets by generating synthetic samples for minority classes.

## Quick Start

```python
from genesis import augment_imbalanced
import pandas as pd

# Load imbalanced data (e.g., 95% class 0, 5% class 1)
df = pd.read_csv('fraud_detection.csv')

# Balance the dataset
balanced = augment_imbalanced(df, target_column='is_fraud')
```

## The Problem

Many real-world datasets are imbalanced:

```python
# Check class distribution
print(df['is_fraud'].value_counts(normalize=True))
# is_fraud
# 0    0.95
# 1    0.05
```

This causes:
- Biased models favoring majority class
- Poor minority class detection
- Misleading accuracy metrics

## Augmentation Strategies

### Oversample Minority (Default)

Generate more minority samples:

```python
balanced = augment_imbalanced(
    df,
    target_column='is_fraud',
    strategy='oversample'  # Default
)

print(balanced['is_fraud'].value_counts())
# is_fraud
# 0    9500  (original)
# 1    9500  (augmented from 500)
```

### Undersample Majority

Reduce majority class (loses information):

```python
balanced = augment_imbalanced(
    df,
    target_column='is_fraud',
    strategy='undersample'
)

print(balanced['is_fraud'].value_counts())
# is_fraud
# 0    500  (reduced from 9500)
# 1    500  (original)
```

### Hybrid (SMOTE-like)

Combination approach:

```python
balanced = augment_imbalanced(
    df,
    target_column='is_fraud',
    strategy='hybrid',
    sampling_ratio=0.5  # Minority at 50% of majority
)
```

## Target Ratios

Control the final class distribution:

```python
# Equal classes (1:1)
balanced = augment_imbalanced(df, 'target', ratio=1.0)

# Minority at 30% of majority
balanced = augment_imbalanced(df, 'target', ratio=0.3)

# Specific counts
balanced = augment_imbalanced(
    df, 'target',
    target_counts={0: 5000, 1: 5000}
)
```

## Multi-Class Balancing

```python
# Original: class A: 1000, B: 500, C: 100
balanced = augment_imbalanced(
    df,
    target_column='category',
    ratio=1.0  # All classes equal
)

# Result: class A: 1000, B: 1000, C: 1000
```

### Specific Ratios Per Class

```python
balanced = augment_imbalanced(
    df,
    target_column='category',
    target_counts={'A': 1000, 'B': 800, 'C': 500}
)
```

## Conditional Generation

Augmentation uses conditional generation under the hood:

```python
from genesis import ConditionalGenerator

# More control with conditional generator
generator = ConditionalGenerator(method='ctgan')
generator.fit(df, discrete_columns=['is_fraud'])

# Generate only fraud cases
fraud_samples = generator.generate(
    n_samples=9000,
    conditions={'is_fraud': 1}
)

# Combine
balanced = pd.concat([df, fraud_samples])
```

## Quality-Aware Augmentation

Ensure synthetic samples are high quality:

```python
balanced, report = augment_imbalanced(
    df,
    target_column='is_fraud',
    return_report=True,
    quality_threshold=0.8  # Reject low-quality samples
)

print(f"Augmentation quality: {report.quality_score:.1%}")
print(f"Samples generated: {report.n_generated}")
print(f"Samples accepted: {report.n_accepted}")
```

## Preserving Correlations

Synthetic samples maintain feature relationships:

```python
import matplotlib.pyplot as plt

# Original fraud correlation
real_fraud = df[df['is_fraud'] == 1]
real_corr = real_fraud.corr()

# Synthetic fraud correlation
syn_fraud = balanced[balanced['_synthetic'] == True]
syn_corr = syn_fraud.corr()

# Compare
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(real_corr)
axes[0].set_title('Real Fraud Correlations')
axes[1].imshow(syn_corr)
axes[1].set_title('Synthetic Fraud Correlations')
```

## Evaluation

Check if augmentation helps your model:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Original data
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train on original
clf_original = RandomForestClassifier()
clf_original.fit(X_train, y_train)
print("Original data:")
print(classification_report(y_test, clf_original.predict(X_test)))

# Train on augmented
balanced = augment_imbalanced(
    pd.concat([X_train, y_train], axis=1),
    target_column='is_fraud'
)
X_aug = balanced.drop('is_fraud', axis=1)
y_aug = balanced['is_fraud']

clf_augmented = RandomForestClassifier()
clf_augmented.fit(X_aug, y_aug)
print("Augmented data:")
print(classification_report(y_test, clf_augmented.predict(X_test)))
```

## Complete Example

```python
import pandas as pd
from genesis import augment_imbalanced
from genesis.evaluation import AugmentationMetrics

# Load imbalanced medical data
df = pd.read_csv('patient_outcomes.csv')

print("Original distribution:")
print(df['adverse_event'].value_counts())
# adverse_event
# 0    9800
# 1     200

# Augment with quality controls
balanced, report = augment_imbalanced(
    df,
    target_column='adverse_event',
    strategy='oversample',
    ratio=0.5,  # 50% positive cases
    discrete_columns=['gender', 'diagnosis_code'],
    constraints=[
        Constraint.positive('age'),
        Constraint.range('age', 0, 120)
    ],
    quality_threshold=0.75,
    return_report=True
)

print("\nBalanced distribution:")
print(balanced['adverse_event'].value_counts())
# adverse_event
# 0    9800
# 1    4900

print(f"\nQuality score: {report.quality_score:.1%}")

# Save
balanced.to_csv('balanced_patients.csv', index=False)
```

## CLI Usage

```bash
# Balance a dataset
genesis augment data.csv \
  --target is_fraud \
  --output balanced.csv \
  --ratio 1.0

# With options
genesis augment data.csv \
  --target category \
  --output balanced.csv \
  --strategy hybrid \
  --ratio 0.5 \
  --discrete-columns region,status
```

## Comparison with Traditional Methods

| Method | Quality | Speed | Diversity |
|--------|---------|-------|-----------|
| SMOTE | ⭐⭐ | ⚡⚡⚡ | ⭐⭐ |
| ADASYN | ⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐ |
| Genesis Augment | ⭐⭐⭐⭐ | ⚡⚡ | ⭐⭐⭐⭐ |

Genesis advantages:
- Handles mixed data types
- Preserves complex relationships
- Works with constraints
- Better for high-dimensional data

## Best Practices

1. **Always evaluate downstream** - Check if augmentation helps your task
2. **Use quality thresholds** - Reject poor synthetic samples
3. **Don't over-augment** - Moderate ratios often work best
4. **Validate correlations** - Ensure relationships preserved
5. **Mark synthetic samples** - Track which are generated

## Troubleshooting

### Model still biased
- Increase ratio
- Try different augmentation strategy
- Check feature quality of synthetic samples

### Synthetic samples unrealistic
- Increase training epochs
- Add constraints for business rules
- Use CTGAN instead of simpler methods

### Performance degradation
- Lower the ratio
- Use quality threshold
- Try hybrid strategy

## Next Steps

- **[Conditional Generation](/docs/guides/conditional-generation)** - Fine-grained control
- **[Evaluation](/docs/concepts/evaluation)** - Measure augmentation quality
- **[Privacy Attacks](/docs/guides/privacy-attacks)** - Verify privacy of synthetic samples
