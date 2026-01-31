# Synthetic Data Augmentation

Genesis provides tools to augment imbalanced datasets with synthetic samples, improving ML model performance on minority classes.

## Overview

Data augmentation addresses class imbalance by generating synthetic samples for underrepresented classes while preserving the statistical properties of the original data.

```python
from genesis import augment_imbalanced

# Automatically balance the dataset
balanced_df = augment_imbalanced(df, target_column="label")
```

## Components

| Component | Purpose |
|-----------|---------|
| **SyntheticAugmenter** | Core augmentation engine |
| **AugmentationPlanner** | Plans augmentation strategy |
| **augment_imbalanced()** | Convenience function |

## Quick Start

### Basic Augmentation

```python
from genesis.augmentation import SyntheticAugmenter, augment_imbalanced

# Using convenience function
balanced = augment_imbalanced(
    df, 
    target_column="fraud_label",
    strategy="oversample"  # or "smote", "combined"
)

# Using class for more control
augmenter = SyntheticAugmenter(strategy="oversample")
augmenter.fit(df, target_column="fraud_label")
balanced = augmenter.augment()

print(f"Original: {len(df)}, Augmented: {len(balanced)}")
print(f"Class distribution:\n{balanced['fraud_label'].value_counts()}")
```

## Strategies

### 1. Oversample (Default)

Generates synthetic samples for minority classes using the fitted generator.

```python
augmenter = SyntheticAugmenter(strategy="oversample")
augmenter.fit(df, target_column="label")
balanced = augmenter.augment(target_ratio=1.0)  # Equal class sizes
```

### 2. SMOTE-like

Uses interpolation between existing minority samples.

```python
augmenter = SyntheticAugmenter(strategy="smote")
augmenter.fit(df, target_column="label")
balanced = augmenter.augment()
```

### 3. Combined

Combines oversampling minority classes with undersampling majority classes.

```python
augmenter = SyntheticAugmenter(strategy="combined")
augmenter.fit(df, target_column="label")
balanced = augmenter.augment(
    minority_ratio=0.8,  # Oversample minorities to 80% of majority
    majority_ratio=1.2   # Undersample majority to 120% of minority
)
```

## Augmentation Planning

```python
from genesis.augmentation import AugmentationPlanner

planner = AugmentationPlanner()
plan = planner.analyze(df, target_column="label")

print(f"Imbalance ratio: {plan.imbalance_ratio:.2f}")
print(f"Majority class: {plan.majority_class}")
print(f"Minority classes: {plan.minority_classes}")
print(f"Recommended strategy: {plan.recommended_strategy}")

# Get detailed recommendations
for class_name, count_needed in plan.samples_needed.items():
    print(f"  {class_name}: generate {count_needed} samples")
```

## Target Ratios

Control the final class distribution:

```python
# Perfect balance (1:1)
balanced = augmenter.augment(target_ratio=1.0)

# Partial balance (minority at 50% of majority)
balanced = augmenter.augment(target_ratio=0.5)

# Specific sample counts per class
balanced = augmenter.augment(target_counts={
    "class_a": 1000,
    "class_b": 1000,
    "class_c": 500
})
```

## Multi-Class Support

```python
# Works with any number of classes
df = pd.DataFrame({
    "feature1": np.random.randn(1000),
    "feature2": np.random.randn(1000),
    "label": ["A"]*800 + ["B"]*150 + ["C"]*40 + ["D"]*10
})

balanced = augment_imbalanced(df, "label", target_ratio=0.5)
print(balanced["label"].value_counts())
# A    800
# B    400  (augmented from 150)
# C    400  (augmented from 40)
# D    400  (augmented from 10)
```

## Quality Control

```python
augmenter = SyntheticAugmenter(
    strategy="oversample",
    quality_threshold=0.8,  # Reject low-quality samples
    validate_samples=True   # Enable validation
)

augmenter.fit(df, target_column="label")
balanced = augmenter.augment()

# Check augmentation quality
report = augmenter.quality_report()
print(f"Samples generated: {report['n_generated']}")
print(f"Samples accepted: {report['n_accepted']}")
print(f"Average quality: {report['avg_quality']:.2%}")
```

## Pipeline Integration

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("imbalanced_data.csv")
    .add_node("augment", "augment", {
        "target_column": "fraud",
        "strategy": "oversample",
        "target_ratio": 1.0
    })
    .sink("balanced_data.csv")
    .build()
)

result = pipeline.execute()
```

## Best Practices

1. **Analyze first**: Use `AugmentationPlanner` to understand your imbalance
2. **Start with oversample**: It's the safest default strategy
3. **Validate quality**: Check that synthetic samples are realistic
4. **Don't over-augment**: Extreme ratios can introduce noise
5. **Preserve test set**: Never augment your test/validation data

## CLI Usage

```bash
# Basic augmentation
genesis augment -i imbalanced.csv -o balanced.csv -t label

# With specific strategy and ratio
genesis augment -i data.csv -o balanced.csv -t fraud --strategy smote --ratio 0.5

# Analyze only (no generation)
genesis augment -i data.csv -t label --analyze-only
```

## Example: Fraud Detection

```python
import pandas as pd
from genesis.augmentation import augment_imbalanced
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load imbalanced fraud data (1% fraud rate)
df = pd.read_csv("transactions.csv")
print(df["is_fraud"].value_counts(normalize=True))
# 0    0.99
# 1    0.01

# Split BEFORE augmentation (important!)
train, test = train_test_split(df, test_size=0.2, stratify=df["is_fraud"])

# Augment only training data
train_balanced = augment_imbalanced(train, "is_fraud", target_ratio=0.3)

# Train model
X_train = train_balanced.drop("is_fraud", axis=1)
y_train = train_balanced["is_fraud"]
X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate on original test set
print(classification_report(y_test, model.predict(X_test)))
```
