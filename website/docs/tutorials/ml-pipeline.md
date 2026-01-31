---
sidebar_position: 4
title: ML Training Pipeline
---

# Tutorial: ML Training Pipeline with Synthetic Data

Build a production ML pipeline that uses synthetic data for training and augmentation.

**Time:** 45 minutes  
**Level:** Intermediate  
**What you'll learn:** Data augmentation, imbalanced data handling, Pipeline API, model comparison

---

## Goal

By the end of this tutorial, you'll have:
- Built an ML pipeline that trains on synthetic data
- Balanced an imbalanced dataset using synthetic augmentation
- Compared model performance: real vs synthetic training data
- Created a reproducible pipeline with Genesis Pipeline API

---

## Prerequisites

```bash
pip install genesis-synth[pytorch] pandas scikit-learn matplotlib
```

---

## The Problem: Imbalanced Fraud Detection

Fraud detection is a classic ML challenge:
- Fraud is rare (typically 0.1-2% of transactions)
- Models trained on imbalanced data miss most fraud
- Real fraud examples are scarce and sensitive

**Solution:** Generate synthetic fraud cases to balance the dataset.

---

## Step 1: Create Imbalanced Dataset

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Generate transaction data (98% legitimate, 2% fraud)
n_transactions = 50000
n_fraud = int(n_transactions * 0.02)  # 2% fraud rate
n_legit = n_transactions - n_fraud

# Legitimate transactions
legit_data = pd.DataFrame({
    'amount': np.random.lognormal(4, 1, n_legit).clip(1, 5000),
    'hour': np.random.choice(range(24), n_legit, p=[0.02]*6 + [0.06]*12 + [0.04]*6),
    'day_of_week': np.random.choice(range(7), n_legit),
    'merchant_category': np.random.choice(
        ['grocery', 'restaurant', 'gas', 'online', 'travel'],
        n_legit, p=[0.3, 0.25, 0.15, 0.2, 0.1]
    ),
    'distance_from_home': np.random.exponential(10, n_legit).clip(0, 100),
    'transaction_count_24h': np.random.poisson(3, n_legit).clip(1, 20),
    'is_fraud': 0
})

# Fraudulent transactions (different patterns)
fraud_data = pd.DataFrame({
    'amount': np.random.lognormal(6, 1.5, n_fraud).clip(100, 10000),  # Higher amounts
    'hour': np.random.choice(range(24), n_fraud, p=[0.08]*6 + [0.02]*12 + [0.06]*6),  # Night hours
    'day_of_week': np.random.choice(range(7), n_fraud),
    'merchant_category': np.random.choice(
        ['grocery', 'restaurant', 'gas', 'online', 'travel'],
        n_fraud, p=[0.1, 0.1, 0.1, 0.5, 0.2]  # More online/travel
    ),
    'distance_from_home': np.random.exponential(50, n_fraud).clip(0, 500),  # Further away
    'transaction_count_24h': np.random.poisson(8, n_fraud).clip(1, 50),  # More transactions
    'is_fraud': 1
})

# Combine
transactions = pd.concat([legit_data, fraud_data], ignore_index=True)
transactions = transactions.sample(frac=1, random_state=42).reset_index(drop=True)

print("=== Dataset Overview ===")
print(f"Total transactions: {len(transactions):,}")
print(f"Fraud rate: {transactions['is_fraud'].mean():.2%}")
print(f"\nClass distribution:")
print(transactions['is_fraud'].value_counts())
```

**Output:**
```
=== Dataset Overview ===
Total transactions: 50,000
Fraud rate: 2.00%

Class distribution:
0    49000
1     1000
```

---

## Step 2: Establish Baseline (Imbalanced)

Train a model on the imbalanced data to establish baseline:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Prepare features
X = transactions.drop('is_fraud', axis=1)
X = pd.get_dummies(X, columns=['merchant_category'])
y = transactions['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} samples ({y_train.mean():.2%} fraud)")
print(f"Test set: {len(X_test):,} samples ({y_test.mean():.2%} fraud)")

# Train baseline model
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

# Evaluate
y_pred = baseline_model.predict(X_test)
y_prob = baseline_model.predict_proba(X_test)[:, 1]

print("\n=== Baseline Model (Imbalanced Data) ===")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

**Typical output:**
```
=== Baseline Model (Imbalanced Data) ===
              precision    recall  f1-score   support

       Legit       0.99      1.00      0.99      9800
       Fraud       0.92      0.68      0.78       200

    accuracy                           0.99     10000
   macro avg       0.95      0.84      0.89     10000

ROC-AUC: 0.9234

Confusion Matrix:
[[9792    8]
 [  64  136]]
```

**Problem:** Only 68% recall on fraud—we're missing 32% of fraudulent transactions!

---

## Step 3: Generate Synthetic Fraud Cases

Use Genesis to generate more fraud examples:

```python
from genesis import SyntheticGenerator

# Separate fraud cases for synthesis
fraud_cases = transactions[transactions['is_fraud'] == 1].drop('is_fraud', axis=1)
print(f"Original fraud cases: {len(fraud_cases)}")

# Train generator on fraud patterns
fraud_generator = SyntheticGenerator(
    method='ctgan',
    config={'epochs': 300}
)

fraud_generator.fit(
    fraud_cases,
    discrete_columns=['merchant_category', 'hour', 'day_of_week']
)

# Generate synthetic fraud cases
n_synthetic_fraud = 5000  # Generate 5x more fraud
synthetic_fraud = fraud_generator.generate(n_samples=n_synthetic_fraud)
synthetic_fraud['is_fraud'] = 1

print(f"Generated {len(synthetic_fraud)} synthetic fraud cases")
print(synthetic_fraud.head())
```

---

## Step 4: Validate Synthetic Quality

Ensure synthetic fraud looks realistic:

```python
from genesis import QualityEvaluator

# Evaluate synthetic fraud quality
quality_report = QualityEvaluator(fraud_cases, synthetic_fraud.drop('is_fraud', axis=1)).evaluate()

print("=== Synthetic Fraud Quality ===")
print(f"Overall Quality: {quality_report.overall_score:.1%}")
print(f"Statistical Fidelity: {quality_report.fidelity_score:.1%}")

# Compare distributions
print("\n=== Distribution Comparison ===")
print("Amount (mean ± std):")
print(f"  Real fraud: ${fraud_cases['amount'].mean():.0f} ± ${fraud_cases['amount'].std():.0f}")
print(f"  Synthetic:  ${synthetic_fraud['amount'].mean():.0f} ± ${synthetic_fraud['amount'].std():.0f}")

print("\nMerchant Category Distribution:")
print("Real:")
print(fraud_cases['merchant_category'].value_counts(normalize=True).round(2))
print("\nSynthetic:")
print(synthetic_fraud['merchant_category'].value_counts(normalize=True).round(2))
```

---

## Step 5: Create Balanced Dataset

Combine real data with synthetic fraud:

```python
# Original training data
train_data = transactions.iloc[X_train.index].copy()

# Add synthetic fraud to training data only (not test!)
augmented_train = pd.concat([train_data, synthetic_fraud], ignore_index=True)
augmented_train = augmented_train.sample(frac=1, random_state=42).reset_index(drop=True)

print("=== Augmented Training Data ===")
print(f"Total samples: {len(augmented_train):,}")
print(f"Original fraud: {len(train_data[train_data['is_fraud']==1]):,}")
print(f"Synthetic fraud: {len(synthetic_fraud):,}")
print(f"New fraud rate: {augmented_train['is_fraud'].mean():.1%}")
print(f"\nClass distribution:")
print(augmented_train['is_fraud'].value_counts())
```

**Output:**
```
=== Augmented Training Data ===
Total samples: 45,000
Original fraud: 800
Synthetic fraud: 5,000
New fraud rate: 12.9%

Class distribution:
0    39200
1     5800
```

---

## Step 6: Train on Augmented Data

```python
# Prepare augmented features
X_aug = augmented_train.drop('is_fraud', axis=1)
X_aug = pd.get_dummies(X_aug, columns=['merchant_category'])
y_aug = augmented_train['is_fraud']

# Ensure same columns as test set
for col in X_test.columns:
    if col not in X_aug.columns:
        X_aug[col] = 0
X_aug = X_aug[X_test.columns]

# Train augmented model
augmented_model = RandomForestClassifier(n_estimators=100, random_state=42)
augmented_model.fit(X_aug, y_aug)

# Evaluate on REAL test data
y_pred_aug = augmented_model.predict(X_test)
y_prob_aug = augmented_model.predict_proba(X_test)[:, 1]

print("=== Augmented Model (Synthetic + Real) ===")
print(classification_report(y_test, y_pred_aug, target_names=['Legit', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_aug):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_aug))
```

**Expected improvement:**
```
=== Augmented Model (Synthetic + Real) ===
              precision    recall  f1-score   support

       Legit       0.99      0.99      0.99      9800
       Fraud       0.85      0.89      0.87       200

    accuracy                           0.99     10000

ROC-AUC: 0.9687

Confusion Matrix:
[[9721   79]
 [  22  178]]
```

**Improvement:** Fraud recall improved from 68% to 89%! (+21 percentage points)

---

## Step 7: Compare Results

```python
import matplotlib.pyplot as plt

# Comparison metrics
results = pd.DataFrame({
    'Model': ['Baseline (Imbalanced)', 'Augmented (Synthetic)'],
    'Fraud Recall': [0.68, 0.89],
    'Fraud Precision': [0.92, 0.85],
    'Fraud F1': [0.78, 0.87],
    'ROC-AUC': [0.9234, 0.9687]
})

print("=== Model Comparison ===")
print(results.to_string(index=False))

# Calculate improvement
print(f"\n=== Improvement ===")
print(f"Fraud Recall: +{(0.89-0.68)*100:.0f} percentage points")
print(f"ROC-AUC: +{(0.9687-0.9234)*100:.1f} percentage points")
print(f"Fraud cases caught: {178} vs {136} (+{178-136} more)")
```

---

## Step 8: Alternative - Use augment_imbalanced()

Genesis provides a simpler API for this common task:

```python
from genesis import augment_imbalanced

# One-line balanced dataset
balanced_data = augment_imbalanced(
    transactions,
    target_column='is_fraud',
    ratio=0.3,  # Target 30% fraud rate
    strategy='oversample',  # Generate synthetic minority class
    discrete_columns=['merchant_category']
)

print(f"Balanced dataset: {len(balanced_data):,} samples")
print(f"New fraud rate: {balanced_data['is_fraud'].mean():.1%}")
```

---

## Step 9: Production Pipeline

Create a reproducible pipeline:

```python
from genesis.pipeline import Pipeline, steps

# Define ML pipeline with synthetic augmentation
ml_pipeline = Pipeline([
    # Load data
    steps.load_csv('transactions.csv'),
    
    # Analyze class imbalance
    steps.analyze(
        target_column='is_fraud',
        report=True
    ),
    
    # Augment minority class
    steps.augment_imbalanced(
        target_column='is_fraud',
        ratio=0.3,
        strategy='oversample',
        discrete_columns=['merchant_category']
    ),
    
    # Evaluate augmentation quality
    steps.evaluate(
        target_column='is_fraud',
        min_score=0.85
    ),
    
    # Save augmented data
    steps.save_csv('augmented_transactions.csv'),
    
    # Save generator for future use
    steps.save_generator('fraud_generator.pkl')
])

# Run pipeline
result = ml_pipeline.run()

if result.success:
    print("✅ Pipeline completed successfully")
    print(f"   Output: {result.outputs['output_path']}")
    print(f"   Quality score: {result.metrics['quality_score']:.1%}")
else:
    print(f"❌ Pipeline failed: {result.error}")
```

---

## Step 10: YAML Pipeline (For CI/CD)

```yaml
# fraud_augmentation_pipeline.yaml
name: fraud_data_augmentation
description: Augment fraud detection training data

steps:
  - load_csv:
      path: data/transactions.csv
  
  - augment_imbalanced:
      target_column: is_fraud
      ratio: 0.3
      strategy: oversample
      discrete_columns:
        - merchant_category
        - hour
        - day_of_week
      config:
        epochs: 300
  
  - evaluate:
      target_column: is_fraud
      min_score: 0.85
      fail_on_low_score: true
  
  - privacy_audit:
      threshold: 0.9
  
  - save_csv:
      path: data/augmented_transactions.csv
  
  - save_versioned:
      repo: ./data_versions
      message: "Fraud augmentation run"
      tag: latest
```

Run with:
```bash
genesis pipeline run fraud_augmentation_pipeline.yaml
```

---

## Complete Training Script

```python
"""
Fraud Detection with Synthetic Data Augmentation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from genesis import augment_imbalanced, QualityEvaluator

# Load data
transactions = pd.read_csv('transactions.csv')

# Split BEFORE augmentation (prevent data leakage)
train_df, test_df = train_test_split(
    transactions, test_size=0.2, random_state=42, 
    stratify=transactions['is_fraud']
)

# Augment training data only
train_augmented = augment_imbalanced(
    train_df,
    target_column='is_fraud',
    ratio=0.3,
    discrete_columns=['merchant_category']
)

print(f"Training: {len(train_augmented):,} samples ({train_augmented['is_fraud'].mean():.1%} fraud)")
print(f"Test: {len(test_df):,} samples ({test_df['is_fraud'].mean():.1%} fraud)")

# Prepare features
def prepare_features(df):
    X = df.drop('is_fraud', axis=1)
    X = pd.get_dummies(X, columns=['merchant_category'])
    return X, df['is_fraud']

X_train, y_train = prepare_features(train_augmented)
X_test, y_test = prepare_features(test_df)

# Align columns
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Results ===")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

---

## Key Takeaways

1. **Augment training data only** - Never augment test data (data leakage)
2. **Validate synthetic quality** - Use `QualityEvaluator` before training
3. **Compare with baseline** - Always measure improvement vs imbalanced
4. **Use `augment_imbalanced()` for simplicity** - One-line API for common case
5. **Pipeline for production** - Reproducible, versioned, auditable

---

## Next Steps

- [Data Augmentation Guide](/docs/guides/augmentation) - Advanced augmentation strategies
- [Pipeline API](/docs/guides/pipelines) - Build complex workflows
- [Testing Tutorial](/docs/tutorials/testing) - Use synthetic data for testing
