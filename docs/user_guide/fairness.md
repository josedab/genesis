# Fairness-Aware Generation

Genesis provides tools for detecting and correcting bias in synthetic data, supporting fairness in ML pipelines.

## Overview

| Class | Purpose |
|-------|---------|
| **FairnessAnalyzer** | Detect and measure bias in data |
| **FairGenerator** | Generate data with fairness constraints |
| **CounterfactualGenerator** | Create counterfactual examples |
| **FairnessAudit** | Compare fairness before/after generation |

## Installation

```bash
pip install genesis-synth[fairness]  # Optional fairness extras
```

## Bias Detection

Analyze datasets for bias across sensitive attributes:

```python
from genesis.fairness import FairnessAnalyzer

analyzer = FairnessAnalyzer(
    sensitive_attributes=["gender", "race"],
    outcome_column="hired",
    privileged_groups={"gender": "male", "race": "white"},
    positive_outcome=1,
)

# Analyze for bias
reports = analyzer.analyze(hiring_data)

# Check demographic parity
for attr, report in reports.items():
    dp = report.metrics["demographic_parity"]
    print(f"{attr}: ratio={dp.value:.3f}, fair={dp.is_fair}")
    
    # View recommendations
    for rec in report.recommendations:
        print(f"  - {rec}")
```

### Supported Metrics

| Metric | Description | Fair Threshold |
|--------|-------------|----------------|
| **Demographic Parity** | Equal positive rates across groups | ≥ 0.8 |
| **Disparate Impact** | 80% rule for adverse impact | ≥ 0.8 |
| **Equal Opportunity** | Equal TPR across groups | Within 5% |
| **Equalized Odds** | Equal TPR and FPR | Within 5% |

### Understanding Results

```python
report = reports["gender"]

# Group statistics
for group, stats in report.group_statistics.items():
    print(f"{group}:")
    print(f"  Count: {stats['count']}")
    print(f"  Positive rate: {stats['positive_rate']:.2%}")

# Metrics
dp = report.metrics["demographic_parity"]
print(f"Demographic Parity: {dp.value:.3f}")
print(f"  Privileged rate: {dp.privileged_value:.3f}")
print(f"  Unprivileged rate: {dp.unprivileged_value:.3f}")
print(f"  Is fair: {dp.is_fair}")
```

## Fair Data Generation

Generate synthetic data that corrects historical biases:

### Resampling Strategy

Oversample underrepresented groups to achieve balance:

```python
from genesis.fairness import FairGenerator

generator = FairGenerator(
    sensitive_attr="gender",
    outcome_col="hired",
    strategy="resampling",
    target_ratio=1.0,  # Perfect parity
)

fair_data = generator.generate(
    biased_data,
    n_samples=10000,
    random_state=42,
)

# Verify fairness improved
new_reports = analyzer.analyze(fair_data)
print(f"New DP ratio: {new_reports['gender'].metrics['demographic_parity'].value:.3f}")
```

### Reweighting Strategy

Assign sampling weights to balance outcomes:

```python
generator = FairGenerator(
    sensitive_attr="gender",
    outcome_col="hired",
    strategy="reweighting",
)

fair_data = generator.generate(biased_data, n_samples=10000)
```

### Counterfactual Strategy

Generate counterfactual examples where sensitive attributes are flipped:

```python
generator = FairGenerator(
    sensitive_attr="gender",
    strategy="counterfactual",
)

augmented_data = generator.generate(original_data, n_samples=20000)
# Half original, half counterfactual
```

## Counterfactual Generation

Create counterfactual examples for causal fairness analysis:

```python
from genesis.fairness import CounterfactualGenerator

generator = CounterfactualGenerator(
    sensitive_attr="gender",
    causal_features=["years_experience"],  # Features causally affected
    preserve_features=["education"],        # Features to keep unchanged
)

# Learn causal relationships
generator.fit(original_data)

# Generate counterfactuals
counterfactuals = generator.generate(original_data)

# All gender values are flipped
# Causally-affected features are adjusted accordingly
```

### Causal Adjustments

The counterfactual generator learns and applies causal adjustments:

```python
# If males have on average 2 more years experience,
# when flipping male -> female, experience is reduced by 2
# to reflect the counterfactual scenario
```

## Fairness Audit

Compare fairness between original and synthetic data:

```python
from genesis.fairness import FairnessAudit

audit = FairnessAudit(
    sensitive_attrs=["gender", "race"],
    outcome_col="hired",
)

# Compare datasets
results = audit.compare(original_data, synthetic_data)

print(f"Original DP (gender): {results['original']['gender']:.3f}")
print(f"Synthetic DP (gender): {results['synthetic']['gender']:.3f}")
print(f"Improvement: {results['improvement']['gender']*100:+.1f}%")

# Generate human-readable report
report = audit.generate_report(original_data, synthetic_data)
print(report)
```

### Sample Audit Report

```
============================================================
FAIRNESS AUDIT REPORT
============================================================

Demographic Parity Comparison:
----------------------------------------
  gender:
    Original:  0.577
    Synthetic: 0.982 ✓
    Change:    ↑ +70.2%

  race:
    Original:  0.651
    Synthetic: 0.945 ✓
    Change:    ↑ +45.2%

----------------------------------------
Legend: ✓ = Fair (≥0.8), ✗ = Unfair (<0.8)
============================================================
```

## Quick Balance Function

For simple dataset balancing without full generation:

```python
from genesis.fairness import balance_dataset

# Oversample minority groups
balanced = balance_dataset(
    data,
    sensitive_attr="gender",
    strategy="oversample",
    random_state=42,
)

# Or undersample majority groups
balanced = balance_dataset(
    data,
    sensitive_attr="gender", 
    strategy="undersample",
)
```

## Fairness Constraints

Define custom fairness constraints for generation:

```python
from genesis.fairness import DemographicParityConstraint

constraint = DemographicParityConstraint(
    sensitive_attr="gender",
    outcome_col="hired",
    target_ratio=1.0,
    tolerance=0.05,
)

# Fit to learn group rates
constraint.fit(original_data)

# Check if data satisfies constraint
is_fair = constraint.check(synthetic_data)

# Get sampling weights
weights = constraint.compute_weight(row)
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator
from genesis.fairness import FairnessAnalyzer, FairGenerator, FairnessAudit

# Load potentially biased data
hiring_data = pd.read_csv("hiring_decisions.csv")

# 1. Analyze for bias
analyzer = FairnessAnalyzer(
    sensitive_attributes=["gender"],
    outcome_column="hired",
    privileged_groups={"gender": "male"},
)

original_reports = analyzer.analyze(hiring_data)
print("Original bias analysis:")
print(f"  Demographic parity: {original_reports['gender'].metrics['demographic_parity'].value:.3f}")

# 2. Generate fair synthetic data
fair_gen = FairGenerator(
    sensitive_attr="gender",
    outcome_col="hired",
    strategy="resampling",
)

fair_synthetic = fair_gen.generate(hiring_data, n_samples=10000)

# 3. Verify fairness improved
synthetic_reports = analyzer.analyze(fair_synthetic)
print("Synthetic data analysis:")
print(f"  Demographic parity: {synthetic_reports['gender'].metrics['demographic_parity'].value:.3f}")

# 4. Generate audit report
audit = FairnessAudit(
    sensitive_attrs=["gender"],
    outcome_col="hired",
)
print(audit.generate_report(hiring_data, fair_synthetic))
```

## Best Practices

### 1. Always Analyze First

```python
# Understand bias before trying to correct it
reports = analyzer.analyze(data)
for attr, report in reports.items():
    print(f"\n{attr}:")
    for rec in report.recommendations:
        print(f"  {rec}")
```

### 2. Choose Strategy Based on Data

```python
# Resampling: Good for small bias, preserves relationships
# Reweighting: Good for moderate bias, fast
# Counterfactual: Good for causal fairness, research settings
```

### 3. Validate After Generation

```python
# Always verify fairness improved
audit = FairnessAudit(...)
results = audit.compare(original, synthetic)

if results["improvement"]["gender"] < 0:
    print("Warning: Fairness degraded!")
```

### 4. Consider Multiple Attributes

```python
# Intersectional fairness
analyzer = FairnessAnalyzer(
    sensitive_attributes=["gender", "race", "age_group"],
    outcome_column="hired",
)
```

## Configuration Reference

### FairnessAnalyzer

| Parameter | Type | Description |
|-----------|------|-------------|
| `sensitive_attributes` | List[str] | Columns with sensitive attributes |
| `outcome_column` | str | Column with outcome to analyze |
| `privileged_groups` | Dict | Mapping of attribute to privileged value |
| `positive_outcome` | Any | Value representing positive outcome |
| `fairness_threshold` | float | Minimum ratio for fairness (default: 0.8) |

### FairGenerator

| Parameter | Type | Description |
|-----------|------|-------------|
| `sensitive_attr` | str | Sensitive attribute column |
| `outcome_col` | str | Outcome column (optional) |
| `strategy` | str | 'resampling', 'reweighting', or 'counterfactual' |
| `target_ratio` | float | Target fairness ratio (default: 1.0) |
