# ADR-0009: Layered Evaluation Architecture

## Status

Accepted

## Context

Evaluating synthetic data quality is multi-dimensional. A synthetic dataset might:
- Perfectly match statistical distributions but fail privacy tests
- Preserve ML model performance but have unrealistic individual records
- Pass all metrics but contain logical inconsistencies

Different users care about different aspects:
- **Data scientists**: ML utility, can I train models on this?
- **Compliance officers**: Privacy metrics, is this safe to share?
- **Domain experts**: Statistical fidelity, does this look realistic?
- **Developers**: Quick sanity checks, is this obviously broken?

A monolithic evaluation function that computes everything is:
- Slow (privacy metrics can take minutes)
- Wasteful (often only need subset of metrics)
- Hard to extend (adding new metrics requires modifying core)
- Confusing (users don't know which metrics matter for their use case)

## Decision

We implement a **layered evaluation architecture** with three independent modules:

```
genesis/evaluation/
├── __init__.py          # Unified QualityEvaluator entry point
├── statistical.py       # Statistical fidelity metrics
├── ml_utility.py        # ML model performance metrics
├── privacy.py           # Privacy and disclosure metrics
├── report.py            # QualityReport aggregation
└── evaluator.py         # Orchestration layer
```

### Layer 1: Statistical Fidelity

Measures how well synthetic data matches real data distributions:

```python
from genesis.evaluation.statistical import (
    compute_statistical_fidelity,
    ks_test,
    chi_squared_test,
    correlation_similarity,
    distribution_shape_metrics,
)

stats = compute_statistical_fidelity(real_data, synthetic_data)
# StatisticalFidelity(
#     overall_score=0.87,
#     column_scores={'age': 0.92, 'income': 0.85, ...},
#     ks_statistics={'age': 0.05, ...},
#     correlation_diff=0.03,
# )
```

### Layer 2: ML Utility

Measures whether models trained on synthetic data perform well on real data:

```python
from genesis.evaluation.ml_utility import (
    compute_ml_utility,
    train_on_synthetic_test_on_real,
    discriminator_score,
)

utility = compute_ml_utility(
    real_train=real_data,
    synthetic=synthetic_data,
    real_test=holdout_data,
    target_column='churn',
)
# MLUtility(
#     overall_score=0.91,
#     tstr_score=0.88,  # Train-Synthetic-Test-Real
#     trtr_score=0.92,  # Train-Real-Test-Real (baseline)
#     discriminator_auc=0.55,  # Closer to 0.5 = better
# )
```

### Layer 3: Privacy Metrics

Measures disclosure and re-identification risks:

```python
from genesis.evaluation.privacy import (
    compute_privacy_metrics,
    distance_to_closest_record,
    membership_inference_attack,
    attribute_disclosure_risk,
)

privacy = compute_privacy_metrics(real_data, synthetic_data)
# PrivacyMetrics(
#     reidentification_risk=0.02,
#     dcr_score=0.15,
#     membership_inference_risk=0.52,
#     attribute_disclosure_risk=0.05,
# )
```

### Unified Interface

The `QualityEvaluator` orchestrates all layers:

```python
from genesis.evaluation import QualityEvaluator

evaluator = QualityEvaluator()

# Full evaluation (all metrics)
report = evaluator.evaluate(real_data, synthetic_data)

# Selective evaluation
report = evaluator.evaluate(
    real_data, 
    synthetic_data,
    metrics=['statistical', 'privacy'],  # Skip ML utility
)

# Quick sanity check (fast subset)
report = evaluator.quick_check(real_data, synthetic_data)
```

## Consequences

### Positive

- **Modularity**: each layer can be used independently
- **Performance**: only compute what you need
- **Extensibility**: add new metrics without touching core
- **Clarity**: users understand which dimension each metric addresses
- **Testability**: each layer has focused unit tests
- **Documentation**: metrics grouped by purpose

### Negative

- **Multiple imports**: users must know which module to import
- **Aggregation complexity**: combining scores across layers requires weighting
- **Consistency**: must ensure layers don't duplicate computations

### Mitigations

1. **Unified entry point**: `QualityEvaluator` handles orchestration
   ```python
   from genesis.evaluation import QualityEvaluator
   report = QualityEvaluator().evaluate(real, synthetic)
   ```

2. **QualityReport aggregation**: combines all metrics with sensible defaults
   ```python
   report.overall_score  # Weighted combination
   report.statistical_fidelity  # Layer-specific
   report.ml_utility
   report.privacy_metrics
   ```

3. **Presets** for common evaluation scenarios:
   ```python
   evaluator.evaluate(real, syn, preset='quick')      # Fast checks
   evaluator.evaluate(real, syn, preset='standard')   # Balanced
   evaluator.evaluate(real, syn, preset='thorough')   # Everything
   evaluator.evaluate(real, syn, preset='compliance') # Privacy focus
   ```

## Metrics Reference

### Statistical Fidelity

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| KS Statistic | Kolmogorov-Smirnov distance per column | 0-1 | < 0.1 |
| Chi-Squared | Categorical distribution similarity | 0-∞ | p > 0.05 |
| Correlation Diff | Difference in correlation matrices | 0-2 | < 0.1 |
| Distribution Shape | Skewness, kurtosis similarity | 0-1 | > 0.8 |

### ML Utility

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| TSTR Score | Train-Synthetic, Test-Real accuracy | 0-1 | > 0.9 × TRTR |
| TRTR Score | Train-Real, Test-Real (baseline) | 0-1 | Baseline |
| Discriminator AUC | Can model distinguish real vs synthetic? | 0.5-1 | < 0.6 |
| Feature Importance Similarity | Do same features matter? | 0-1 | > 0.8 |

### Privacy

| Metric | Description | Range | Good Value |
|--------|-------------|-------|------------|
| DCR | Distance to Closest Record | 0-1 | > 0.1 |
| Re-identification Risk | Probability of unique record match | 0-1 | < 0.05 |
| Membership Inference | Attack success rate | 0.5-1 | < 0.55 |
| Attribute Disclosure | Sensitive attribute inference risk | 0-1 | < 0.1 |

## Report Format

```python
report = evaluator.evaluate(real, synthetic)

print(report.summary())
# ┌─────────────────────────────────────────────┐
# │         Synthetic Data Quality Report       │
# ├─────────────────────────────────────────────┤
# │ Overall Score:           0.85 / 1.00        │
# ├─────────────────────────────────────────────┤
# │ Statistical Fidelity:    0.87               │
# │   - Column distributions: PASS              │
# │   - Correlations:         PASS              │
# ├─────────────────────────────────────────────┤
# │ ML Utility:               0.91              │
# │   - TSTR/TRTR ratio:      0.96              │
# │   - Discriminator AUC:    0.54              │
# ├─────────────────────────────────────────────┤
# │ Privacy:                  0.78              │
# │   - Re-identification:    LOW RISK          │
# │   - Membership inference: LOW RISK          │
# └─────────────────────────────────────────────┘

report.to_dict()   # Machine-readable
report.to_html()   # Visual report
```
