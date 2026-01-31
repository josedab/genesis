---
sidebar_position: 9
title: Drift Detection
---

# Drift Detection

Monitor for statistical drift between datasets to ensure synthetic data quality and detect data evolution.

## Quick Start

```python
from genesis import detect_drift
import pandas as pd

# Compare two datasets
baseline = pd.read_csv('customers_2023.csv')
current = pd.read_csv('customers_2024.csv')

report = detect_drift(baseline, current)

print(f"Drift detected: {report.has_drift}")
print(f"Drift score: {report.drift_score:.3f}")
```

## Use Cases

### 1. Model Monitoring

Check if production data has drifted from training data:

```python
training_data = pd.read_csv('model_training_data.csv')
production_data = get_recent_production_data()

report = detect_drift(training_data, production_data)

if report.has_drift:
    alert("Data drift detected - consider retraining model")
```

### 2. Synthetic Data Validation

Verify synthetic data matches the original:

```python
real = pd.read_csv('real_customers.csv')
synthetic = pd.read_csv('synthetic_customers.csv')

report = detect_drift(real, synthetic)

# Lower drift = better synthetic data quality
if report.drift_score > 0.1:
    print("Warning: Synthetic data deviates from real data")
```

### 3. Dataset Version Comparison

Compare dataset versions over time:

```python
v1 = pd.read_csv('dataset_v1.csv')
v2 = pd.read_csv('dataset_v2.csv')

report = detect_drift(v1, v2)
print(f"Changes between versions: {report.summary()}")
```

## Drift Metrics

### Numeric Columns

```python
report = detect_drift(baseline, current)

for col in report.numeric_columns:
    metrics = report.column_metrics[col]
    print(f"{col}:")
    print(f"  KS Statistic: {metrics['ks_statistic']:.3f}")
    print(f"  KS p-value: {metrics['ks_pvalue']:.3f}")
    print(f"  Mean diff: {metrics['mean_diff']:.2f}")
    print(f"  Std diff: {metrics['std_diff']:.2f}")
```

**KS (Kolmogorov-Smirnov) Test:**
- Measures maximum difference between distributions
- p-value < 0.05 indicates significant drift

### Categorical Columns

```python
for col in report.categorical_columns:
    metrics = report.column_metrics[col]
    print(f"{col}:")
    print(f"  JS Divergence: {metrics['js_divergence']:.3f}")
    print(f"  Chi-square p-value: {metrics['chi2_pvalue']:.3f}")
    print(f"  New categories: {metrics['new_categories']}")
    print(f"  Missing categories: {metrics['missing_categories']}")
```

**JS (Jensen-Shannon) Divergence:**
- 0 = identical distributions
- 1 = completely different

## Threshold Configuration

```python
from genesis.drift import DriftDetector

detector = DriftDetector(
    numeric_threshold=0.1,      # KS statistic threshold
    categorical_threshold=0.1,  # JS divergence threshold
    pvalue_threshold=0.05       # Significance level
)

report = detector.detect(baseline, current)
```

## Per-Column Analysis

```python
report = detect_drift(baseline, current)

# Columns with drift
drifted = report.drifted_columns()
print(f"Columns with drift: {drifted}")

# Most drifted column
worst = report.most_drifted_column()
print(f"Highest drift: {worst['column']} ({worst['score']:.3f})")

# Visualization
report.plot_drifted_columns()  # Bar chart of drift scores
```

## Distribution Comparison

Visualize distribution changes:

```python
import matplotlib.pyplot as plt

report = detect_drift(baseline, current)

# Plot specific column
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Numeric column comparison
report.plot_distribution('age', ax=ax[0])

# Categorical column comparison
report.plot_distribution('status', ax=ax[1])

plt.tight_layout()
plt.show()
```

## Continuous Monitoring

Set up automated drift monitoring:

```python
from genesis.drift import DriftMonitor

monitor = DriftMonitor(
    baseline=training_data,
    check_interval='daily',
    alert_threshold=0.15
)

# Check new data
for batch in daily_batches:
    report = monitor.check(batch)
    
    if report.alert:
        send_notification(f"Drift alert: {report.summary()}")
```

## Drift Report

Generate detailed reports:

```python
report = detect_drift(baseline, current)

# Summary
print(report.summary())

# Detailed text report
print(report.detailed_report())

# HTML report
report.save_html('drift_report.html')

# JSON for programmatic use
import json
print(json.dumps(report.to_dict(), indent=2))
```

## Complete Example

```python
import pandas as pd
from genesis.drift import DriftDetector

# Load data
train_data = pd.read_csv('model_training.csv')
prod_data = pd.read_csv('recent_production.csv')

# Configure detector
detector = DriftDetector(
    numeric_threshold=0.1,
    categorical_threshold=0.1,
    columns_to_monitor=['age', 'income', 'category', 'region']
)

# Detect drift
report = detector.detect(train_data, prod_data)

# Overall assessment
print(f"Drift detected: {report.has_drift}")
print(f"Overall drift score: {report.drift_score:.3f}")

# Per-column breakdown
print("\nColumn-wise drift:")
for col, metrics in report.column_metrics.items():
    status = "⚠️ DRIFT" if metrics['has_drift'] else "✓ OK"
    print(f"  {col}: {metrics['score']:.3f} {status}")

# Recommendations
print("\nRecommendations:")
for rec in report.recommendations:
    print(f"  - {rec}")

# Save report
report.save_html('drift_analysis.html')
```

## CLI Usage

```bash
# Basic drift check
genesis drift baseline.csv current.csv

# With options
genesis drift baseline.csv current.csv \
  --threshold 0.1 \
  --columns age,income,status \
  --output drift_report.html

# JSON output for automation
genesis drift baseline.csv current.csv --format json
```

## Integration with Pipelines

```python
from genesis import Pipeline, detect_drift

# Use drift detection in a pipeline
pipeline = Pipeline([
    ('load', DataLoader('current_data.csv')),
    ('check_drift', DriftChecker(baseline='training_data.csv', threshold=0.15)),
    ('generate', SyntheticGenerator(method='ctgan')),
    ('validate', QualityValidator(min_score=0.8))
])

result = pipeline.run()

if result['drift_check'].has_drift:
    print("Warning: Input data has drifted from baseline")
```

## Best Practices

1. **Establish baselines early** - Save snapshots of training data
2. **Monitor continuously** - Drift often happens gradually
3. **Set appropriate thresholds** - Based on your domain tolerance
4. **Investigate root causes** - Drift symptoms often indicate upstream issues
5. **Version your baselines** - Track what you're comparing against

## Troubleshooting

### False positives
- Increase threshold values
- Check sample sizes (small samples = noisy statistics)
- Consider seasonal patterns

### Missing drift
- Lower threshold values
- Check all relevant columns
- Consider multivariate drift (correlations)

### Slow performance
- Sample large datasets
- Monitor fewer columns
- Use approximate methods

## Next Steps

- **[Versioning](/docs/guides/versioning)** - Track dataset versions
- **[Evaluation](/docs/concepts/evaluation)** - Quality metrics
- **[Pipelines](/docs/guides/pipelines)** - Automate drift checks
