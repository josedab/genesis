# Data Drift Detection

Genesis provides tools to detect drift between datasets and generate synthetic data that adapts to changing data distributions.

## Overview

Data drift detection helps you:
- Monitor changes between original and synthetic data
- Detect concept drift in production data over time
- Generate drift-aware synthetic data that matches current distributions

```python
from genesis import detect_drift

report = detect_drift(baseline_df, current_df)
print(f"Drift detected: {report.has_significant_drift}")
print(f"Drifted columns: {report.drifted_columns}")
```

## Components

| Component | Purpose |
|-----------|---------|
| **DataDriftDetector** | Compares two datasets for distribution shifts |
| **DriftAwareGenerator** | Generates data adapting to detected drift |
| **ContinuousMonitor** | Monitors ongoing data streams for drift |
| **detect_drift()** | Convenience function |

## Drift Detection

### Basic Detection

```python
from genesis.drift import DataDriftDetector, detect_drift

# Using convenience function
report = detect_drift(baseline_df, current_df)

# Using class for more control
detector = DataDriftDetector()
report = detector.detect(
    baseline=original_df,
    current=new_df,
    significance_level=0.05
)

print(f"Overall drift score: {report.overall_drift_score:.3f}")
print(f"Has significant drift: {report.has_significant_drift}")

for col, result in report.column_results.items():
    if result.has_drift:
        print(f"  {col}: {result.drift_type} drift (p={result.p_value:.4f})")
```

### Detection Methods

| Data Type | Method | What it Measures |
|-----------|--------|-----------------|
| Numeric | Kolmogorov-Smirnov test | Distribution shape difference |
| Categorical | Jensen-Shannon divergence | Category frequency changes |
| Numeric | Population Stability Index (PSI) | Binned distribution shift |

```python
detector = DataDriftDetector(
    numeric_method="ks",        # or "psi", "wasserstein"
    categorical_method="js",     # or "chi2", "psi"
    significance_level=0.05
)
```

## Drift Report

```python
report = detect_drift(baseline_df, current_df)

# Summary
print(report.summary())

# Per-column details
for col, result in report.column_results.items():
    print(f"""
Column: {col}
  Data Type: {result.data_type}
  Has Drift: {result.has_drift}
  Drift Score: {result.drift_score:.4f}
  P-value: {result.p_value:.4f}
  Method: {result.method}
""")

# Export
report.to_json("drift_report.json")
report.to_html("drift_report.html")
```

## Drift-Aware Generation

Generate synthetic data that adapts to detected drift:

```python
from genesis.drift import DriftAwareGenerator

generator = DriftAwareGenerator()

# Fit on baseline data
generator.fit(baseline_df)

# Generate data matching current distribution
synthetic = generator.generate(
    n_samples=1000,
    target_distribution=current_df,
    drift_adaptation="weighted"  # or "blend", "retrain"
)
```

### Adaptation Strategies

| Strategy | Behavior |
|----------|----------|
| **weighted** | Blends baseline and target distributions based on drift |
| **blend** | Equal mix of baseline patterns and target distribution |
| **retrain** | Fully retrains on target distribution |

```python
# Heavy adaptation to target
synthetic = generator.generate(
    n_samples=1000,
    target_distribution=current_df,
    drift_adaptation="weighted",
    adaptation_strength=0.8  # 80% weight to target
)

# Light adaptation
synthetic = generator.generate(
    n_samples=1000,
    target_distribution=current_df,
    drift_adaptation="weighted",
    adaptation_strength=0.3  # 30% weight to target
)
```

## Continuous Monitoring

Monitor production data streams for drift:

```python
from genesis.drift import ContinuousMonitor

monitor = ContinuousMonitor(
    baseline=baseline_df,
    check_interval="1h",  # Check every hour
    alert_threshold=0.1   # Alert if drift score > 0.1
)

# Register alert callback
def on_drift_detected(report):
    print(f"ALERT: Drift detected in {report.drifted_columns}")
    # Trigger retraining, alerts, etc.

monitor.on_drift(on_drift_detected)

# Add new data batches
for batch in data_stream:
    monitor.add_batch(batch)
    
    if monitor.has_significant_drift():
        report = monitor.get_latest_report()
        print(f"Drift score: {report.overall_drift_score}")
```

## Population Stability Index (PSI)

A common industry metric for drift:

```python
from genesis.drift import calculate_psi

psi_scores = calculate_psi(baseline_df, current_df)

for col, psi in psi_scores.items():
    if psi < 0.1:
        status = "No drift"
    elif psi < 0.25:
        status = "Moderate drift"
    else:
        status = "Significant drift"
    print(f"{col}: PSI={psi:.4f} ({status})")
```

### PSI Interpretation

| PSI Value | Interpretation |
|-----------|---------------|
| < 0.10 | No significant drift |
| 0.10 - 0.25 | Moderate drift - investigate |
| > 0.25 | Significant drift - action required |

## Multi-Dataset Monitoring

```python
from genesis.drift import DriftDashboard

dashboard = DriftDashboard()

# Register multiple datasets
dashboard.add_baseline("customers", customers_baseline)
dashboard.add_baseline("transactions", transactions_baseline)

# Check against current data
results = dashboard.check_all({
    "customers": customers_current,
    "transactions": transactions_current
})

# Get summary
for name, report in results.items():
    print(f"{name}: {'DRIFT' if report.has_significant_drift else 'OK'}")
```

## Pipeline Integration

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("baseline.csv", name="baseline")
    .source("current.csv", name="current")
    .add_node("detect", "detect_drift", {
        "baseline_input": "baseline",
        "current_input": "current"
    })
    .add_node("generate", "synthesize", {
        "drift_aware": True,
        "n_samples": 10000
    })
    .sink("adapted_synthetic.csv")
    .build()
)

result = pipeline.execute()
print(f"Drift detected: {result['detect']['has_significant_drift']}")
```

## CLI Usage

```bash
# Detect drift between two files
genesis drift -b baseline.csv -c current.csv

# With detailed report
genesis drift -b baseline.csv -c current.csv -o drift_report.html --format html

# Generate drift-adapted data
genesis drift -b baseline.csv -c current.csv --generate -n 10000 -o adapted.csv
```

## Best Practices

1. **Establish baselines**: Save reference distributions when deploying models
2. **Monitor regularly**: Check for drift on a schedule appropriate to your data velocity
3. **Use appropriate thresholds**: PSI > 0.25 is a common action threshold
4. **Investigate before adapting**: Understand why drift occurred before adjusting
5. **Version your baselines**: Keep historical baselines for trend analysis

## Example: ML Model Monitoring

```python
import pandas as pd
from genesis.drift import DataDriftDetector, DriftAwareGenerator
from sklearn.ensemble import RandomForestClassifier

# Training data (baseline)
train_df = pd.read_csv("train_data.csv")
model = RandomForestClassifier()
model.fit(train_df.drop("target", axis=1), train_df["target"])

# Save baseline
baseline = train_df.drop("target", axis=1)

# Production monitoring
detector = DataDriftDetector()
generator = DriftAwareGenerator()
generator.fit(baseline)

def check_production_data(new_data):
    report = detector.detect(baseline, new_data)
    
    if report.has_significant_drift:
        print(f"⚠️ Drift detected in: {report.drifted_columns}")
        
        # Generate adapted training data
        adapted = generator.generate(
            n_samples=len(train_df),
            target_distribution=new_data,
            drift_adaptation="weighted"
        )
        
        # Suggest retraining
        return {"action": "retrain", "adapted_data": adapted}
    
    return {"action": "none"}

# Check weekly production data
for week_data in weekly_batches:
    result = check_production_data(week_data)
    if result["action"] == "retrain":
        # Retrain model with adapted synthetic data
        pass
```
