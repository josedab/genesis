# Quality SLA Contracts

Genesis provides comprehensive SLA (Service Level Agreement) contracts to ensure synthetic data meets quality, privacy, and utility requirements.

## Overview

| Component | Purpose |
|-----------|---------|
| **SLAContract** | Define quality requirements |
| **SLAValidator** | Validate data against contracts |
| **MetricCalculator** | Calculate quality metrics |
| **SLAReport** | Generate compliance reports |

## Defining SLA Contracts

Create contracts that specify data quality requirements:

```python
from genesis.sla import SLAContract, Metric, Threshold

contract = SLAContract(
    name="Production Data Quality SLA",
    version="1.0",
    metrics=[
        Metric(
            name="statistical_similarity",
            threshold=Threshold(min=0.90),
            description="KS test similarity score",
        ),
        Metric(
            name="correlation_preservation",
            threshold=Threshold(min=0.85),
            description="Correlation matrix similarity",
        ),
        Metric(
            name="null_ratio_match",
            threshold=Threshold(max=0.05),
            description="Max deviation in null ratios",
        ),
        Metric(
            name="uniqueness_preservation",
            threshold=Threshold(min=0.95),
            description="Unique value ratio preservation",
        ),
    ],
)

print(f"Contract: {contract.name}")
print(f"Metrics: {len(contract.metrics)}")
```

### Threshold Types

```python
from genesis.sla import Threshold

# Minimum value required
threshold = Threshold(min=0.90)

# Maximum value allowed
threshold = Threshold(max=0.10)

# Range (both min and max)
threshold = Threshold(min=0.80, max=0.95)

# Exact value
threshold = Threshold(exact=1.0, tolerance=0.01)
```

### Built-in Metrics

| Metric | Description | Typical Threshold |
|--------|-------------|-------------------|
| `statistical_similarity` | Overall distribution similarity | min=0.85 |
| `correlation_preservation` | Correlation matrix similarity | min=0.80 |
| `null_ratio_match` | Null ratio deviation | max=0.05 |
| `uniqueness_preservation` | Unique value preservation | min=0.90 |
| `referential_integrity` | FK constraint compliance | min=1.0 |
| `privacy_score` | Re-identification risk | max=0.05 |
| `utility_score` | ML model utility | min=0.90 |

## Validating Data

Validate synthetic data against contracts:

```python
from genesis.sla import SLAValidator

validator = SLAValidator(contract)

# Validate synthetic data
result = validator.validate(
    original_data=training_data,
    synthetic_data=synthetic_data,
)

print(f"Overall Status: {result.status}")  # PASSED or FAILED
print(f"Score: {result.score:.2f}")

for metric_result in result.metrics:
    status = "✓" if metric_result.passed else "✗"
    print(f"  {status} {metric_result.name}: {metric_result.value:.4f}")
    if not metric_result.passed:
        print(f"    Expected: {metric_result.threshold}")
        print(f"    Actual: {metric_result.value}")
```

### Validation Modes

```python
# Strict mode - fail on any violation
result = validator.validate(
    original_data=training_data,
    synthetic_data=synthetic_data,
    mode="strict",
)

# Lenient mode - allow warnings
result = validator.validate(
    original_data=training_data,
    synthetic_data=synthetic_data,
    mode="lenient",
    warning_threshold=0.95,  # Warn if within 95% of threshold
)

# Sample mode - validate on sample for speed
result = validator.validate(
    original_data=training_data,
    synthetic_data=synthetic_data,
    sample_size=10000,
)
```

### Custom Validators

```python
from genesis.sla import CustomMetric

# Define custom metric
def business_rule_compliance(original, synthetic):
    """Check that salary > 0 when employment_status = 'employed'"""
    employed = synthetic[synthetic["employment_status"] == "employed"]
    valid = (employed["salary"] > 0).mean()
    return valid

contract.add_metric(
    CustomMetric(
        name="salary_employment_rule",
        function=business_rule_compliance,
        threshold=Threshold(min=1.0),
        description="Employed records must have positive salary",
    )
)
```

## Metric Calculator

Calculate individual metrics programmatically:

```python
from genesis.sla import MetricCalculator

calculator = MetricCalculator()

# Calculate specific metrics
similarity = calculator.statistical_similarity(
    original_data=training_data,
    synthetic_data=synthetic_data,
)
print(f"Statistical similarity: {similarity:.4f}")

correlation = calculator.correlation_preservation(
    original_data=training_data,
    synthetic_data=synthetic_data,
)
print(f"Correlation preservation: {correlation:.4f}")

privacy = calculator.privacy_score(
    original_data=training_data,
    synthetic_data=synthetic_data,
    method="dcr",  # Distance to Closest Record
)
print(f"Privacy score: {privacy:.4f}")
```

### Column-Level Metrics

```python
# Per-column analysis
column_metrics = calculator.column_metrics(
    original_data=training_data,
    synthetic_data=synthetic_data,
)

for col, metrics in column_metrics.items():
    print(f"\n{col}:")
    print(f"  KS statistic: {metrics['ks_statistic']:.4f}")
    print(f"  Mean difference: {metrics['mean_diff']:.4f}")
    print(f"  Std difference: {metrics['std_diff']:.4f}")
```

## SLA Reports

Generate comprehensive compliance reports:

```python
from genesis.sla import SLAReport

report = SLAReport(
    contract=contract,
    validation_result=result,
)

# Generate report
report.generate()

# Save as HTML
report.save("sla_report.html", format="html")

# Save as PDF
report.save("sla_report.pdf", format="pdf")

# Save as JSON
report.save("sla_report.json", format="json")
```

### Report Contents

1. **Executive Summary**
   - Overall pass/fail status
   - Compliance score
   - Critical violations

2. **Metric Details**
   - Per-metric results
   - Threshold comparisons
   - Trend analysis (if historical data available)

3. **Column Analysis**
   - Per-column quality scores
   - Distribution comparisons
   - Anomaly detection

4. **Recommendations**
   - Suggested improvements
   - Parameter tuning guidance

## Continuous Monitoring

Set up continuous SLA monitoring:

```python
from genesis.sla import SLAMonitor

monitor = SLAMonitor(
    contract=contract,
    alert_channels=["email", "slack"],
    check_interval_hours=24,
)

# Register data sources
monitor.register_source(
    name="daily_synthetic",
    original_data_path="s3://bucket/original/",
    synthetic_data_path="s3://bucket/synthetic/",
)

# Start monitoring
monitor.start()

# Manual check
result = monitor.check_now("daily_synthetic")
if not result.passed:
    print(f"SLA violation: {result.violations}")
```

### Alert Configuration

```python
monitor = SLAMonitor(
    contract=contract,
    alert_config={
        "email": {
            "recipients": ["team@company.com"],
            "on_failure": True,
            "on_warning": False,
        },
        "slack": {
            "webhook_url": "https://hooks.slack.com/...",
            "channel": "#data-quality",
            "on_failure": True,
            "on_warning": True,
        },
        "pagerduty": {
            "integration_key": "...",
            "on_failure": True,
            "severity": "critical",
        },
    }
)
```

## Complete Example

```python
from genesis import SyntheticGenerator
from genesis.sla import (
    SLAContract, Metric, Threshold,
    SLAValidator, MetricCalculator, SLAReport
)

# Define comprehensive SLA contract
contract = SLAContract(
    name="Customer Data SLA",
    version="2.0",
    metrics=[
        # Statistical fidelity
        Metric(
            name="statistical_similarity",
            threshold=Threshold(min=0.90),
            weight=0.3,
        ),
        Metric(
            name="correlation_preservation",
            threshold=Threshold(min=0.85),
            weight=0.2,
        ),
        # Privacy requirements
        Metric(
            name="privacy_score",
            threshold=Threshold(max=0.05),
            weight=0.3,
        ),
        # Utility requirements
        Metric(
            name="utility_score",
            threshold=Threshold(min=0.85),
            weight=0.2,
        ),
    ],
)

# Generate synthetic data
generator = SyntheticGenerator(method="gaussian_copula")
generator.fit(training_data)
synthetic_data = generator.generate(len(training_data))

# Validate against SLA
validator = SLAValidator(contract)
result = validator.validate(
    original_data=training_data,
    synthetic_data=synthetic_data,
)

# Print results
print("=" * 50)
print(f"SLA Validation: {contract.name}")
print("=" * 50)
print(f"Status: {'PASSED ✓' if result.passed else 'FAILED ✗'}")
print(f"Overall Score: {result.score:.2%}")
print()

for metric in result.metrics:
    icon = "✓" if metric.passed else "✗"
    print(f"  [{icon}] {metric.name}")
    print(f"      Value: {metric.value:.4f}")
    print(f"      Threshold: {metric.threshold}")
    if not metric.passed:
        print(f"      VIOLATION: {metric.violation_message}")
print()

# Generate report
report = SLAReport(contract=contract, validation_result=result)
report.generate()
report.save("customer_data_sla_report.html", format="html")
print("Report saved to customer_data_sla_report.html")

# Fail pipeline if SLA not met
if not result.passed:
    raise ValueError(f"SLA validation failed: {result.violation_summary}")
```

## Configuration Reference

### SLAContract

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Contract name |
| `version` | str | Contract version |
| `metrics` | List[Metric] | Quality metrics |
| `description` | str | Contract description |
| `owner` | str | Contract owner |

### Metric

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Metric identifier |
| `threshold` | Threshold | Pass/fail threshold |
| `weight` | float | Weight in overall score |
| `description` | str | Human-readable description |
| `category` | str | Metric category |

### Threshold

| Parameter | Type | Description |
|-----------|------|-------------|
| `min` | float | Minimum acceptable value |
| `max` | float | Maximum acceptable value |
| `exact` | float | Exact required value |
| `tolerance` | float | Tolerance for exact match |

### SLAValidator

| Parameter | Type | Description |
|-----------|------|-------------|
| `contract` | SLAContract | Contract to validate against |
| `mode` | str | Validation mode (strict/lenient) |
| `sample_size` | int | Sample size for validation |
| `random_state` | int | Random seed |

### MetricCalculator

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_iterations` | int | Iterations for statistical tests |
| `significance_level` | float | Statistical significance level |
| `privacy_method` | str | Privacy calculation method |
