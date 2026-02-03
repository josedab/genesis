# Explainable Synthetic Data Generation

Genesis provides comprehensive explainability features to understand how synthetic data is generated, trace data lineage, and analyze feature importance.

## Overview

| Component | Purpose |
|-----------|---------|
| **AttributionTracker** | Track generation decisions |
| **FeatureImportanceCalculator** | Analyze feature contributions |
| **LineageTracker** | Trace data provenance |
| **ExplanationReport** | Human-readable explanations |

## Attribution Tracking

Understand why specific values were generated:

```python
from genesis import SyntheticGenerator
from genesis.explainability import AttributionTracker

# Create generator with attribution
generator = SyntheticGenerator(method="gaussian_copula")
tracker = AttributionTracker(generator)

# Fit and generate with tracking
tracker.fit(training_data)
synthetic_data, attributions = tracker.generate(1000, track=True)

# Get attribution for a specific record
record_attr = attributions.get_record_attribution(row_index=42)

print("Generation attributions for record 42:")
for column, attr in record_attr.items():
    print(f"  {column}:")
    print(f"    Value: {attr.value}")
    print(f"    Source: {attr.source}")
    print(f"    Confidence: {attr.confidence:.2f}")
    print(f"    Contributing factors: {attr.factors}")
```

### Attribution Sources

| Source | Description |
|--------|-------------|
| `correlation` | Generated based on column correlations |
| `distribution` | Sampled from learned distribution |
| `constraint` | Enforced by explicit constraint |
| `default` | Used default/fallback value |
| `conditional` | Conditioned on other columns |

### Detailed Attributions

```python
# Get column-level attribution details
attr = attributions.get_column_attribution("income")

print(f"Column: income")
print(f"Distribution type: {attr.distribution_type}")
print(f"Parameters: {attr.parameters}")
print(f"Correlation with:")
for col, corr in attr.correlations.items():
    print(f"  {col}: {corr:.3f}")
```

## Feature Importance

Analyze which features most influence the generated data:

```python
from genesis.explainability import FeatureImportanceCalculator

calculator = FeatureImportanceCalculator(generator)

# Calculate importance scores
importance = calculator.calculate(training_data)

print("Feature Importance Scores:")
for feature, score in importance.scores.items():
    print(f"  {feature}: {score:.4f}")

# Visualize importance
importance.plot()  # Requires matplotlib
```

### Importance Methods

```python
# Permutation importance (default)
importance = calculator.calculate(
    data=training_data,
    method="permutation",
    n_iterations=100,
)

# SHAP-style importance
importance = calculator.calculate(
    data=training_data,
    method="shap",
)

# Correlation-based importance
importance = calculator.calculate(
    data=training_data,
    method="correlation",
)
```

### Feature Interactions

```python
# Analyze feature interactions
interactions = calculator.calculate_interactions(training_data)

print("Top Feature Interactions:")
for pair, score in interactions.top_pairs(10):
    print(f"  {pair[0]} <-> {pair[1]}: {score:.4f}")
```

## Lineage Tracking

Track data provenance from source to synthetic output:

```python
from genesis.explainability import LineageTracker

tracker = LineageTracker()

# Register source data
source_id = tracker.register_source(
    data=training_data,
    name="customer_data_2026",
    metadata={
        "source_system": "CRM",
        "extraction_date": "2026-01-15",
        "row_count": len(training_data),
    }
)

# Track transformation
transform_id = tracker.register_transform(
    parent_id=source_id,
    transform_type="preprocessing",
    params={"normalize": True, "drop_nulls": True},
)

# Track generation
generation_id = tracker.register_generation(
    parent_id=transform_id,
    generator_type="gaussian_copula",
    params={"epochs": 100},
    output_rows=10000,
)

# Get full lineage
lineage = tracker.get_lineage(generation_id)

print("Data Lineage:")
for step in lineage.steps:
    print(f"  {step.type}: {step.name}")
    print(f"    ID: {step.id}")
    print(f"    Timestamp: {step.timestamp}")
    print(f"    Metadata: {step.metadata}")
```

### Lineage Visualization

```python
# Export lineage as graph
lineage.to_graph("lineage.png")  # Visual diagram

# Export as JSON
lineage_json = lineage.to_json()

# Export for data catalog integration
lineage.to_openlineage()  # OpenLineage format
```

### Query Lineage

```python
# Find all descendants of a source
descendants = tracker.get_descendants(source_id)

# Find ancestors of synthetic data
ancestors = tracker.get_ancestors(generation_id)

# Find by metadata
results = tracker.search(
    source_system="CRM",
    date_range=("2026-01-01", "2026-01-31")
)
```

## Explanation Reports

Generate human-readable explanations:

```python
from genesis.explainability import ExplanationReport

report = ExplanationReport(
    generator=generator,
    training_data=training_data,
    synthetic_data=synthetic_data,
)

# Generate full report
report.generate()

# Save as HTML
report.save("explanation_report.html", format="html")

# Save as Markdown
report.save("explanation_report.md", format="markdown")

# Save as JSON
report.save("explanation_report.json", format="json")
```

### Report Contents

The explanation report includes:

1. **Data Overview**
   - Source data statistics
   - Synthetic data statistics
   - Column-by-column comparison

2. **Generation Method**
   - Algorithm description
   - Hyperparameters used
   - Training metrics

3. **Statistical Fidelity**
   - Distribution comparisons
   - Correlation preservation
   - Marginal distributions

4. **Feature Analysis**
   - Importance scores
   - Interaction analysis
   - Dependency graphs

5. **Quality Metrics**
   - Privacy metrics
   - Utility metrics
   - Custom metrics

### Custom Report Sections

```python
report = ExplanationReport(
    generator=generator,
    training_data=training_data,
    synthetic_data=synthetic_data,
    sections=[
        "data_overview",
        "feature_importance",
        "correlations",
        "privacy_analysis",
    ],
    custom_metrics={
        "business_rule_compliance": compliance_score,
    }
)
```

## Complete Example

```python
from genesis import SyntheticGenerator
from genesis.explainability import (
    AttributionTracker,
    FeatureImportanceCalculator,
    LineageTracker,
    ExplanationReport,
)

# Load training data
training_data = pd.read_csv("customers.csv")

# Setup lineage tracking
lineage = LineageTracker()
source_id = lineage.register_source(
    data=training_data,
    name="customers_source",
)

# Create generator with attribution
generator = SyntheticGenerator(method="gaussian_copula")
tracker = AttributionTracker(generator)

# Fit and generate
tracker.fit(training_data)
synthetic_data, attributions = tracker.generate(10000, track=True)

# Register generation in lineage
gen_id = lineage.register_generation(
    parent_id=source_id,
    generator_type="gaussian_copula",
    output_rows=10000,
)

# Calculate feature importance
importance_calc = FeatureImportanceCalculator(generator)
importance = importance_calc.calculate(training_data)

print("\n=== Feature Importance ===")
for feature, score in importance.scores.items():
    print(f"{feature}: {score:.4f}")

# Generate attribution for sample record
print("\n=== Sample Record Attribution ===")
attr = attributions.get_record_attribution(0)
for col, info in attr.items():
    print(f"{col}: {info.value} (source: {info.source})")

# Generate explanation report
print("\n=== Generating Report ===")
report = ExplanationReport(
    generator=generator,
    training_data=training_data,
    synthetic_data=synthetic_data,
)
report.generate()
report.save("synthetic_data_explanation.html", format="html")
print("Report saved to synthetic_data_explanation.html")

# Export lineage
lineage.get_lineage(gen_id).to_json("lineage.json")
print("Lineage saved to lineage.json")
```

## Configuration Reference

### AttributionTracker

| Parameter | Type | Description |
|-----------|------|-------------|
| `generator` | BaseGenerator | Generator to track |
| `track_correlations` | bool | Track correlation-based attributions |
| `track_constraints` | bool | Track constraint enforcement |
| `confidence_threshold` | float | Min confidence for attributions |

### FeatureImportanceCalculator

| Parameter | Type | Description |
|-----------|------|-------------|
| `generator` | BaseGenerator | Fitted generator |
| `method` | str | Calculation method |
| `n_iterations` | int | Iterations for permutation |
| `random_state` | int | Random seed |

### LineageTracker

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage_backend` | str | Storage backend (memory, file, database) |
| `auto_timestamp` | bool | Auto-add timestamps |
| `track_checksums` | bool | Track data checksums |

### ExplanationReport

| Parameter | Type | Description |
|-----------|------|-------------|
| `generator` | BaseGenerator | Generator used |
| `training_data` | DataFrame | Original data |
| `synthetic_data` | DataFrame | Generated data |
| `sections` | List[str] | Report sections to include |
| `custom_metrics` | dict | Additional metrics |
