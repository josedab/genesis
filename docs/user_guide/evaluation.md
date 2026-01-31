# Quality Evaluation

Evaluate the quality of synthetic data.

## Quick Evaluation

```python
from genesis import QualityEvaluator

evaluator = QualityEvaluator(real_data, synthetic_data)
report = evaluator.evaluate()
print(report.summary())
```

## Metrics

### Statistical Fidelity

Measures how well distributions match:

- **KS Test**: Kolmogorov-Smirnov test for continuous columns
- **Chi-squared**: Chi-squared test for categorical columns
- **Correlation**: Correlation matrix similarity
- **Wasserstein**: Earth mover's distance

### ML Utility

Measures usefulness for machine learning:

- **TSTR**: Train-Synthetic, Test-Real accuracy
- **TRTS**: Train-Real, Test-Synthetic accuracy
- **Feature Importance**: Importance similarity

```python
report = evaluator.evaluate(target_column='label')
print(f"ML Utility: {report.utility_score:.2f}")
```

### Privacy Metrics

Measures privacy protection:

- **DCR**: Distance to Closest Record
- **Re-identification Risk**: Probability of matching
- **Attribute Disclosure**: Risk of revealing attributes

## Detailed Report

```python
# Column-by-column analysis
for col, metrics in report.column_metrics.items():
    print(f"{col}: {metrics['score']:.2f}")
```

## Export Reports

```python
# HTML report
report.save_html('quality_report.html')

# JSON report
report.save_json('quality_report.json')

# Dictionary
report_dict = report.to_dict()
```

## Custom Evaluation

```python
from genesis.evaluation.statistical import kolmogorov_smirnov_test

# Test specific column
result = kolmogorov_smirnov_test(
    real_data['income'], 
    synthetic_data['income']
)
print(f"KS Score: {result['score']:.3f}")
```
