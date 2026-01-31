---
sidebar_position: 11
title: Pipelines
---

# Pipeline Builder

Create reproducible, reusable data generation workflows with the Pipeline API.

## Quick Start

```python
from genesis import Pipeline
from genesis.pipeline import steps

# Define pipeline
pipeline = Pipeline([
    steps.load_csv('customers.csv'),
    steps.fit_generator('ctgan'),
    steps.generate(1000),
    steps.evaluate(),
    steps.save_csv('synthetic.csv')
])

# Run
result = pipeline.run()
print(f"Quality: {result['evaluate']['quality_score']:.1%}")
```

## Pipeline Concepts

```mermaid
flowchart LR
    A[Load] --> B[Transform]
    B --> C[Fit Generator]
    C --> D[Generate]
    D --> E[Validate]
    E --> F[Save]
```

Pipelines:
- Chain multiple steps together
- Pass data between steps automatically
- Track execution and metrics
- Enable reproducibility

## Built-in Steps

### Data Loading

```python
from genesis.pipeline import steps

# From CSV
steps.load_csv('data.csv')

# From Parquet
steps.load_parquet('data.parquet')

# From DataFrame
steps.load_dataframe(df)

# From database
steps.load_sql('SELECT * FROM customers', connection_string)
```

### Data Transformation

```python
# Select columns
steps.select_columns(['name', 'age', 'city'])

# Drop columns
steps.drop_columns(['id', 'created_at'])

# Filter rows
steps.filter_rows(lambda df: df['age'] > 18)

# Type conversion
steps.convert_types({'age': 'int', 'score': 'float'})

# Custom transform
steps.transform(lambda df: df.dropna())
```

### Generator Configuration

```python
# Fit a generator
steps.fit_generator(
    method='ctgan',
    discrete_columns=['category', 'status'],
    config={'epochs': 300}
)

# AutoML
steps.auto_fit()

# Load pre-trained generator
steps.load_generator('trained_model.pkl')
```

### Generation

```python
# Generate samples
steps.generate(n_samples=1000)

# Conditional generation
steps.generate(
    n_samples=1000,
    conditions={'status': 'active'}
)

# Generate to match original size
steps.generate_matching_size()
```

### Validation

```python
# Quality evaluation
steps.evaluate(target_column='churn')

# Privacy audit
steps.privacy_audit(
    sensitive_columns=['income'],
    threshold=0.9
)

# Drift detection
steps.check_drift(baseline='baseline.csv', threshold=0.1)

# Schema validation
steps.validate_schema(expected_schema)
```

### Output

```python
# Save to file
steps.save_csv('output.csv')
steps.save_parquet('output.parquet')

# Save to database
steps.save_sql('synthetic_customers', connection_string)

# Version and save
steps.save_versioned(repo='./datasets', tag='v1.0')
```

## Conditional Steps

```python
from genesis.pipeline import Pipeline, steps, conditions

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    steps.fit_generator('ctgan'),
    steps.generate(1000),
    steps.evaluate(),
    
    # Only save if quality is good
    conditions.if_metric(
        'quality_score', '>', 0.8,
        then=steps.save_csv('good_quality.csv'),
        else_=steps.log_warning("Quality below threshold")
    )
])
```

## Branching Pipelines

```python
from genesis.pipeline import Pipeline, Branch

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    steps.fit_generator('ctgan'),
    
    # Run multiple branches in parallel
    Branch([
        # Branch 1: Small sample
        [
            steps.generate(100),
            steps.save_csv('sample_100.csv')
        ],
        # Branch 2: Large sample
        [
            steps.generate(10000),
            steps.save_csv('sample_10000.csv')
        ]
    ])
])
```

## Error Handling

```python
from genesis.pipeline import Pipeline, steps

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    steps.fit_generator('ctgan'),
    steps.generate(1000),
    steps.evaluate(),
    steps.save_csv('output.csv')
], 
    on_error='continue',  # 'stop' or 'continue'
    retry_failed=3,
    notify_on_failure='email@example.com'
)
```

## Pipeline Configuration

### From YAML

```yaml
# pipeline.yaml
name: customer_synthesis
steps:
  - load_csv: customers.csv
  - fit_generator:
      method: ctgan
      discrete_columns: [status, region]
      config:
        epochs: 300
  - generate:
      n_samples: 10000
  - evaluate:
      target_column: churn
  - privacy_audit:
      sensitive_columns: [income, credit_score]
      threshold: 0.9
  - save_csv: synthetic_customers.csv
```

```python
from genesis.pipeline import Pipeline

pipeline = Pipeline.from_yaml('pipeline.yaml')
result = pipeline.run()
```

### Save and Load

```python
# Save pipeline definition
pipeline.save('my_pipeline.pkl')

# Load and run
loaded = Pipeline.load('my_pipeline.pkl')
result = loaded.run()
```

## Execution Tracking

```python
result = pipeline.run()

# Execution summary
print(result.summary())

# Per-step results
for step_name, step_result in result.steps.items():
    print(f"{step_name}:")
    print(f"  Status: {step_result.status}")
    print(f"  Duration: {step_result.duration:.2f}s")
    print(f"  Output shape: {step_result.output_shape}")

# Metrics
print(f"Total time: {result.total_duration:.2f}s")
print(f"Quality score: {result.metrics.get('quality_score')}")
```

## Custom Steps

```python
from genesis.pipeline import Step

class MyCustomStep(Step):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def run(self, context):
        # Access input data
        df = context.data
        
        # Process
        result = df[df['value'] > self.param1].head(self.param2)
        
        # Update context
        context.data = result
        context.metrics['custom_metric'] = len(result)
        
        return context

# Use in pipeline
pipeline = Pipeline([
    steps.load_csv('data.csv'),
    MyCustomStep(param1=100, param2=500),
    steps.save_csv('output.csv')
])
```

## Complete Example

```python
from genesis.pipeline import Pipeline, steps

# Production-ready pipeline
pipeline = Pipeline(
    name='customer_synthesis_v2',
    steps=[
        # Load and prepare
        steps.load_csv('raw_customers.csv'),
        steps.drop_columns(['internal_id', 'created_at']),
        steps.convert_types({
            'age': 'int',
            'income': 'float'
        }),
        
        # Train generator
        steps.fit_generator(
            method='ctgan',
            discrete_columns=['region', 'status', 'segment'],
            config={'epochs': 300, 'batch_size': 500}
        ),
        
        # Generate synthetic data
        steps.generate(n_samples=50000),
        
        # Validate quality
        steps.evaluate(target_column='churn'),
        steps.privacy_audit(
            sensitive_columns=['income', 'credit_score'],
            threshold=0.9
        ),
        
        # Save results
        steps.save_versioned(
            repo='./synthetic_repo',
            message='Production run',
            tag='latest'
        )
    ],
    on_error='stop',
    log_level='info'
)

# Run pipeline
result = pipeline.run()

# Check results
if result.success:
    print(f"✅ Pipeline completed successfully")
    print(f"Quality: {result.metrics['quality_score']:.1%}")
    print(f"Privacy: {result.metrics['privacy_score']:.1%}")
    print(f"Version: {result.outputs['version_id']}")
else:
    print(f"❌ Pipeline failed at step: {result.failed_step}")
    print(f"Error: {result.error}")
```

## CLI Usage

```bash
# Run pipeline from YAML
genesis pipeline run pipeline.yaml

# Run with overrides
genesis pipeline run pipeline.yaml \
  --set steps.generate.n_samples=5000 \
  --set output=custom_output.csv

# Validate pipeline without running
genesis pipeline validate pipeline.yaml

# Show pipeline graph
genesis pipeline visualize pipeline.yaml
```

## Best Practices

1. **Start simple** - Build incrementally, test each step
2. **Use YAML for production** - Version control friendly
3. **Add validation steps** - Catch issues early
4. **Log metrics** - Track quality over time
5. **Handle errors gracefully** - Define fallback behavior

## Troubleshooting

### Step fails silently
- Check `log_level='debug'`
- Review `result.steps[step_name].error`

### Pipeline is slow
- Profile with `result.timing_breakdown()`
- Optimize slowest steps

### Memory issues
- Add `steps.checkpoint()` to save intermediate results
- Use batched generation

## Next Steps

- **[Versioning](/docs/guides/versioning)** - Track pipeline outputs
- **[AutoML](/docs/guides/automl)** - Automatic method selection
- **[CLI Reference](/docs/api/cli)** - Pipeline commands
