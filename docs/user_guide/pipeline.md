# Visual Pipeline Builder

Genesis provides a pipeline system for building complex data generation workflows with a fluent API.

## Overview

The pipeline builder allows you to:
- Chain multiple operations together
- Create reusable generation workflows
- Visualize and debug data flows
- Execute parallel and sequential operations

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("customers.csv")
    .transform("filter", {"condition": "age >= 18"})
    .synthesize(n_samples=10000)
    .sink("synthetic_customers.csv")
    .build()
)

result = pipeline.execute()
```

## Components

| Component | Purpose |
|-----------|---------|
| **PipelineBuilder** | Fluent API for building pipelines |
| **PipelineExecutor** | Executes pipeline graphs |
| **Node Types** | Source, transform, synthesize, evaluate, sink, etc. |

## Quick Start

### Simple Pipeline

```python
from genesis.pipeline import PipelineBuilder

# Basic: load → generate → save
pipeline = (
    PipelineBuilder()
    .source("data.csv")
    .synthesize(n_samples=1000)
    .sink("synthetic.csv")
    .build()
)

pipeline.execute()
```

### With Transformations

```python
pipeline = (
    PipelineBuilder()
    .source("raw_data.csv")
    .transform("clean", {
        "drop_na": True,
        "drop_duplicates": True
    })
    .transform("filter", {
        "condition": "amount > 0"
    })
    .synthesize(method="ctgan", n_samples=5000)
    .evaluate()
    .sink("output.csv")
    .build()
)

result = pipeline.execute()
print(f"Quality score: {result['evaluate']['overall_score']:.1%}")
```

## Node Types

### Source Nodes

Load data into the pipeline:

```python
# From CSV
pipeline.source("data.csv")

# From Parquet
pipeline.source("data.parquet")

# From DataFrame
pipeline.source(df, name="input_data")

# From database
pipeline.source_db(
    connection="postgresql://user:pass@host/db",
    query="SELECT * FROM customers"
)

# Multiple sources
pipeline.source("customers.csv", name="customers")
pipeline.source("orders.csv", name="orders")
```

### Transform Nodes

Modify data:

```python
# Built-in transforms
pipeline.transform("filter", {"condition": "age >= 18"})
pipeline.transform("clean", {"drop_na": True})
pipeline.transform("sample", {"n": 1000})
pipeline.transform("select", {"columns": ["id", "name", "email"]})

# Custom transform
def my_transform(df):
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 50, 100])
    return df

pipeline.transform("custom", {"function": my_transform})
```

### Synthesize Nodes

Generate synthetic data:

```python
# Basic synthesis
pipeline.synthesize(n_samples=10000)

# With method selection
pipeline.synthesize(method="ctgan", n_samples=10000)

# With full configuration
pipeline.synthesize(
    method="ctgan",
    n_samples=10000,
    config={
        "epochs": 300,
        "batch_size": 500,
        "privacy": {"epsilon": 1.0}
    }
)
```

### Evaluate Nodes

Assess data quality:

```python
# Evaluate synthetic vs original
pipeline.evaluate()

# With specific metrics
pipeline.evaluate(metrics=["fidelity", "utility", "privacy"])

# Save report
pipeline.evaluate(output="report.html")
```

### Sink Nodes

Save results:

```python
# To file
pipeline.sink("output.csv")
pipeline.sink("output.parquet")

# To database
pipeline.sink_db(
    connection="postgresql://user:pass@host/db",
    table="synthetic_customers"
)

# Multiple outputs
pipeline.sink("synthetic.csv", name="csv_output")
pipeline.sink("synthetic.parquet", name="parquet_output")
```

## Advanced Pipelines

### Branching

```python
from genesis.pipeline import PipelineBuilder

# Create branches
pipeline = (
    PipelineBuilder()
    .source("data.csv", name="source")
    
    # Branch 1: High privacy
    .synthesize(
        method="dpctgan",
        n_samples=5000,
        config={"epsilon": 0.1},
        input="source",
        name="high_privacy"
    )
    
    # Branch 2: High quality
    .synthesize(
        method="ctgan",
        n_samples=5000,
        input="source",
        name="high_quality"
    )
    
    # Save both
    .sink("high_privacy.csv", input="high_privacy")
    .sink("high_quality.csv", input="high_quality")
    .build()
)

pipeline.execute()
```

### Multi-Table Pipelines

```python
pipeline = (
    PipelineBuilder()
    # Load multiple tables
    .source("customers.csv", name="customers")
    .source("orders.csv", name="orders")
    
    # Join tables
    .transform("join", {
        "left": "customers",
        "right": "orders",
        "on": "customer_id"
    }, name="joined")
    
    # Generate
    .synthesize(n_samples=10000, input="joined")
    
    # Split back
    .transform("split", {
        "tables": ["customers", "orders"],
        "key": "customer_id"
    })
    
    .sink("synthetic_customers.csv", input="customers")
    .sink("synthetic_orders.csv", input="orders")
    .build()
)
```

### Conditional Execution

```python
pipeline = (
    PipelineBuilder()
    .source("data.csv")
    
    # Conditional transform
    .transform("filter", {
        "condition": "amount > 0"
    }, when=lambda ctx: ctx.config.get("filter_positive", True))
    
    .synthesize(n_samples=1000)
    .sink("output.csv")
    .build()
)

# Execute with config
pipeline.execute(config={"filter_positive": True})
```

### Iterative Pipelines

```python
from genesis.pipeline import PipelineBuilder

# Quality-driven iteration
pipeline = (
    PipelineBuilder()
    .source("data.csv")
    .synthesize(n_samples=1000)
    .evaluate(threshold=0.85)  # Repeat until quality >= 85%
    .sink("output.csv")
    .build()
)

result = pipeline.execute(max_iterations=5)
print(f"Achieved quality: {result['evaluate']['overall_score']:.1%}")
```

## Pipeline Execution

### Basic Execution

```python
result = pipeline.execute()

# Access results
print(f"Rows generated: {result['synthesize']['n_rows']}")
print(f"Quality score: {result['evaluate']['overall_score']}")
```

### With Progress

```python
from genesis.pipeline import PipelineExecutor

executor = PipelineExecutor(pipeline)

for step in executor.execute_with_progress():
    print(f"Completed: {step.name}")
    print(f"Duration: {step.duration:.2f}s")
```

### Parallel Execution

```python
# Independent nodes execute in parallel
pipeline = (
    PipelineBuilder()
    .source("data.csv", name="source")
    
    # These run in parallel
    .synthesize(method="ctgan", input="source", name="ctgan_output")
    .synthesize(method="tvae", input="source", name="tvae_output")
    .synthesize(method="gaussian_copula", input="source", name="gc_output")
    
    .build()
)

# Executes parallel nodes concurrently
result = pipeline.execute(parallel=True)
```

## Pipeline Validation

```python
# Validate before execution
pipeline = (
    PipelineBuilder()
    .source("data.csv")
    .synthesize(n_samples=1000)
    .sink("output.csv")
    .build()
)

# Check for issues
validation = pipeline.validate()

if not validation.is_valid:
    for error in validation.errors:
        print(f"Error: {error}")
else:
    pipeline.execute()
```

## Saving and Loading Pipelines

```python
# Save pipeline definition
pipeline.save("my_pipeline.yaml")

# Load and execute
from genesis.pipeline import Pipeline
pipeline = Pipeline.load("my_pipeline.yaml")
pipeline.execute()
```

### YAML Format

```yaml
# my_pipeline.yaml
name: customer_synthesis
version: "1.0"

nodes:
  - name: source
    type: source
    config:
      path: data.csv
      
  - name: clean
    type: transform
    input: source
    config:
      operation: clean
      drop_na: true
      
  - name: generate
    type: synthesize
    input: clean
    config:
      method: ctgan
      n_samples: 10000
      
  - name: output
    type: sink
    input: generate
    config:
      path: synthetic.csv
```

## CLI Usage

```bash
# Run a pipeline file
genesis pipeline run my_pipeline.yaml

# With config overrides
genesis pipeline run my_pipeline.yaml --config n_samples=5000

# Validate only
genesis pipeline validate my_pipeline.yaml

# Visualize pipeline
genesis pipeline visualize my_pipeline.yaml -o pipeline.png
```

## Convenience Functions

```python
from genesis.pipeline import create_simple_pipeline, create_evaluation_pipeline

# Quick generation pipeline
pipeline = create_simple_pipeline(
    source="data.csv",
    output="synthetic.csv",
    n_samples=10000
)
pipeline.execute()

# Evaluation pipeline
pipeline = create_evaluation_pipeline(
    real_data="real.csv",
    synthetic_data="synthetic.csv",
    report_output="report.html"
)
pipeline.execute()
```

## Best Practices

1. **Name your nodes**: Makes debugging and branching easier
2. **Validate first**: Check pipeline before long executions
3. **Use built-in transforms**: They're optimized and tested
4. **Save pipelines**: Version control your pipeline definitions
5. **Monitor progress**: Use progress callbacks for long pipelines
6. **Test incrementally**: Build and test one node at a time

## Example: Complete ML Data Pipeline

```python
from genesis.pipeline import PipelineBuilder

# Production ML data pipeline
pipeline = (
    PipelineBuilder()
    
    # Load and clean
    .source("raw_data.csv")
    .transform("clean", {
        "drop_na": True,
        "drop_duplicates": True,
        "remove_outliers": True,
        "outlier_method": "iqr"
    })
    
    # Split for privacy audit
    .transform("split", {
        "train_ratio": 0.8
    }, name="splitter")
    
    # Generate synthetic training data
    .synthesize(
        method="ctgan",
        n_samples=50000,
        input="splitter.train",
        name="synthetic"
    )
    
    # Evaluate quality
    .evaluate(
        real_input="splitter.train",
        synthetic_input="synthetic",
        output="quality_report.html"
    )
    
    # Privacy audit
    .add_node("audit", "privacy_audit", {
        "real_input": "splitter.test",
        "synthetic_input": "synthetic",
        "sensitive_columns": ["ssn", "income"]
    })
    
    # Version the result
    .add_node("version", "commit", {
        "repository": "./data_versions",
        "message": "Pipeline generated data",
        "input": "synthetic"
    })
    
    # Final output
    .sink("synthetic_training_data.csv", input="synthetic")
    .build()
)

result = pipeline.execute()

print(f"Quality: {result['evaluate']['overall_score']:.1%}")
print(f"Privacy audit: {'PASSED' if result['audit']['passed'] else 'FAILED'}")
print(f"Version: {result['version']['commit_hash']}")
```
