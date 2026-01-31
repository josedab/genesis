# Conditional Generation

Genesis provides powerful conditional generation capabilities to create synthetic data that satisfies specific constraints and conditions.

## Overview

| Class | Purpose | Use Case |
|-------|---------|----------|
| **ConditionBuilder** | Fluent API for building conditions | Simple, readable condition creation |
| **GuidedConditionalSampler** | Smart sampling with constraints | Generating rare scenarios efficiently |
| **ConditionalSampler** | Basic rejection sampling | Simple filtering needs |

## ConditionBuilder

Build complex conditions using a fluent, chainable API.

```python
from genesis.generators.conditional import ConditionBuilder

# Simple condition
conditions = (
    ConditionBuilder()
    .where("age").gte(21)
    .build()
)

# Complex multi-condition
conditions = (
    ConditionBuilder()
    .where("age").gte(21).lte(65)
    .where("income").gt(50000)
    .where("country").in_(["US", "UK", "CA"])
    .where("status").eq("active")
    .build()
)
```

### Available Operators

| Method | Operator | Example |
|--------|----------|---------|
| `.eq(value)` | Equals | `.where("status").eq("active")` |
| `.ne(value)` | Not equals | `.where("status").ne("inactive")` |
| `.gt(value)` | Greater than | `.where("age").gt(18)` |
| `.gte(value)` | Greater than or equal | `.where("age").gte(21)` |
| `.lt(value)` | Less than | `.where("score").lt(100)` |
| `.lte(value)` | Less than or equal | `.where("score").lte(100)` |
| `.in_(values)` | In list | `.where("region").in_(["N", "S"])` |
| `.not_in(values)` | Not in list | `.where("status").not_in(["deleted"])` |
| `.between(min, max)` | Range inclusive | `.where("age").between(18, 65)` |

## GuidedConditionalSampler

Efficiently generate data satisfying conditions using intelligent sampling strategies.

```python
from genesis.generators.conditional import GuidedConditionalSampler, ConditionBuilder

# Create and fit sampler
sampler = GuidedConditionalSampler(strategy="iterative_refinement")
sampler.fit(training_data)

# Define conditions
conditions = (
    ConditionBuilder()
    .where("age").gte(65)
    .where("income").gt(100000)
    .where("region").eq("West")
    .build()
)

# Generate with a base generator function
def generator_fn(n):
    return my_generator.generate(n)

result = sampler.sample(
    generator_fn=generator_fn,
    n_samples=1000,
    conditions=conditions,
)
```

### Sampling Strategies

#### Iterative Refinement (Default)
Generates samples in batches, filtering and regenerating until target count is reached.

```python
sampler = GuidedConditionalSampler(strategy="iterative_refinement")
```

**Pros:** Simple, works with any generator  
**Cons:** Inefficient for rare conditions

#### Importance Sampling
Uses condition statistics to oversample from regions likely to satisfy conditions.

```python
sampler = GuidedConditionalSampler(strategy="importance_sampling")
```

**Pros:** More efficient for rare conditions  
**Cons:** Requires good statistics from fit()

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | str | "iterative_refinement" | Sampling strategy |
| `max_iterations` | int | 100 | Maximum sampling attempts |
| `batch_multiplier` | float | 5.0 | Oversample factor per batch |

## ConditionalSampler

Basic rejection sampling for simple use cases.

```python
from genesis.generators.conditional import ConditionalSampler, ConditionSet

# From dictionary
sampler = ConditionalSampler(base_data)
result = sampler.sample(
    conditions={"age": (">=", 21), "status": "active"},
    n_samples=500
)

# Using ConditionSet
conditions = ConditionSet.from_dict({
    "age": ("between", (18, 65)),
    "income": (">", 50000),
})
result = sampler.sample(conditions=conditions, n_samples=500)
```

### Feasibility Estimation

Check if conditions are achievable before sampling:

```python
sampler = ConditionalSampler(base_data)
feasibility = sampler.estimate_feasibility(conditions)

print(f"Estimated match rate: {feasibility:.2%}")
if feasibility < 0.01:
    print("Warning: Very rare condition, consider relaxing constraints")
```

## Scenario Generator

Generate multiple scenario variations efficiently:

```python
from genesis.generators.conditional import ScenarioGenerator

generator = ScenarioGenerator(base_generator)

scenarios = [
    {"name": "young_high_income", "conditions": {"age": ("<", 30), "income": (">", 100000)}},
    {"name": "senior_retired", "conditions": {"age": (">=", 65), "employment": "retired"}},
    {"name": "family_suburban", "conditions": {"children": (">", 0), "area": "suburban"}},
]

results = generator.generate_scenarios(scenarios, samples_per_scenario=1000)
# Returns: {"young_high_income": DataFrame, "senior_retired": DataFrame, ...}
```

## Best Practices

### 1. Start with Feasibility Check
```python
feasibility = sampler.estimate_feasibility(conditions)
if feasibility < 0.001:
    # Consider relaxing conditions or using importance sampling
    sampler = GuidedConditionalSampler(strategy="importance_sampling")
```

### 2. Use Appropriate Strategy
- **Simple conditions (>10% match rate):** `ConditionalSampler`
- **Moderate conditions (1-10%):** `GuidedConditionalSampler` with iterative
- **Rare conditions (<1%):** `GuidedConditionalSampler` with importance sampling

### 3. Validate Results
```python
result = sampler.sample(conditions=conditions, n_samples=1000)

# Verify conditions are met
assert all(result["age"] >= 21)
assert all(result["status"] == "active")
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator
from genesis.generators.conditional import (
    GuidedConditionalSampler,
    ConditionBuilder,
)

# Load and train base generator
data = pd.read_csv("customers.csv")
generator = SyntheticGenerator(method="ctgan")
generator.fit(data)

# Setup guided sampler
sampler = GuidedConditionalSampler(strategy="importance_sampling")
sampler.fit(data)

# Define complex conditions
conditions = (
    ConditionBuilder()
    .where("age").between(25, 35)
    .where("annual_income").gte(75000)
    .where("credit_score").gte(720)
    .where("account_type").in_(["premium", "platinum"])
    .build()
)

# Generate targeted synthetic data
synthetic = sampler.sample(
    generator_fn=lambda n: generator.generate(n),
    n_samples=5000,
    conditions=conditions,
)

print(f"Generated {len(synthetic)} records matching conditions")
print(synthetic.describe())
```
