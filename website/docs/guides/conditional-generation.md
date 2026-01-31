---
sidebar_position: 4
title: Conditional Generation
---

# Conditional Generation

Generate synthetic data that matches specific scenarios, conditions, or target distributions.

## Quick Start

```python
from genesis import ConditionalGenerator

generator = ConditionalGenerator()
generator.fit(df)

# Generate only high-value customers
high_value = generator.generate(
    n_samples=100,
    conditions={'customer_segment': 'premium', 'lifetime_value': ('>', 10000)}
)
```

## Condition Types

### Exact Match

```python
# Category equals specific value
conditions = {'status': 'active'}

# Multiple exact matches
conditions = {'status': 'active', 'region': 'US'}
```

### Numeric Comparisons

```python
# Greater than
conditions = {'age': ('>', 18)}

# Less than or equal
conditions = {'income': ('<=', 100000)}

# Range
conditions = {'age': ('between', 25, 45)}
```

### Multiple Values

```python
# Any of these values
conditions = {'category': ('in', ['A', 'B', 'C'])}

# Not these values  
conditions = {'status': ('not_in', ['cancelled', 'expired'])}
```

### Combining Conditions

```python
conditions = {
    'age': ('>', 30),
    'income': ('between', 50000, 100000),
    'status': 'active',
    'region': ('in', ['US', 'UK', 'CA'])
}
```

## Sampling Strategies

### Rejection Sampling (Default)

Generates samples and filters:

```python
generator.generate(
    n_samples=100,
    conditions=conditions,
    strategy='rejection'  # Default
)
```

Pros: Simple, exact matches
Cons: Slow for rare conditions

### Guided Sampling

Steers generation toward conditions:

```python
generator.generate(
    n_samples=100,
    conditions=conditions,
    strategy='guided'
)
```

Pros: Faster for rare conditions
Cons: May not match exactly

### Importance Sampling

Reweights samples:

```python
generator.generate(
    n_samples=100,
    conditions=conditions,
    strategy='importance'
)
```

Pros: Maintains distribution properties
Cons: Requires more samples

## Target Distribution Matching

Match a desired distribution:

```python
# Current data is 90% class 0, 10% class 1
# Generate balanced data
generator.generate(
    n_samples=1000,
    target_distribution={'class': {0: 0.5, 1: 0.5}}
)
```

## Scenario Generation

Generate specific business scenarios:

```python
from genesis import ScenarioGenerator

generator = ScenarioGenerator()
generator.fit(df)

# Define scenarios
scenarios = {
    'high_risk_customer': {
        'credit_score': ('between', 300, 550),
        'debt_ratio': ('>', 0.7),
        'late_payments': ('>', 3)
    },
    'ideal_customer': {
        'credit_score': ('>', 750),
        'debt_ratio': ('<', 0.3),
        'late_payments': 0
    }
}

# Generate each scenario
for name, conditions in scenarios.items():
    data = generator.generate(100, conditions=conditions)
    data.to_csv(f'{name}.csv')
```

## Interpolation

Generate data between existing samples:

```python
from genesis import interpolate_samples

# Create data between two customer profiles
profile_a = df[df['segment'] == 'budget'].iloc[0]
profile_b = df[df['segment'] == 'premium'].iloc[0]

# Generate 10 samples transitioning from A to B
interpolated = interpolate_samples(
    generator, 
    profile_a, 
    profile_b, 
    n_samples=10
)
```

## Counterfactual Generation

"What if" scenarios:

```python
from genesis import CounterfactualGenerator

generator = CounterfactualGenerator()
generator.fit(df)

# Take a real sample
sample = df.iloc[0]

# Generate counterfactual: what if this customer was premium?
counterfactual = generator.generate_counterfactual(
    sample,
    changes={'customer_segment': 'premium'}
)

print("Original:", sample['lifetime_value'])
print("Counterfactual:", counterfactual['lifetime_value'])
```

## Conditional Time Series

```python
from genesis import ConditionalTimeSeriesGenerator

generator = ConditionalTimeSeriesGenerator()
generator.fit(stock_data)

# Generate bull market scenario
bull_market = generator.generate(
    n_sequences=10,
    conditions={'trend': 'up', 'volatility': 'low'}
)

# Generate crash scenario
crash = generator.generate(
    n_sequences=10,
    conditions={'trend': 'down', 'volatility': 'high'}
)
```

## Evaluation

Verify conditions are satisfied:

```python
from genesis.evaluation import ConditionalMetrics

metrics = ConditionalMetrics(synthetic_data, conditions)

print(f"Condition Satisfaction: {metrics.satisfaction_rate():.1%}")
print(f"Distribution Match: {metrics.distribution_match():.1%}")

# Per-condition breakdown
for cond, rate in metrics.per_condition_rates().items():
    print(f"  {cond}: {rate:.1%}")
```

## Complete Example

```python
import pandas as pd
from genesis import ConditionalGenerator

# Load e-commerce data
df = pd.read_csv('customers.csv')

# Create generator
generator = ConditionalGenerator(method='ctgan')
generator.fit(df, discrete_columns=['segment', 'region', 'is_churned'])

# Generate test scenarios
scenarios = {
    # For testing retention campaigns
    'at_risk_customers': {
        'days_since_last_purchase': ('>', 60),
        'total_orders': ('between', 1, 3),
        'is_churned': 0  # Not yet churned
    },
    # For testing premium features
    'high_value_segment': {
        'segment': 'premium',
        'total_spend': ('>', 5000),
        'loyalty_years': ('>', 2)
    },
    # For testing new user onboarding
    'new_users': {
        'loyalty_years': 0,
        'total_orders': 0,
        'days_since_signup': ('<', 30)
    }
}

# Generate each scenario
for name, conditions in scenarios.items():
    data = generator.generate(500, conditions=conditions)
    
    # Verify
    print(f"\n{name}:")
    print(f"  Generated: {len(data)} samples")
    
    data.to_csv(f'test_data_{name}.csv', index=False)
```

## Best Practices

1. **Start broad, then narrow** - Begin with fewer conditions
2. **Check feasibility** - Ensure conditions exist in training data
3. **Use guided strategy for rare conditions** - Rejection sampling can be slow
4. **Verify conditions post-generation** - Always check satisfaction rate
5. **Combine with constraints** - For additional data quality rules

## Troubleshooting

### "No samples found matching conditions"
- Conditions may be too restrictive
- Check if combination exists in training data
- Use guided sampling instead of rejection

### Poor distribution match
- Increase samples generated
- Use importance sampling strategy
- Relax numeric range conditions

### Slow generation
- Use guided strategy instead of rejection
- Generate more than needed and filter
- Parallelize generation

## Next Steps

- **[Augmentation](/docs/guides/augmentation)** - Balance datasets with conditional generation
- **[Pipelines](/docs/guides/pipelines)** - Chain conditional generators
- **[Constraints](/docs/concepts/constraints)** - Combine with business rules
