# ADR-0004: Rejection Sampling for Conditional Generation

## Status

Accepted

## Context

Users frequently need to generate synthetic data that satisfies specific conditions:
- "Generate 1000 customers where age > 65 AND income > 100000"
- "Generate transactions where fraud_flag = True" (rare class upsampling)
- "Generate scenarios where region IN ['US', 'CA'] AND product_type = 'premium'"

Two primary approaches exist for conditional generation:

### Approach 1: Guided Generation
Modify the generator's sampling process to directly produce samples matching conditions. This requires:
- Access to generator internals (latent space manipulation)
- Custom implementation per generator type
- Complex math for multi-condition scenarios
- Tight coupling between conditions and model architecture

### Approach 2: Rejection Sampling
Generate samples unconditionally, filter to those matching conditions, repeat until enough samples are collected. This requires:
- Only a filter function
- Works with any generator (black box)
- Simple to implement and understand
- May be inefficient for rare conditions

## Decision

We implement **rejection sampling with adaptive batch sizing** as the primary conditional generation mechanism:

```python
class ConditionalSampler:
    def sample(
        self,
        generator: BaseGenerator,
        conditions: ConditionSet,
        n_samples: int,
        max_attempts: int = 100,
    ) -> pd.DataFrame:
        collected = []
        attempts = 0
        batch_size = n_samples * 2  # Start with 2x requested
        
        while len(collected) < n_samples and attempts < max_attempts:
            # Generate batch
            candidates = generator.generate(batch_size)
            
            # Filter by conditions
            mask = conditions.evaluate(candidates)
            matches = candidates[mask]
            collected.append(matches)
            
            # Adaptive batch sizing based on match rate
            match_rate = len(matches) / batch_size
            if match_rate > 0:
                # Adjust batch size to expected need
                remaining = n_samples - sum(len(c) for c in collected)
                batch_size = int(remaining / match_rate * 1.5)
                batch_size = max(batch_size, 100)  # Minimum batch
            
            attempts += 1
        
        result = pd.concat(collected, ignore_index=True)
        return result.head(n_samples)
```

User-facing API:

```python
from genesis import SyntheticGenerator, Condition, Operator

gen = SyntheticGenerator(method='ctgan')
gen.fit(real_data)

# Simple conditions (equality implied)
synthetic = gen.generate_conditional(
    n_samples=1000,
    conditions={'gender': 'F', 'age': 30}
)

# Complex conditions with operators
synthetic = gen.generate_conditional(
    n_samples=500,
    conditions=[
        Condition('age', Operator.GE, 65),
        Condition('income', Operator.BETWEEN, (100000, 500000)),
        Condition('state', Operator.IN, ['CA', 'NY', 'TX']),
    ]
)
```

## Consequences

### Positive

- **Generator-agnostic**: works with CTGAN, TVAE, Gaussian Copula, any future generator
- **Simple implementation**: ~100 lines of code vs. ~1000+ for guided generation
- **Predictable behavior**: output always exactly matches conditions
- **Composable**: any combination of conditions works automatically
- **Debuggable**: easy to understand why samples were accepted/rejected
- **No model changes**: existing trained models work without modification

### Negative

- **Efficiency for rare conditions**: if only 0.1% of data matches, need 1000x oversampling
- **No distribution shaping**: can't smoothly "nudge" distributions, only hard filters
- **Memory pressure**: large batch generation for low match rates
- **Theoretical limits**: some conditions may be impossible for the learned distribution

### Mitigations

1. **Feasibility estimation**: before sampling, estimate match probability
   ```python
   feasibility = sampler.estimate_feasibility(conditions)
   # Returns: {'estimated_match_rate': 0.05, 'suggested_batch_size': 2000}
   ```

2. **Early termination**: fail fast if match rate is too low
   ```python
   if match_rate < 0.001:
       raise InfeasibleConditionError(
           f"Condition matches <0.1% of distribution. Consider relaxing constraints."
       )
   ```

3. **Batch size limits**: cap maximum batch size to prevent OOM
   ```python
   batch_size = min(batch_size, max_batch_size)  # Default 100,000
   ```

4. **Future option**: add guided generation for specific generators as an optimization
   ```python
   gen.generate_conditional(..., strategy='guided')  # Future
   gen.generate_conditional(..., strategy='rejection')  # Current default
   ```

## Supported Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `EQ` (=) | `age = 30` | Exact equality |
| `NE` (!=) | `status != 'cancelled'` | Not equal |
| `GT` (>) | `amount > 1000` | Greater than |
| `GE` (>=) | `age >= 18` | Greater than or equal |
| `LT` (<) | `score < 0.5` | Less than |
| `LE` (<=) | `quantity <= 100` | Less than or equal |
| `IN` | `country IN ['US', 'CA']` | Value in list |
| `NOT_IN` | `status NOT IN ['error']` | Value not in list |
| `BETWEEN` | `age BETWEEN (18, 65)` | Inclusive range |
| `LIKE` | `email LIKE '%@gmail.com'` | String pattern |
| `IS_NULL` | `middle_name IS NULL` | Null/NaN check |
| `NOT_NULL` | `email NOT NULL` | Non-null check |
