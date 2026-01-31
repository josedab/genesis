# ADR-0019: Constraint System with Post-Generation Validation

## Status

Accepted

## Context

Synthetic data must often satisfy business rules:

- Ages must be positive integers
- Dates must be in valid ranges
- IDs must be unique
- Columns must satisfy conditional relationships (e.g., `end_date > start_date`)

Two architectural approaches exist:

**A. Built-in constraints** (during generation):
- Constraints encoded in the model architecture
- Generator only produces valid samples
- Complex implementation, method-specific
- Limited constraint types

**B. Post-generation validation** (after generation):
- Generator produces unconstrained samples
- Separate validation/transformation step
- Simpler implementation, method-agnostic
- Flexible constraint types, may require rejection sampling

We needed to balance realism (models that understand constraints) with flexibility (arbitrary business rules).

## Decision

We implement a **post-generation constraint system** that validates and optionally transforms data after generation:

```python
from genesis import SyntheticGenerator, Constraint

generator = SyntheticGenerator(method='ctgan')

generator.fit(
    data=real_df,
    discrete_columns=['gender', 'city'],
    constraints=[
        Constraint.positive('age'),
        Constraint.range('age', 0, 120),
        Constraint.unique('customer_id'),
        Constraint.greater_than('end_date', 'start_date'),
    ]
)

# Constraints applied during generate()
synthetic = generator.generate(
    n_samples=10000,
    apply_constraints=True  # Default
)
```

### Constraint Architecture

```python
class BaseConstraint(ABC):
    """Base class for all constraints."""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> pd.Series:
        """Return boolean Series indicating valid rows."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data to satisfy constraint."""
        pass


class ConstraintSet:
    """Collection of constraints with validation/transformation."""
    
    def validate_and_transform(
        self, 
        data: pd.DataFrame,
        max_retries: int = 3
    ) -> Tuple[pd.DataFrame, ConstraintReport]:
        """Apply all constraints, regenerating invalid rows if needed."""
        pass
```

### Constraint Application Flow

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Generator  │────▶│  Raw Synthetic  │────▶│ ConstraintSet   │
│  .generate()│     │     Data        │     │ .validate_and_  │
└─────────────┘     └─────────────────┘     │  transform()    │
                                            └────────┬────────┘
                                                     │
                         ┌───────────────────────────┼───────────────────────────┐
                         ▼                           ▼                           ▼
                  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
                  │   Validate  │           │  Transform  │           │  Rejection  │
                  │  (boolean)  │           │  (fix data) │           │  Sampling   │
                  └─────────────┘           └─────────────┘           └─────────────┘
                         │                           │                           │
                         └───────────────────────────┴───────────────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │    Valid        │
                                            │  Synthetic Data │
                                            └─────────────────┘
```

## Consequences

### Positive

- **Method-agnostic**: Same constraints work with CTGAN, TVAE, Gaussian Copula
- **Flexible**: Any Python-expressible constraint can be implemented
- **Composable**: Constraints combine via `ConstraintSet`
- **Debuggable**: Clear separation between generation and validation
- **Recoverable**: Transform step can fix minor violations

### Negative

- **Distribution shift**: Post-hoc fixes may distort learned distributions
- **Rejection overhead**: Strict constraints may require many regeneration cycles
- **Not model-aware**: Generator doesn't learn constraints, just produces candidates
- **Potential infinite loops**: Impossible constraints (e.g., `age < 0 AND age > 100`) never satisfy

### Mitigations

- `max_retries` parameter prevents infinite loops
- `ConstraintReport` documents how many rows were fixed/rejected
- Warning logged when rejection rate exceeds threshold
- Soft constraints (transform) preferred over hard constraints (reject)

## Built-in Constraint Types

| Constraint | Description | Strategy |
|------------|-------------|----------|
| `positive(col)` | Values > 0 | Transform: `abs()` |
| `non_negative(col)` | Values >= 0 | Transform: `max(0, x)` |
| `range(col, min, max)` | Values in range | Transform: `clip()` |
| `unique(col)` | No duplicates | Reject + regenerate |
| `not_null(col)` | No missing values | Transform: impute |
| `greater_than(col1, col2)` | col1 > col2 | Transform: swap if needed |
| `one_hot(cols)` | Exactly one True | Transform: argmax |
| `sum_to(cols, target)` | Sum equals target | Transform: normalize |
| `custom(func)` | User-defined | User-defined |

## Examples

```python
# Basic constraints
from genesis import Constraint, ConstraintSet

constraints = [
    Constraint.positive('amount'),
    Constraint.range('age', 18, 100),
    Constraint.unique('transaction_id'),
]

# Date ordering
constraints.append(
    Constraint.greater_than('delivery_date', 'order_date')
)

# Custom constraint
@Constraint.custom
def valid_email(df):
    return df['email'].str.contains('@')

constraints.append(valid_email)

# Apply to generator
gen = SyntheticGenerator(method='ctgan')
gen.fit(data, constraints=constraints)
synthetic = gen.generate(10000)

# Constraint report
report = gen.last_constraint_report
print(f"Rows transformed: {report.transformed_count}")
print(f"Rows rejected: {report.rejected_count}")
print(f"Regeneration cycles: {report.cycles}")

# Validate without transforming
constraint_set = ConstraintSet(constraints)
valid_mask = constraint_set.validate(synthetic)
print(f"Valid rows: {valid_mask.sum()} / {len(synthetic)}")
```

## Rejection Sampling Details

When constraints cannot be satisfied via transformation:

```python
def _rejection_sample(
    self,
    generator_fn: Callable,
    n_samples: int,
    constraints: ConstraintSet,
    max_cycles: int = 10,
) -> pd.DataFrame:
    """Generate samples satisfying constraints via rejection."""
    
    collected = []
    n_remaining = n_samples
    
    for cycle in range(max_cycles):
        # Over-generate to account for rejection
        batch_size = int(n_remaining * 1.5)
        candidates = generator_fn(batch_size)
        
        # Keep valid rows
        valid_mask = constraints.validate(candidates)
        valid_rows = candidates[valid_mask]
        collected.append(valid_rows)
        
        n_remaining = n_samples - sum(len(c) for c in collected)
        if n_remaining <= 0:
            break
    
    result = pd.concat(collected).head(n_samples)
    return result
```
