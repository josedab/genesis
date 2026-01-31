# ADR 003: Declarative Constraint System

## Status

Accepted

## Context

Synthetic data often needs to satisfy business rules and domain constraints:
- Age must be positive
- Salary must be within a range
- IDs must be unique
- Date ranges must be valid

Options for implementing constraints:

1. **Post-processing**: Generate freely, then clip/transform
2. **Hard constraints in generator**: Modify loss function to penalize violations
3. **Declarative constraints**: User specifies constraints, system applies them

## Decision

Implement a **declarative constraint system** with a factory pattern:

```python
from genesis import Constraint

constraints = [
    Constraint.positive('age'),
    Constraint.range('salary', 30000, 200000),
    Constraint.unique('employee_id'),
    Constraint.one_of('status', ['active', 'inactive']),
]

generator.fit(data, constraints=constraints)
synthetic = generator.generate(n_samples=1000)
# All constraints are automatically enforced
```

### Constraint Types

| Type | Description | Example |
|------|-------------|---------|
| `positive` | Value > 0 | `Constraint.positive('price')` |
| `range` | min ≤ value ≤ max | `Constraint.range('age', 0, 120)` |
| `unique` | All values distinct | `Constraint.unique('id')` |
| `one_of` | Value in allowed set | `Constraint.one_of('color', ['red', 'blue'])` |
| `regex` | Matches pattern | `Constraint.regex('email', r'.+@.+')` |
| `custom` | User-defined function | `Constraint.custom('col', lambda x: x > 0)` |

### Implementation Strategy

1. Constraints are validated after generation
2. Invalid rows are transformed (clipped, regenerated, etc.)
3. Transformation method is configurable per constraint
4. Validation can be run separately via `constraint.validate(df)`

## Consequences

### Positive

- Intuitive, readable API
- Separation of concerns (generation vs. validation)
- Extensible via custom constraints
- Constraints are reusable and composable
- Can validate both real and synthetic data

### Negative

- Post-hoc constraint enforcement may distort distributions
- Some constraints (like uniqueness for continuous) can be expensive
- Order of constraint application may matter

### Neutral

- Trade-off between constraint strictness and distribution fidelity
- Users can choose transform method (clip vs. regenerate vs. reject)
