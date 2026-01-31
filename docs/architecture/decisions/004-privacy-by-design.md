# ADR 004: Privacy-by-Design Architecture

## Status

Accepted

## Context

Privacy is a core requirement for synthetic data generation. Users need:
- Formal privacy guarantees (differential privacy)
- Protection against re-identification
- Compliance with regulations (GDPR, HIPAA)
- Configurable privacy levels

Privacy must be integrated throughout the system, not bolted on.

## Decision

Implement **privacy-by-design** with a layered approach:

### 1. Privacy Configuration

```python
from genesis import PrivacyConfig

privacy = PrivacyConfig(
    enable_differential_privacy=True,
    epsilon=1.0,
    delta=1e-5,
    k_anonymity=5,
    suppress_rare_categories=True,
)

generator = SyntheticGenerator(privacy=privacy)
```

### 2. Privacy Levels (Presets)

| Level | Epsilon | K-Anonymity | Rare Suppression |
|-------|---------|-------------|------------------|
| `low` | 10.0 | None | No |
| `medium` | 1.0 | 5 | Yes |
| `high` | 0.1 | 10 | Yes |

### 3. Privacy Metrics in Evaluation

```python
report = evaluator.evaluate()
print(report.privacy_metrics)
# - Distance to Closest Record (DCR)
# - Re-identification Risk
# - Attribute Disclosure Risk
```

### 4. Implementation Components

```
genesis/privacy/
├── config.py       # PrivacyConfig dataclass
├── differential.py # DP-SGD implementation
├── anonymity.py    # K-anonymity, L-diversity
└── metrics.py      # Privacy risk metrics
```

### Key Design Principles

1. **Privacy is opt-in but easy**: Default is no DP, but enabling it is one flag
2. **Transparency**: Users see privacy budget consumption
3. **Layered protection**: DP + k-anonymity + rare suppression can combine
4. **Measurable**: Privacy metrics are always computed in evaluation

## Consequences

### Positive

- Privacy is a first-class concern
- Formal DP guarantees when enabled
- Flexible configuration for different use cases
- Privacy metrics included in standard evaluation

### Negative

- DP reduces data utility
- Computational overhead for privacy accounting
- Users must understand privacy-utility trade-offs

### Neutral

- Default is no DP (maximum utility, user-accepted risk)
- Privacy presets simplify common configurations
