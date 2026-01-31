# ADR-0006: Privacy as a First-Class Concern

## Status

Accepted

## Context

Synthetic data is often generated from sensitive sources: healthcare records, financial transactions, customer data. Organizations use synthetic data specifically because they need privacy protection. Yet many synthetic data tools treat privacy as an afterthought:

- Privacy features buried in documentation
- Differential privacy available but not integrated
- No privacy metrics in quality reports
- Users must manually implement k-anonymity
- No guidance on quasi-identifier detection

Privacy failures in synthetic data can cause:
- Regulatory violations (GDPR, HIPAA, CCPA)
- Reputational damage from data breaches
- Re-identification of individuals from "anonymized" data
- Membership inference attacks revealing training set membership

Genesis targets privacy-conscious organizations. Privacy must be central, not optional.

## Decision

We make **privacy a first-class concern** throughout the architecture:

### 1. Privacy Configuration at Generator Level

```python
from genesis import SyntheticGenerator, PrivacyConfig, PrivacyLevel

# Preset privacy levels
privacy = PrivacyConfig.from_level(PrivacyLevel.HIGH)

# Or explicit configuration
privacy = PrivacyConfig(
    epsilon=0.5,           # Differential privacy budget
    delta=1e-5,            # DP delta parameter
    k_anonymity=5,         # Minimum k-anonymity
    l_diversity=3,         # L-diversity for sensitive attrs
    sensitive_columns=['income', 'diagnosis'],
)

gen = SyntheticGenerator(method='ctgan', privacy=privacy)
```

### 2. Privacy Metrics in Quality Reports

```python
report = gen.quality_report()

print(report.privacy_metrics)
# PrivacyMetrics(
#     reidentification_risk=0.02,
#     attribute_disclosure_risk=0.05,
#     membership_inference_risk=0.51,
#     distance_to_closest_record=0.15,
#     k_anonymity_level=12,
#     epsilon_spent=0.5,
# )
```

### 3. Automatic Quasi-Identifier Detection

```python
from genesis.analyzers import PrivacyAnalyzer

analyzer = PrivacyAnalyzer()
risk = analyzer.analyze(data)

print(risk.quasi_identifiers)
# ['age', 'zipcode', 'gender']  # Auto-detected

print(risk.reidentification_risk)
# 0.15  # 15% of records uniquely identifiable
```

### 4. Privacy-Preserving Defaults

The default `PrivacyLevel.NONE` makes privacy explicit—users must opt-in to privacy guarantees. But we provide convenient presets:

| Level | Epsilon | K-Anonymity | Use Case |
|-------|---------|-------------|----------|
| NONE | ∞ | None | Development, testing |
| LOW | 10.0 | None | Internal analytics |
| MEDIUM | 1.0 | None | Standard privacy |
| HIGH | 0.1 | 5 | Sensitive data |
| MAXIMUM | 0.01 | 10 | Highly regulated |

### 5. Differential Privacy Integration

```python
from genesis.privacy import DPAccountant

accountant = DPAccountant(epsilon=1.0, delta=1e-5)

gen = SyntheticGenerator(
    method='ctgan',
    privacy=PrivacyConfig(epsilon=1.0),
)

gen.fit(data)

print(f"Privacy budget spent: {accountant.spent_budget}")
print(f"Remaining budget: {accountant.remaining_budget}")
```

## Consequences

### Positive

- **Privacy by design**: impossible to accidentally ignore privacy
- **Compliance support**: built-in metrics for GDPR/HIPAA audits
- **User education**: preset levels teach privacy concepts
- **Audit trail**: privacy parameters recorded in data lineage
- **Defense in depth**: multiple privacy mechanisms (DP, k-anon, etc.)

### Negative

- **Performance overhead**: DP training is slower (noise injection, clipping)
- **Utility tradeoff**: stronger privacy = lower data quality
- **Complexity**: more parameters for users to understand
- **False confidence**: users may over-trust privacy guarantees

### Mitigations

- Clear documentation on privacy-utility tradeoffs
- `quality_report()` shows both utility AND privacy metrics together
- Warnings when privacy settings may be insufficient:
  ```
  Warning: k_anonymity=2 provides weak protection. Consider k>=5.
  ```
- Default is NONE so users consciously enable privacy

## Privacy Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| `reidentification_risk` | Probability of uniquely identifying a record | < 0.05 |
| `attribute_disclosure_risk` | Risk of inferring sensitive attributes | < 0.10 |
| `membership_inference_risk` | Can attacker tell if record was in training set? | < 0.55 |
| `distance_to_closest_record` | Minimum distance to any training record | > 0.10 |
| `k_anonymity_level` | Actual k achieved (records per equivalence class) | ≥ 5 |
| `epsilon_spent` | Differential privacy budget consumed | ≤ 1.0 |

## Privacy Module Structure

```
genesis/privacy/
├── __init__.py
├── config.py           # PrivacyConfig, PrivacyLevel
├── differential.py     # DPAccountant, noise mechanisms
├── anonymity.py        # K-anonymity, l-diversity enforcement
└── metrics.py          # Privacy metric calculations
```
