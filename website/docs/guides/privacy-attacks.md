---
sidebar_position: 8
title: Privacy Attacks
---

# Privacy Attack Testing

Verify the privacy guarantees of your synthetic data by simulating adversarial attacks.

## Why Test Privacy?

Synthetic data isn't automatically private. Without proper safeguards:
- Outliers can leak directly
- Unique records may be memorized
- Attackers can infer membership in original data

Genesis provides tools to quantify these risks.

## Quick Start

```python
from genesis import run_privacy_audit
import pandas as pd

# Load original and synthetic data
real = pd.read_csv('customers.csv')
synthetic = pd.read_csv('synthetic_customers.csv')

# Run comprehensive privacy audit
report = run_privacy_audit(real, synthetic)

print(f"Privacy Score: {report.overall_score:.1%}")
print(f"Safe to release: {report.is_safe}")
```

## Attack Types

### Membership Inference

Can an attacker determine if a record was in the training data?

```python
from genesis.privacy_attacks import MembershipInferenceAttack

attack = MembershipInferenceAttack()
result = attack.evaluate(real, synthetic)

print(f"Attack accuracy: {result.accuracy:.1%}")
print(f"Risk level: {result.risk_level}")  # Low/Medium/High/Critical
```

Attack accuracy interpretations:
- **< 55%**: Safe (near random chance)
- **55-65%**: Low risk
- **65-80%**: Medium risk
- **> 80%**: High risk (potential memorization)

### Attribute Inference

Can an attacker infer sensitive attributes from other features?

```python
from genesis.privacy_attacks import AttributeInferenceAttack

attack = AttributeInferenceAttack(
    sensitive_column='income',
    known_columns=['age', 'education', 'occupation']
)
result = attack.evaluate(real, synthetic)

print(f"Inference accuracy: {result.accuracy:.1%}")
```

### Singling Out

Can unique individuals be identified?

```python
from genesis.privacy_attacks import SinglingOutAttack

attack = SinglingOutAttack(
    quasi_identifiers=['zip_code', 'age', 'gender']
)
result = attack.evaluate(real, synthetic)

print(f"Singling out risk: {result.risk:.1%}")
print(f"Unique records exposed: {result.n_unique}")
```

### Linkage Attack

Can synthetic records be linked to real individuals?

```python
from genesis.privacy_attacks import LinkageAttack

attack = LinkageAttack(
    linking_columns=['name', 'date_of_birth', 'zip_code']
)
result = attack.evaluate(real, synthetic)

print(f"Linkage success rate: {result.success_rate:.1%}")
```

## Comprehensive Audit

Run all attacks at once:

```python
from genesis import run_privacy_audit

report = run_privacy_audit(
    real,
    synthetic,
    sensitive_columns=['income', 'health_status'],
    quasi_identifiers=['age', 'gender', 'zip_code']
)

# Summary
print(report.summary())

# Individual attack results
for attack_name, result in report.attack_results.items():
    print(f"{attack_name}: {result.risk_level}")

# Recommendations
for rec in report.recommendations:
    print(f"- {rec}")
```

## Privacy Score

The overall privacy score combines multiple factors:

```python
report = run_privacy_audit(real, synthetic)

print(f"Overall: {report.overall_score:.1%}")
print(f"  Membership: {report.membership_score:.1%}")
print(f"  Attribute: {report.attribute_score:.1%}")
print(f"  Singling Out: {report.singling_out_score:.1%}")
print(f"  Linkage: {report.linkage_score:.1%}")
```

Score thresholds:
- **≥ 90%**: Excellent privacy
- **80-90%**: Good privacy
- **70-80%**: Acceptable with caveats
- **< 70%**: Privacy concerns, review needed

## Generating Private Data

If privacy tests fail, strengthen privacy protections:

### Differential Privacy

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {
            'epsilon': 1.0,  # Lower = more private
            'delta': 1e-5
        }
    }
)
generator.fit(real)
private_synthetic = generator.generate(1000)

# Re-test
report = run_privacy_audit(real, private_synthetic)
```

### K-Anonymity

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'k_anonymity': {
            'k': 5,
            'quasi_identifiers': ['age', 'zip_code', 'gender']
        }
    }
)
```

### Combined Protections

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {'epsilon': 2.0},
        'k_anonymity': {'k': 3},
        'suppress_outliers': True,
        'min_category_count': 10
    }
)
```

## Iterative Privacy Improvement

```python
# Start with no special privacy settings
generator = SyntheticGenerator(method='ctgan')
generator.fit(real)
synthetic = generator.generate(1000)

# Test privacy
report = run_privacy_audit(real, synthetic)

# If privacy score is low, add protections
while report.overall_score < 0.9:
    current_epsilon = generator.config.get('epsilon', 10)
    generator.config['epsilon'] = current_epsilon / 2
    
    generator.fit(real)
    synthetic = generator.generate(1000)
    report = run_privacy_audit(real, synthetic)
    
    print(f"ε={current_epsilon/2}: Privacy={report.overall_score:.1%}")
```

## Sensitive Column Detection

Automatically identify potentially sensitive columns:

```python
from genesis.privacy_attacks import detect_sensitive_columns

sensitive = detect_sensitive_columns(real)

print("Detected sensitive columns:")
for col, reason in sensitive.items():
    print(f"  {col}: {reason}")

# Use in audit
report = run_privacy_audit(real, synthetic, sensitive_columns=list(sensitive.keys()))
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator, run_privacy_audit

# Load healthcare data
patients = pd.read_csv('patients.csv')

# Initial generation
generator = SyntheticGenerator(method='ctgan')
generator.fit(patients, discrete_columns=['diagnosis', 'gender'])
synthetic = generator.generate(len(patients))

# Privacy audit
report = run_privacy_audit(
    patients,
    synthetic,
    sensitive_columns=['diagnosis', 'ssn', 'income'],
    quasi_identifiers=['age', 'gender', 'zip_code']
)

print(f"Initial privacy score: {report.overall_score:.1%}")

# If not private enough, add protections
if report.overall_score < 0.9:
    generator = SyntheticGenerator(
        method='ctgan',
        privacy={
            'differential_privacy': {'epsilon': 1.0},
            'k_anonymity': {'k': 5, 'quasi_identifiers': ['age', 'gender', 'zip_code']},
            'suppress_outliers': True
        }
    )
    generator.fit(patients, discrete_columns=['diagnosis', 'gender'])
    synthetic = generator.generate(len(patients))
    
    # Re-test
    report = run_privacy_audit(
        patients,
        synthetic,
        sensitive_columns=['diagnosis', 'ssn', 'income'],
        quasi_identifiers=['age', 'gender', 'zip_code']
    )
    print(f"Protected privacy score: {report.overall_score:.1%}")

# Save with privacy report
synthetic.to_csv('private_patients.csv', index=False)
report.save('privacy_report.html')
```

## CLI Usage

```bash
# Quick privacy audit
genesis privacy-audit real.csv synthetic.csv

# With options
genesis privacy-audit real.csv synthetic.csv \
  --sensitive-columns income,health_status \
  --quasi-identifiers age,gender,zip \
  --output privacy_report.html
```

## Regulatory Compliance

Map privacy scores to compliance requirements:

```python
# GDPR-focused audit
report = run_privacy_audit(
    real, synthetic,
    compliance='gdpr',
    sensitive_columns=['health_data', 'ethnicity']
)

print(f"GDPR compliance: {report.gdpr_compliant}")

# HIPAA-focused audit
report = run_privacy_audit(
    real, synthetic,
    compliance='hipaa',
    phi_columns=['name', 'dob', 'ssn', 'medical_record']
)

print(f"HIPAA compliance: {report.hipaa_compliant}")
```

## Best Practices

1. **Always test before release** - Never assume synthetic = private
2. **Test with realistic adversaries** - Use strong attack assumptions
3. **Document privacy guarantees** - Record epsilon, k values used
4. **Re-test after any changes** - New data or config can affect privacy
5. **Use multiple attack types** - One test isn't enough

## Next Steps

- **[Privacy Configuration](/docs/concepts/privacy)** - Learn about privacy settings
- **[Evaluation](/docs/concepts/evaluation)** - Balance privacy with utility
- **[Privacy Compliance](/docs/guides/privacy-compliance)** - Regulatory requirements
