---
sidebar_position: 14
title: Privacy Compliance
---

# Privacy Compliance

Generate synthetic data that meets regulatory requirements.

## Overview

Genesis helps achieve compliance with:
- **GDPR** (EU General Data Protection Regulation)
- **HIPAA** (US Health Insurance Portability and Accountability Act)
- **CCPA** (California Consumer Privacy Act)
- **LGPD** (Brazil's General Data Protection Law)

## Compliance Modes

### GDPR Mode

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='ctgan',
    compliance='gdpr',
    privacy={
        'differential_privacy': {'epsilon': 1.0},
        'k_anonymity': {'k': 5},
        'remove_direct_identifiers': True
    }
)
```

### HIPAA Mode

```python
generator = SyntheticGenerator(
    method='ctgan',
    compliance='hipaa',
    privacy={
        'phi_handling': 'synthetic',
        'de_identification': 'safe_harbor'
    }
)
```

### CCPA Mode

```python
generator = SyntheticGenerator(
    method='ctgan',
    compliance='ccpa',
    privacy={
        'personal_info_handling': 'synthetic'
    }
)
```

## GDPR Compliance

### Article 5: Data Minimization

```python
# Generate only necessary columns
generator.fit(data[['age_group', 'region', 'purchase_category']])
# Not: data[['name', 'email', 'exact_address', ...]]
```

### Article 17: Right to Erasure

```python
# Synthetic data has no link to originals
# No individual's data exists to be erased
synthetic = generator.generate(10000)

# Verify no linkage
from genesis import run_privacy_audit
report = run_privacy_audit(real_data, synthetic)
assert report.linkage_score < 0.05  # Less than 5% linkable
```

### Article 25: Privacy by Design

```python
# Build privacy into the generation process
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {'epsilon': 1.0},
        'suppress_outliers': True,
        'min_category_count': 10
    }
)
```

## HIPAA Compliance

### Protected Health Information (PHI)

HIPAA defines 18 PHI identifiers:

```python
phi_identifiers = [
    'name', 'address', 'dates', 'phone', 'fax', 'email',
    'ssn', 'mrn', 'health_plan_id', 'account_numbers',
    'certificate_numbers', 'vehicle_ids', 'device_ids',
    'urls', 'ip_addresses', 'biometric_ids', 'photos',
    'any_unique_identifier'
]
```

### Safe Harbor De-identification

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='ctgan',
    compliance='hipaa',
    privacy={
        'de_identification': 'safe_harbor',
        'phi_columns': [
            'patient_name', 'dob', 'ssn', 'address',
            'phone', 'email', 'mrn'
        ],
        'age_threshold': 89,  # Ages 90+ generalized
        'geographic_threshold': 3  # First 3 digits of zip only
    }
)
```

### Expert Determination

```python
# Statistical method for de-identification
from genesis.privacy_attacks import run_re_identification_risk

risk = run_re_identification_risk(
    synthetic_data,
    population_data,
    quasi_identifiers=['age', 'gender', 'zip3']
)

# HIPAA requires "very small" risk - typically < 0.05
assert risk < 0.05, "Re-identification risk too high"
```

## K-Anonymity

Ensure each record is indistinguishable from k-1 others:

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'k_anonymity': {
            'k': 5,
            'quasi_identifiers': ['age', 'gender', 'zip_code']
        }
    }
)

# Verify k-anonymity
from genesis.privacy import check_k_anonymity

k = check_k_anonymity(
    synthetic,
    quasi_identifiers=['age', 'gender', 'zip_code']
)
print(f"Achieved k={k}")  # Should be >= 5
```

## L-Diversity

Ensure diversity in sensitive attributes:

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'l_diversity': {
            'l': 3,
            'sensitive_columns': ['diagnosis'],
            'quasi_identifiers': ['age', 'gender', 'zip_code']
        }
    }
)
```

## T-Closeness

Limit distribution distance of sensitive attributes:

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        't_closeness': {
            't': 0.2,  # Maximum distance
            'sensitive_columns': ['income'],
            'quasi_identifiers': ['age', 'education']
        }
    }
)
```

## Differential Privacy

Mathematically provable privacy guarantee:

```python
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {
            'epsilon': 1.0,   # Privacy budget
            'delta': 1e-5     # Failure probability
        }
    }
)
```

### Epsilon Guidelines

| Use Case | Epsilon | Privacy Level |
|----------|---------|---------------|
| Medical research | 0.1-1.0 | Very high |
| Financial data | 1.0-3.0 | High |
| Marketing analytics | 3.0-10.0 | Moderate |
| Public statistics | 10.0+ | Low |

## Compliance Verification

### Generate Compliance Report

```python
from genesis import run_privacy_audit

report = run_privacy_audit(
    real_data,
    synthetic_data,
    compliance='gdpr',  # or 'hipaa', 'ccpa'
    sensitive_columns=['income', 'health_status'],
    quasi_identifiers=['age', 'gender', 'zip_code']
)

# Check compliance
print(f"GDPR Compliant: {report.gdpr_compliant}")
print(f"Privacy Score: {report.overall_score:.1%}")

# Generate audit report
report.save('compliance_report.pdf')
```

### Compliance Checklist

```python
from genesis.compliance import ComplianceChecker

checker = ComplianceChecker(regulation='gdpr')

# Run all checks
results = checker.check(real_data, synthetic_data)

for check, passed in results.items():
    status = "✓" if passed else "✗"
    print(f"{status} {check}")

# Example output:
# ✓ No direct identifiers
# ✓ K-anonymity (k=5)
# ✓ Differential privacy (ε=1.0)
# ✓ No linkage to original records
# ✗ L-diversity (l=2, required l=3)
```

## Data Processing Agreements

When using Genesis for third-party data:

```python
# Document data handling
from genesis.compliance import DataProcessingRecord

record = DataProcessingRecord(
    purpose="ML model testing",
    legal_basis="Legitimate interest",
    data_categories=["demographic", "behavioral"],
    retention_period="30 days",
    security_measures=["encryption", "access_control"]
)

# Attach to synthetic data metadata
synthetic.attrs['processing_record'] = record.to_dict()
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator, run_privacy_audit
from genesis.compliance import ComplianceChecker

# Load sensitive healthcare data
patients = pd.read_csv('patients.csv')

# Configure for HIPAA compliance
generator = SyntheticGenerator(
    method='ctgan',
    compliance='hipaa',
    privacy={
        'differential_privacy': {'epsilon': 1.0},
        'k_anonymity': {
            'k': 5,
            'quasi_identifiers': ['age_group', 'gender', 'state']
        },
        'de_identification': 'safe_harbor',
        'phi_columns': ['patient_name', 'dob', 'ssn', 'address', 'phone']
    }
)

# Generate synthetic patients
generator.fit(
    patients,
    discrete_columns=['diagnosis', 'treatment', 'gender']
)
synthetic = generator.generate(len(patients))

# Verify compliance
checker = ComplianceChecker(regulation='hipaa')
results = checker.check(patients, synthetic)

print("\nHIPAA Compliance Check:")
for check, passed in results.items():
    status = "✓" if passed else "✗"
    print(f"  {status} {check}")

# Run privacy audit
report = run_privacy_audit(
    patients,
    synthetic,
    sensitive_columns=['diagnosis', 'treatment'],
    quasi_identifiers=['age_group', 'gender', 'state']
)

print(f"\nPrivacy Score: {report.overall_score:.1%}")
print(f"HIPAA Compliant: {report.hipaa_compliant}")

# Save with compliance documentation
if report.hipaa_compliant:
    synthetic.to_csv('synthetic_patients_hipaa.csv', index=False)
    report.save('hipaa_compliance_report.pdf')
else:
    print("\n⚠️ Data does not meet HIPAA requirements")
    for rec in report.recommendations:
        print(f"  - {rec}")
```

## Best Practices

1. **Know your regulation** - Requirements vary by jurisdiction
2. **Use appropriate privacy levels** - Match epsilon to sensitivity
3. **Document everything** - Maintain compliance records
4. **Test rigorously** - Run privacy attacks before release
5. **Get legal review** - Consult compliance experts

## Regulatory Resources

- [GDPR Official Text](https://gdpr.eu/)
- [HIPAA Privacy Rule](https://www.hhs.gov/hipaa/for-professionals/privacy/)
- [CCPA Official Resources](https://oag.ca.gov/privacy/ccpa)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)

## Next Steps

- **[Privacy Attacks](/docs/guides/privacy-attacks)** - Test your data
- **[Privacy Concepts](/docs/concepts/privacy)** - Understand the theory
- **[API Reference](/docs/api/reference)** - Privacy settings
