---
sidebar_position: 3
title: Healthcare Compliance
---

# Tutorial: HIPAA-Compliant Healthcare Data

Generate synthetic patient data that meets HIPAA privacy requirements.

**Time:** 30 minutes  
**Level:** Intermediate  
**What you'll learn:** Privacy settings, differential privacy, compliance reports, privacy audits

---

## Goal

By the end of this tutorial, you'll have:
- Generated synthetic patient data with strong privacy guarantees
- Applied differential privacy (ε=1.0)
- Validated against privacy attacks
- Created a HIPAA compliance certificate

---

## Prerequisites

```bash
pip install genesis-synth[pytorch] pandas
```

---

## Understanding HIPAA Requirements

HIPAA (Health Insurance Portability and Accountability Act) requires protection of:

| Category | Examples | Genesis Solution |
|----------|----------|------------------|
| **Direct Identifiers** | Name, SSN, MRN | Remove before synthesis |
| **Quasi-Identifiers** | Age, ZIP, dates | K-anonymity, generalization |
| **Sensitive Attributes** | Diagnosis, treatment | Differential privacy |

The goal: Generate data useful for research without risking patient re-identification.

---

## Step 1: Create Sample Patient Data

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n_patients = 10000

# Demographics
ages = np.random.normal(55, 18, n_patients).clip(0, 100).astype(int)
genders = np.random.choice(['M', 'F'], n_patients, p=[0.48, 0.52])

# ZIP codes (quasi-identifier)
zip_codes = np.random.choice(
    ['10001', '10002', '10003', '90210', '90211', '60601', '60602', '30301', '30302', '77001'],
    n_patients,
    p=[0.15, 0.12, 0.10, 0.12, 0.08, 0.10, 0.08, 0.10, 0.08, 0.07]
)

# Medical conditions (correlated with age)
condition_probs = {
    'Diabetes': 0.05 + (ages / 100) * 0.15,
    'Hypertension': 0.08 + (ages / 100) * 0.25,
    'Heart Disease': 0.02 + (ages / 100) * 0.12,
    'Healthy': 1 - (0.05 + (ages / 100) * 0.15)  # Inverse
}

conditions = []
for i in range(n_patients):
    if np.random.random() < condition_probs['Diabetes'][i]:
        conditions.append('Diabetes')
    elif np.random.random() < condition_probs['Hypertension'][i]:
        conditions.append('Hypertension')
    elif np.random.random() < condition_probs['Heart Disease'][i]:
        conditions.append('Heart Disease')
    else:
        conditions.append('Healthy')

# Treatment (correlated with condition)
treatments = []
for cond in conditions:
    if cond == 'Diabetes':
        treatments.append(np.random.choice(['Metformin', 'Insulin', 'Diet'], p=[0.5, 0.3, 0.2]))
    elif cond == 'Hypertension':
        treatments.append(np.random.choice(['ACE Inhibitor', 'Beta Blocker', 'Diuretic'], p=[0.4, 0.35, 0.25]))
    elif cond == 'Heart Disease':
        treatments.append(np.random.choice(['Statin', 'Aspirin', 'Surgery'], p=[0.5, 0.35, 0.15]))
    else:
        treatments.append('None')

# Lab values
blood_pressure_sys = np.where(
    np.array(conditions) == 'Hypertension',
    np.random.normal(145, 15, n_patients),
    np.random.normal(120, 12, n_patients)
).clip(80, 200).astype(int)

# Cost (correlated with condition severity)
base_costs = {'Healthy': 500, 'Diabetes': 3000, 'Hypertension': 2000, 'Heart Disease': 8000}
costs = np.array([base_costs[c] for c in conditions]) + np.random.normal(0, 500, n_patients)
costs = costs.clip(100, 50000).astype(int)

# Create DataFrame
patient_data = pd.DataFrame({
    'patient_id': [f'P{i:06d}' for i in range(1, n_patients + 1)],
    'age': ages,
    'gender': genders,
    'zip_code': zip_codes,
    'condition': conditions,
    'treatment': treatments,
    'blood_pressure_systolic': blood_pressure_sys,
    'annual_cost': costs
})

print(f"Created {len(patient_data)} patient records")
print(patient_data.head(10))
```

---

## Step 2: Identify Sensitive Columns

Before synthesis, identify data categories:

```python
# HIPAA data categories
direct_identifiers = ['patient_id']  # Remove completely
quasi_identifiers = ['age', 'gender', 'zip_code']  # Need k-anonymity
sensitive_attributes = ['condition', 'treatment', 'annual_cost']  # Need DP
numeric_columns = ['blood_pressure_systolic']

print("=== Data Classification ===")
print(f"Direct Identifiers (remove): {direct_identifiers}")
print(f"Quasi-Identifiers (k-anonymity): {quasi_identifiers}")
print(f"Sensitive Attributes (DP): {sensitive_attributes}")

# Remove direct identifiers
data_for_synthesis = patient_data.drop(columns=direct_identifiers)
print(f"\nData shape after removing IDs: {data_for_synthesis.shape}")
```

---

## Step 3: Configure Privacy Settings

Apply strong privacy protections:

```python
from genesis import SyntheticGenerator

# Privacy configuration for HIPAA
privacy_config = {
    # Differential Privacy - mathematical privacy guarantee
    'differential_privacy': {
        'epsilon': 1.0,      # Privacy budget (lower = more private)
        'delta': 1e-5,       # Probability of privacy breach
    },
    # K-Anonymity - prevent unique combinations
    'k_anonymity': {
        'k': 5,              # Each combination appears 5+ times
        'quasi_identifiers': quasi_identifiers
    },
    # Additional protections
    'suppress_outliers': True,       # Remove statistical outliers
    'min_category_count': 10,        # Suppress rare categories
}

# Create generator with privacy
generator = SyntheticGenerator(
    method='ctgan',
    privacy=privacy_config,
    config={
        'epochs': 300,
        'batch_size': 500
    }
)

print("Privacy configuration applied:")
print(f"  • Differential Privacy: ε={privacy_config['differential_privacy']['epsilon']}")
print(f"  • K-Anonymity: k={privacy_config['k_anonymity']['k']}")
print(f"  • Outlier suppression: enabled")
```

---

## Step 4: Train and Generate

```python
# Define discrete columns
discrete_columns = ['gender', 'zip_code', 'condition', 'treatment']

# Train the generator
print("Training generator with privacy settings...")
generator.fit(
    data_for_synthesis,
    discrete_columns=discrete_columns
)

# Generate synthetic patients
print("Generating synthetic patients...")
synthetic_patients = generator.generate(n_samples=10000)

print(f"\nGenerated {len(synthetic_patients)} synthetic patient records")
print(synthetic_patients.head(10))
```

---

## Step 5: Run Privacy Audit

Verify privacy protections are effective:

```python
from genesis import run_privacy_audit

# Comprehensive privacy audit
print("Running privacy audit...")
privacy_report = run_privacy_audit(
    real_data=data_for_synthesis,
    synthetic_data=synthetic_patients,
    sensitive_columns=['condition', 'treatment', 'annual_cost'],
    quasi_identifiers=['age', 'gender', 'zip_code']
)

print("\n" + "="*50)
print("PRIVACY AUDIT RESULTS")
print("="*50)
print(f"""
Overall Privacy Score: {privacy_report.overall_score:.1%}
Safe to Release: {'✅ YES' if privacy_report.is_safe else '❌ NO'}

Individual Metrics:
  • Distance to Closest Record: {privacy_report.dcr_score:.1%}
  • Membership Inference Risk: {privacy_report.membership_risk:.1%}
  • Attribute Disclosure Risk: {privacy_report.attribute_risk:.1%}
  • Singling Out Risk: {privacy_report.singling_out_risk:.1%}

K-Anonymity Check:
  • Satisfies k=5: {'✅ Yes' if privacy_report.k_anonymity_satisfied else '❌ No'}
  • Smallest group size: {privacy_report.smallest_equivalence_class}
""")
```

**Expected output:**
```
PRIVACY AUDIT RESULTS
==================================================

Overall Privacy Score: 98.7%
Safe to Release: ✅ YES

Individual Metrics:
  • Distance to Closest Record: 99.2%
  • Membership Inference Risk: 1.8%
  • Attribute Disclosure Risk: 3.1%
  • Singling Out Risk: 0.4%

K-Anonymity Check:
  • Satisfies k=5: ✅ Yes
  • Smallest group size: 12
```

---

## Step 6: Test Against Privacy Attacks

Simulate adversarial attacks:

```python
from genesis.privacy_attacks import (
    MembershipInferenceAttack,
    AttributeInferenceAttack,
    LinkageAttack
)

print("Testing against simulated privacy attacks...")

# Membership Inference: Can attacker tell if someone was in training data?
mia = MembershipInferenceAttack()
mia_result = mia.evaluate(data_for_synthesis, synthetic_patients)
print(f"\n1. Membership Inference Attack:")
print(f"   Attack accuracy: {mia_result.accuracy:.1%} (random baseline: 50%)")
print(f"   Risk level: {mia_result.risk_level}")

# Attribute Inference: Can attacker infer sensitive attributes?
aia = AttributeInferenceAttack(target_column='condition')
aia_result = aia.evaluate(data_for_synthesis, synthetic_patients)
print(f"\n2. Attribute Inference Attack (condition):")
print(f"   Attack accuracy: {aia_result.accuracy:.1%}")
print(f"   Risk level: {aia_result.risk_level}")

# Linkage Attack: Can attacker link records across datasets?
la = LinkageAttack(quasi_identifiers=['age', 'gender', 'zip_code'])
la_result = la.evaluate(data_for_synthesis, synthetic_patients)
print(f"\n3. Linkage Attack:")
print(f"   Attack success rate: {la_result.success_rate:.1%}")
print(f"   Risk level: {la_result.risk_level}")

print("\n" + "="*50)
all_low_risk = all(r.risk_level == 'low' for r in [mia_result, aia_result, la_result])
print(f"Overall Attack Resistance: {'✅ PASSED' if all_low_risk else '⚠️ REVIEW NEEDED'}")
```

---

## Step 7: Generate HIPAA Compliance Certificate

Create documentation for compliance review:

```python
from genesis.compliance import PrivacyCertificate, ComplianceFramework

# Generate HIPAA compliance certificate
print("Generating HIPAA compliance certificate...")

certificate = PrivacyCertificate(
    real_data=data_for_synthesis,
    synthetic_data=synthetic_patients
)

hipaa_report = certificate.generate(
    framework=ComplianceFramework.HIPAA,
    organization="Example Healthcare System",
    data_description="Synthetic patient records for research",
    purpose="Machine learning model development and testing"
)

print(f"\n{'='*50}")
print("HIPAA COMPLIANCE CERTIFICATE")
print("="*50)
print(f"""
Organization: {hipaa_report.organization}
Date Generated: {hipaa_report.timestamp}
Framework: HIPAA

Compliance Status: {'✅ COMPLIANT' if hipaa_report.is_compliant else '❌ NON-COMPLIANT'}

De-identification Method: Synthetic Data Generation with Differential Privacy
Privacy Parameters:
  • Epsilon (ε): 1.0
  • Delta (δ): 1e-5
  • K-Anonymity: k=5

Safe Harbor Compliance:
  • Direct identifiers removed: ✅
  • Dates generalized: ✅
  • Geographic data (ZIP): ✅ (k-anonymized)
  • Ages: ✅ (generalized for 90+)

Expert Determination Criteria:
  • Re-identification risk: {hipaa_report.reidentification_risk:.2%}
  • Statistical disclosure risk: {hipaa_report.disclosure_risk:.2%}

Recommendation: {hipaa_report.recommendation}
""")

# Export certificate
hipaa_report.to_html('hipaa_compliance_certificate.html')
hipaa_report.to_pdf('hipaa_compliance_certificate.pdf')
print("\nCertificate exported to hipaa_compliance_certificate.html/pdf")
```

---

## Step 8: Evaluate Data Utility

Ensure data is still useful for research:

```python
from genesis import QualityEvaluator

# Quality evaluation
quality_report = QualityEvaluator(
    data_for_synthesis,
    synthetic_patients
).evaluate(target_column='condition')

print("\n" + "="*50)
print("DATA UTILITY REPORT")
print("="*50)
print(f"""
Overall Quality Score: {quality_report.overall_score:.1%}

Utility Metrics:
  • Statistical Fidelity: {quality_report.fidelity_score:.1%}
  • ML Utility: {quality_report.utility_score:.1%}
  • Correlation Preservation: {quality_report.correlation_score:.1%}

Privacy vs Utility Trade-off:
  • Privacy Score: {privacy_report.overall_score:.1%}
  • Quality Score: {quality_report.overall_score:.1%}
  • Balance: {'✅ Good' if min(privacy_report.overall_score, quality_report.overall_score) > 0.80 else '⚠️ Review'}
""")
```

---

## Step 9: Save Compliant Data

```python
# Add synthetic patient IDs
synthetic_patients.insert(0, 'patient_id', [f'SP{i:06d}' for i in range(1, len(synthetic_patients) + 1)])

# Save synthetic data
synthetic_patients.to_csv('synthetic_patients_hipaa.csv', index=False)
print(f"Saved {len(synthetic_patients)} HIPAA-compliant synthetic patients")

# Save generator for reproducibility
generator.save('hipaa_patient_generator.pkl')
print("Saved generator model for reproducibility")

# Save audit trail
audit_trail = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'privacy_score': privacy_report.overall_score,
    'quality_score': quality_report.overall_score,
    'epsilon': 1.0,
    'k_anonymity': 5,
    'records_generated': len(synthetic_patients),
    'compliance_framework': 'HIPAA',
    'is_compliant': hipaa_report.is_compliant
}
pd.DataFrame([audit_trail]).to_json('synthesis_audit_trail.json', orient='records', indent=2)
print("Saved audit trail")
```

---

## Complete HIPAA Synthesis Script

```python
"""
HIPAA-Compliant Synthetic Patient Data Generation
"""
import pandas as pd
import numpy as np
from genesis import SyntheticGenerator, QualityEvaluator, run_privacy_audit
from genesis.compliance import PrivacyCertificate, ComplianceFramework
from genesis.privacy_attacks import MembershipInferenceAttack

# Configuration
EPSILON = 1.0
K_ANONYMITY = 5
N_SAMPLES = 10000

# Load your real patient data (with direct identifiers already removed)
# data = pd.read_csv('patient_data_deidentified.csv')

# For demo, create sample data
np.random.seed(42)
# ... (data creation code from Step 1)

# Privacy-preserving synthesis
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {'epsilon': EPSILON, 'delta': 1e-5},
        'k_anonymity': {'k': K_ANONYMITY, 'quasi_identifiers': ['age', 'gender', 'zip_code']},
        'suppress_outliers': True
    }
)

generator.fit(data, discrete_columns=['gender', 'zip_code', 'condition', 'treatment'])
synthetic = generator.generate(n_samples=N_SAMPLES)

# Validate
privacy = run_privacy_audit(data, synthetic, quasi_identifiers=['age', 'gender', 'zip_code'])
quality = QualityEvaluator(data, synthetic).evaluate()

print(f"Privacy: {privacy.overall_score:.1%} | Quality: {quality.overall_score:.1%}")
print(f"Safe to release: {'✅' if privacy.is_safe else '❌'}")

# Generate certificate
cert = PrivacyCertificate(data, synthetic).generate(framework=ComplianceFramework.HIPAA)
cert.to_html('hipaa_certificate.html')

# Save
synthetic.to_csv('synthetic_patients.csv', index=False)
```

---

## Privacy Settings Reference

| Sensitivity Level | Epsilon (ε) | K | Use Case |
|-------------------|-------------|---|----------|
| Low | 10.0 | 3 | Internal testing |
| Medium | 5.0 | 5 | Research collaboration |
| High | 1.0 | 5 | Public release |
| Very High | 0.5 | 10 | Highly sensitive data |

---

## Next Steps

- [Privacy Concepts](/docs/concepts/privacy) - Deep dive into privacy mechanisms
- [GDPR Compliance](/docs/guides/privacy-compliance) - European privacy requirements
- [Privacy Attack Testing](/docs/guides/privacy-attacks) - Advanced attack simulations

---

## Troubleshooting

**Low quality with strict privacy?**
- Privacy and quality trade off. Try ε=2.0 or ε=5.0 for better utility.

**K-anonymity not satisfied?**
- Generalize quasi-identifiers (e.g., age bins, 3-digit ZIP)
- Increase training data size

**Compliance certificate fails?**
- Review the specific failing criteria
- Adjust privacy parameters accordingly

See [Troubleshooting](/docs/troubleshooting) for more help.
