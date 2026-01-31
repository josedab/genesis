# Privacy Attack Testing

Genesis provides tools to audit synthetic data for privacy vulnerabilities by simulating common privacy attacks.

## Overview

Privacy attack testing helps validate that synthetic data doesn't leak information about the original training data. Genesis implements three major attack types:

| Attack | What it Tests | Risk Level |
|--------|---------------|------------|
| **Membership Inference** | Can an attacker determine if a record was in the training data? | High |
| **Attribute Inference** | Can an attacker infer sensitive attributes from known attributes? | Medium |
| **Re-identification** | Can an attacker link synthetic records to real individuals? | Critical |

## Quick Start

```python
from genesis import run_privacy_audit

# Run all attacks
report = run_privacy_audit(
    real_data=original_df,
    synthetic_data=synthetic_df,
    sensitive_columns=["ssn", "income", "diagnosis"]
)

print(f"Overall Risk: {report.overall_risk}")
print(f"Passed: {report.passed}")
report.print_summary()
```

## Components

| Component | Purpose |
|-----------|---------|
| **MembershipInferenceAttack** | Tests if training records can be identified |
| **AttributeInferenceAttack** | Tests if sensitive attributes can be inferred |
| **ReidentificationAttack** | Tests if records can be linked to individuals |
| **PrivacyAttackTester** | Orchestrates multiple attacks |
| **run_privacy_audit()** | Convenience function for full audit |

## Membership Inference Attack

Tests whether an attacker can determine if a specific record was used to train the synthetic data generator.

```python
from genesis.privacy_attacks import MembershipInferenceAttack

attack = MembershipInferenceAttack()
result = attack.run(
    real_data=original_df,
    synthetic_data=synthetic_df,
    holdout_data=holdout_df  # Data NOT used in training
)

print(f"Attack accuracy: {result.accuracy:.2%}")
print(f"Advantage over random: {result.advantage:.2%}")
print(f"Risk level: {result.risk_level}")  # LOW, MEDIUM, HIGH
```

### Interpretation

| Metric | Good | Concerning |
|--------|------|------------|
| Accuracy | ~50% (random guessing) | >60% |
| Advantage | <5% | >10% |

## Attribute Inference Attack

Tests whether an attacker can infer sensitive attributes given knowledge of other attributes.

```python
from genesis.privacy_attacks import AttributeInferenceAttack

attack = AttributeInferenceAttack()
result = attack.run(
    real_data=original_df,
    synthetic_data=synthetic_df,
    sensitive_column="income",
    known_columns=["age", "education", "occupation"]
)

print(f"Inference accuracy: {result.accuracy:.2%}")
print(f"Baseline accuracy: {result.baseline:.2%}")
print(f"Lift: {result.lift:.2f}x")
```

### Multiple Sensitive Columns

```python
results = {}
for col in ["income", "diagnosis", "credit_score"]:
    result = attack.run(
        real_data=original_df,
        synthetic_data=synthetic_df,
        sensitive_column=col,
        known_columns=[c for c in df.columns if c != col]
    )
    results[col] = result.accuracy

# Find most vulnerable columns
vulnerable = [c for c, acc in results.items() if acc > 0.7]
```

## Re-identification Attack

Tests whether synthetic records can be linked back to real individuals using quasi-identifiers.

```python
from genesis.privacy_attacks import ReidentificationAttack

attack = ReidentificationAttack()
result = attack.run(
    real_data=original_df,
    synthetic_data=synthetic_df,
    quasi_identifiers=["age", "zipcode", "gender"]
)

print(f"Re-identification rate: {result.reidentification_rate:.2%}")
print(f"Unique matches: {result.unique_matches}")
print(f"Risk level: {result.risk_level}")
```

### Interpretation

| Re-identification Rate | Risk Level |
|-----------------------|------------|
| <1% | Low |
| 1-5% | Medium |
| >5% | High |
| >20% | Critical |

## Full Privacy Audit

```python
from genesis.privacy_attacks import PrivacyAttackTester, run_privacy_audit

# Using the tester class
tester = PrivacyAttackTester(
    sensitive_columns=["ssn", "income"],
    quasi_identifiers=["age", "zipcode", "gender"]
)

report = tester.run_all_attacks(
    real_data=original_df,
    synthetic_data=synthetic_df,
    holdout_data=holdout_df
)

# Or use convenience function
report = run_privacy_audit(
    real_data=original_df,
    synthetic_data=synthetic_df,
    sensitive_columns=["ssn", "income"],
    quasi_identifiers=["age", "zipcode", "gender"]
)

# Access results
print(f"Overall risk: {report.overall_risk}")
print(f"Membership attack: {report.membership_result.risk_level}")
print(f"Attribute attack: {report.attribute_results}")
print(f"Re-identification: {report.reidentification_result.risk_level}")
```

## Privacy Audit Report

```python
report = run_privacy_audit(real_df, synthetic_df)

# Print formatted summary
report.print_summary()

# Export to JSON
report.to_json("privacy_audit.json")

# Export to HTML
report.to_html("privacy_audit.html")

# Get pass/fail status
if report.passed:
    print("✓ Synthetic data passed privacy audit")
else:
    print("✗ Privacy concerns detected")
    for concern in report.concerns:
        print(f"  - {concern}")
```

## Configuring Risk Thresholds

```python
from genesis.privacy_attacks import PrivacyAttackTester, RiskThresholds

# Custom thresholds
thresholds = RiskThresholds(
    membership_advantage_max=0.05,  # Max 5% advantage
    attribute_accuracy_max=0.7,     # Max 70% inference accuracy
    reidentification_rate_max=0.01  # Max 1% re-identification
)

tester = PrivacyAttackTester(
    thresholds=thresholds,
    sensitive_columns=["income"]
)

report = tester.run_all_attacks(real_df, synthetic_df)
```

## Best Practices

1. **Always audit before release**: Run privacy attacks on synthetic data before sharing
2. **Use holdout data**: Keep a portion of data out of training for accurate membership tests
3. **Identify QIs carefully**: Include all potential quasi-identifiers in re-identification tests
4. **Set appropriate thresholds**: Risk tolerance depends on your use case and regulations
5. **Re-audit after changes**: Privacy properties can change with different generation settings

## CLI Usage

```bash
# Full privacy audit
genesis privacy-audit -r original.csv -s synthetic.csv --sensitive ssn,income

# Just membership inference
genesis privacy-audit -r original.csv -s synthetic.csv --attack membership

# With holdout data
genesis privacy-audit -r original.csv -s synthetic.csv -h holdout.csv

# Export report
genesis privacy-audit -r original.csv -s synthetic.csv -o audit_report.html
```

## Example: HIPAA-Compliant Synthetic Data

```python
import pandas as pd
from genesis import SyntheticGenerator, PrivacyConfig
from genesis.privacy_attacks import run_privacy_audit

# Load PHI data
df = pd.read_csv("patient_records.csv")

# HIPAA quasi-identifiers
hipaa_qis = ["zip", "birth_date", "admission_date", "gender", "race"]

# Generate with strong privacy
generator = SyntheticGenerator(
    method="dpctgan",
    privacy=PrivacyConfig(
        epsilon=0.1,  # Strong privacy
        enable_differential_privacy=True
    )
)
generator.fit(df)
synthetic = generator.generate(len(df))

# Audit for HIPAA compliance
report = run_privacy_audit(
    real_data=df,
    synthetic_data=synthetic,
    sensitive_columns=["diagnosis", "procedure_code", "prescription"],
    quasi_identifiers=hipaa_qis
)

if report.passed:
    print("✓ Synthetic data suitable for HIPAA-covered sharing")
    synthetic.to_csv("hipaa_safe_synthetic.csv", index=False)
else:
    print("✗ Privacy concerns - adjust generation parameters")
    report.print_summary()
```
