# ADR-0018: Compliance Certificates as First-Class Artifacts

## Status

Accepted

## Context

Organizations using synthetic data for sensitive applications face compliance requirements:

| Regulation | Requirement |
|------------|-------------|
| **GDPR Article 35** | Data Protection Impact Assessment for high-risk processing |
| **HIPAA** | Documentation that PHI is not disclosed in synthetic data |
| **CCPA** | Demonstrate synthetic data doesn't constitute "personal information" |
| **SOC 2** | Evidence of data protection controls |

Traditional approaches:

- Manual documentation written by compliance teams
- Separate privacy audits performed post-hoc
- No standardized format across organizations
- Privacy metrics computed but not formally certified

We observed that:

1. Privacy metrics are already computed during evaluation
2. The generator knows its privacy configuration
3. Documentation could be auto-generated with the right structure
4. Certificates should be machine-readable for automation

## Decision

We implement **privacy compliance certificates as automated, first-class artifacts**:

```python
from genesis.compliance import PrivacyCertificate, ComplianceFramework

cert = PrivacyCertificate(
    real_data=original_df,
    synthetic_data=synthetic_df,
    generator=trained_generator,
)

# Generate framework-specific certificate
report = cert.generate(framework=ComplianceFramework.GDPR)

# Check compliance status
print(f"Compliant: {report.is_compliant}")
print(f"Risk Level: {report.overall_risk_level}")

# Export in multiple formats
report.to_html("privacy_certificate.html")
report.to_pdf("privacy_certificate.pdf")
report.to_json("privacy_certificate.json")
```

### Certificate Structure

```python
@dataclass
class ComplianceReport:
    # Identification
    certificate_id: str
    generation_date: str
    framework: ComplianceFramework
    
    # Metadata
    metadata: CertificateMetadata
    
    # Privacy Metrics
    metrics: List[PrivacyMetricResult]
    
    # Compliance Assessment
    is_compliant: bool
    overall_risk_level: RiskLevel
    
    # Recommendations
    recommendations: List[str]
    
    # Signatures
    data_hash: str       # SHA-256 of synthetic data
    config_hash: str     # SHA-256 of generator config
```

### Privacy Metrics Evaluated

| Metric | Description | GDPR | HIPAA | CCPA |
|--------|-------------|------|-------|------|
| **Distance to Closest Record** | Minimum distance to any real record | ✓ | ✓ | ✓ |
| **Membership Inference Risk** | Can attacker determine if record was in training? | ✓ | ✓ | ✓ |
| **Attribute Inference Risk** | Can attacker infer sensitive attributes? | ✓ | ✓ | |
| **Re-identification Risk** | Probability of linking to real individual | ✓ | ✓ | ✓ |
| **k-Anonymity** | Minimum group size for quasi-identifiers | ✓ | ✓ | |
| **l-Diversity** | Diversity of sensitive values per group | | ✓ | |
| **Differential Privacy ε** | Privacy budget consumed | ✓ | ✓ | |

## Consequences

### Positive

- **Automated compliance**: Certificates generated without manual effort
- **Quantified risk**: Risk levels based on measured metrics, not opinions
- **Audit-ready**: Machine-readable formats for compliance tools
- **Framework-aware**: Different frameworks get tailored assessments
- **Reproducible**: Certificate includes hashes for verification

### Negative

- **False confidence risk**: Certificate doesn't guarantee legal compliance
- **Metric limitations**: Some privacy risks aren't quantifiable
- **Framework evolution**: Regulations change; certificates need updates
- **Not legal advice**: Users must still consult legal/compliance teams

### Risk Levels

```python
class RiskLevel(Enum):
    NEGLIGIBLE = "negligible"  # DCR > 0.5, MIA < 0.55
    LOW = "low"                # DCR > 0.3, MIA < 0.60
    MEDIUM = "medium"          # DCR > 0.1, MIA < 0.70
    HIGH = "high"              # DCR > 0.05, MIA < 0.80
    CRITICAL = "critical"      # Below thresholds
```

### Framework-Specific Checks

```python
def _evaluate_gdpr(self) -> List[PrivacyMetricResult]:
    """GDPR Article 35 DPIA requirements."""
    return [
        self._check_purpose_limitation(),
        self._check_data_minimization(),
        self._check_reidentification_risk(),
        self._check_differential_privacy(),
        self._check_automated_decision_making(),
    ]

def _evaluate_hipaa(self) -> List[PrivacyMetricResult]:
    """HIPAA Safe Harbor and Expert Determination."""
    return [
        self._check_18_identifiers_removed(),
        self._check_expert_determination_standard(),
        self._check_phi_disclosure_risk(),
        self._check_minimum_necessary(),
    ]
```

## Examples

```python
# Basic certificate generation
from genesis.compliance import PrivacyCertificate, ComplianceFramework

cert = PrivacyCertificate(real_data, synthetic_data, generator)

# GDPR compliance
gdpr_report = cert.generate(framework=ComplianceFramework.GDPR)
print(f"GDPR Compliant: {gdpr_report.is_compliant}")

for metric in gdpr_report.metrics:
    status = "✓" if metric.passed else "✗"
    print(f"  {status} {metric.metric_name}: {metric.value:.3f} (threshold: {metric.threshold})")

# HIPAA compliance
hipaa_report = cert.generate(framework=ComplianceFramework.HIPAA)
if not hipaa_report.is_compliant:
    print("Recommendations:")
    for rec in hipaa_report.recommendations:
        print(f"  - {rec}")

# Export for audit
gdpr_report.to_pdf("gdpr_certificate.pdf")
gdpr_report.to_json("gdpr_certificate.json")

# Batch certification
for dataset_name, synthetic_df in datasets.items():
    cert = PrivacyCertificate(real_df, synthetic_df, generator)
    report = cert.generate(ComplianceFramework.GENERAL)
    report.to_json(f"certificates/{dataset_name}.json")
```

## Certificate Output Example

```
╔══════════════════════════════════════════════════════════════════╗
║                    PRIVACY COMPLIANCE CERTIFICATE                 ║
║                         GDPR Assessment                           ║
╠══════════════════════════════════════════════════════════════════╣
║ Certificate ID: cert-a1b2c3d4-5678-90ab-cdef                     ║
║ Generated: 2024-06-15T14:30:00Z                                  ║
║ Framework: GDPR (EU General Data Protection Regulation)          ║
╠══════════════════════════════════════════════════════════════════╣
║ OVERALL ASSESSMENT: COMPLIANT                                     ║
║ Risk Level: LOW                                                   ║
╠══════════════════════════════════════════════════════════════════╣
║ PRIVACY METRICS                                                   ║
║ ✓ Distance to Closest Record: 0.342 (threshold: 0.100)           ║
║ ✓ Membership Inference Risk: 0.523 (threshold: 0.600)            ║
║ ✓ Attribute Inference Risk: 0.481 (threshold: 0.600)             ║
║ ✓ Re-identification Risk: 0.002 (threshold: 0.050)               ║
║ ✓ Differential Privacy ε: 1.000 (threshold: 10.000)              ║
╠══════════════════════════════════════════════════════════════════╣
║ DATA SUMMARY                                                      ║
║ Original Records: 50,000 | Synthetic Records: 50,000             ║
║ Columns: 15 | Generator: CTGAN                                   ║
╠══════════════════════════════════════════════════════════════════╣
║ HASHES (for verification)                                         ║
║ Synthetic Data: sha256:e5f6g7h8...                               ║
║ Generator Config: sha256:i9j0k1l2...                             ║
╚══════════════════════════════════════════════════════════════════╝
```
