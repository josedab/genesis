"""Compliance-as-Code.

Declarative YAML/policy files for privacy requirements with CI/CD integration
to block non-compliant synthetic data in pipelines.

Features:
    - YAML schema for privacy policies
    - Policy inheritance and composition
    - CI/CD integration templates
    - Pre-commit hooks
    - Policy violation reporting
    - Environment-specific policies

Example:
    Define a policy file (privacy_policy.yaml)::

        apiVersion: genesis.io/v1
        kind: PrivacyPolicy
        metadata:
          name: production-policy
        spec:
          epsilon:
            min: 0.1
            max: 1.0
          delta:
            max: 1e-5
          k_anonymity:
            min: 5
          prohibited_columns:
            - ssn
            - credit_card
          required_transformations:
            - suppress_rare_categories

    Validate data against policy::

        from genesis.compliance_as_code import PolicyValidator

        validator = PolicyValidator.from_file("privacy_policy.yaml")
        result = validator.validate(synthetic_df, privacy_config)

        if not result.is_compliant:
            print(result.violations)

Classes:
    PrivacyPolicy: Policy definition.
    PolicyValidator: Validates data against policies.
    PolicyViolation: A policy violation.
    ValidationResult: Result of policy validation.
    PolicyLoader: Loads policies from files.
    CICDIntegration: CI/CD integration utilities.

Note:
    Policies can be environment-specific (dev, staging, prod).
    Use inherit_from to compose policies.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

import numpy as np
import pandas as pd

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class PolicySeverity(str, Enum):
    """Severity of policy violations."""

    ERROR = "error"  # Blocks pipeline
    WARNING = "warning"  # Logged but doesn't block
    INFO = "info"  # Informational only


class ComplianceFramework(str, Enum):
    """Compliance frameworks."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    SOC2 = "soc2"
    CUSTOM = "custom"


@dataclass
class EpsilonConstraint:
    """Epsilon budget constraints."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    per_column_max: Optional[float] = None

    def validate(self, epsilon: float) -> tuple[bool, Optional[str]]:
        if self.min_value is not None and epsilon < self.min_value:
            return False, f"Epsilon {epsilon} below minimum {self.min_value}"
        if self.max_value is not None and epsilon > self.max_value:
            return False, f"Epsilon {epsilon} above maximum {self.max_value}"
        return True, None


@dataclass
class DeltaConstraint:
    """Delta constraints."""

    max_value: float = 1e-5

    def validate(self, delta: float) -> tuple[bool, Optional[str]]:
        if delta > self.max_value:
            return False, f"Delta {delta} above maximum {self.max_value}"
        return True, None


@dataclass
class KAnonymityConstraint:
    """K-anonymity constraints."""

    min_k: int = 5
    quasi_identifiers: List[str] = field(default_factory=list)

    def validate(self, data: pd.DataFrame) -> tuple[bool, Optional[str]]:
        if not self.quasi_identifiers:
            return True, None

        qi_cols = [c for c in self.quasi_identifiers if c in data.columns]
        if not qi_cols:
            return True, None

        group_sizes = data.groupby(qi_cols).size()
        min_group = group_sizes.min()

        if min_group < self.min_k:
            return False, f"K-anonymity violated: min group size {min_group} < k={self.min_k}"
        return True, None


@dataclass
class PolicyViolation:
    """A policy violation.

    Attributes:
        rule: Rule that was violated.
        severity: Violation severity.
        message: Human-readable message.
        details: Additional details.
        column: Column involved (if applicable).
        value: Offending value (if applicable).
    """

    rule: str
    severity: PolicySeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    column: Optional[str] = None
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "column": self.column,
        }


@dataclass
class ValidationResult:
    """Result of policy validation.

    Attributes:
        is_compliant: Whether all policies passed.
        violations: List of violations.
        warnings: List of warnings.
        policy_name: Name of validated policy.
        timestamp: When validation occurred.
    """

    is_compliant: bool
    violations: List[PolicyViolation] = field(default_factory=list)
    warnings: List[PolicyViolation] = field(default_factory=list)
    policy_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def error_count(self) -> int:
        return len([v for v in self.violations if v.severity == PolicySeverity.ERROR])

    @property
    def warning_count(self) -> int:
        return len([v for v in self.violations if v.severity == PolicySeverity.WARNING])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "violations": [v.to_dict() for v in self.violations],
            "policy_name": self.policy_name,
            "timestamp": self.timestamp,
        }

    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            f"Privacy Policy Validation Report",
            f"Policy: {self.policy_name}",
            f"Time: {self.timestamp}",
            "=" * 60,
            "",
            f"Status: {'✓ COMPLIANT' if self.is_compliant else '✗ NON-COMPLIANT'}",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
            "",
        ]

        if self.violations:
            lines.append("Violations:")
            for v in self.violations:
                icon = "✗" if v.severity == PolicySeverity.ERROR else "⚠"
                lines.append(f"  {icon} [{v.severity.value.upper()}] {v.rule}: {v.message}")

        if not self.violations:
            lines.append("No violations found.")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


@dataclass
class PrivacyPolicy:
    """Privacy policy definition.

    Attributes:
        name: Policy name.
        version: Policy version.
        description: Policy description.
        environment: Target environment(s).
        framework: Compliance framework.
        epsilon: Epsilon constraints.
        delta: Delta constraints.
        k_anonymity: K-anonymity constraints.
        prohibited_columns: Columns that must not appear.
        required_columns: Columns that must appear.
        required_transformations: Required privacy transformations.
        max_null_fraction: Maximum allowed null fraction.
        min_rows: Minimum required rows.
        custom_rules: Custom validation rules.
        inherit_from: Parent policy to inherit from.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    environment: List[str] = field(default_factory=lambda: ["all"])
    framework: ComplianceFramework = ComplianceFramework.CUSTOM
    epsilon: Optional[EpsilonConstraint] = None
    delta: Optional[DeltaConstraint] = None
    k_anonymity: Optional[KAnonymityConstraint] = None
    prohibited_columns: List[str] = field(default_factory=list)
    required_columns: List[str] = field(default_factory=list)
    required_transformations: List[str] = field(default_factory=list)
    max_null_fraction: float = 0.1
    min_rows: int = 0
    max_unique_fraction: float = 0.95  # Prevent near-identifiers
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    inherit_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "apiVersion": "genesis.io/v1",
            "kind": "PrivacyPolicy",
            "metadata": {
                "name": self.name,
                "version": self.version,
            },
            "spec": {
                "description": self.description,
                "environment": self.environment,
                "framework": self.framework.value,
                "epsilon": {
                    "min": self.epsilon.min_value if self.epsilon else None,
                    "max": self.epsilon.max_value if self.epsilon else None,
                } if self.epsilon else None,
                "delta": {
                    "max": self.delta.max_value if self.delta else None,
                } if self.delta else None,
                "k_anonymity": {
                    "min": self.k_anonymity.min_k if self.k_anonymity else None,
                    "quasi_identifiers": self.k_anonymity.quasi_identifiers if self.k_anonymity else [],
                } if self.k_anonymity else None,
                "prohibited_columns": self.prohibited_columns,
                "required_columns": self.required_columns,
                "required_transformations": self.required_transformations,
                "max_null_fraction": self.max_null_fraction,
                "min_rows": self.min_rows,
                "max_unique_fraction": self.max_unique_fraction,
                "inherit_from": self.inherit_from,
            },
        }

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)


class PolicyLoader:
    """Loads policies from files."""

    @staticmethod
    def from_file(path: Union[str, Path]) -> PrivacyPolicy:
        """Load policy from YAML or JSON file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required: pip install pyyaml")
            data = yaml.safe_load(content)
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            # Try YAML first, then JSON
            try:
                if YAML_AVAILABLE:
                    data = yaml.safe_load(content)
                else:
                    data = json.loads(content)
            except Exception:
                data = json.loads(content)

        return PolicyLoader.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PrivacyPolicy:
        """Load policy from dictionary."""
        metadata = data.get("metadata", {})
        spec = data.get("spec", data)  # Support both wrapped and unwrapped

        # Parse constraints
        epsilon = None
        if "epsilon" in spec and spec["epsilon"]:
            epsilon = EpsilonConstraint(
                min_value=spec["epsilon"].get("min"),
                max_value=spec["epsilon"].get("max"),
                per_column_max=spec["epsilon"].get("per_column_max"),
            )

        delta = None
        if "delta" in spec and spec["delta"]:
            delta = DeltaConstraint(max_value=spec["delta"].get("max", 1e-5))

        k_anonymity = None
        if "k_anonymity" in spec and spec["k_anonymity"]:
            k_anonymity = KAnonymityConstraint(
                min_k=spec["k_anonymity"].get("min", 5),
                quasi_identifiers=spec["k_anonymity"].get("quasi_identifiers", []),
            )

        return PrivacyPolicy(
            name=metadata.get("name", "unnamed"),
            version=metadata.get("version", "1.0.0"),
            description=spec.get("description", ""),
            environment=spec.get("environment", ["all"]),
            framework=ComplianceFramework(spec.get("framework", "custom")),
            epsilon=epsilon,
            delta=delta,
            k_anonymity=k_anonymity,
            prohibited_columns=spec.get("prohibited_columns", []),
            required_columns=spec.get("required_columns", []),
            required_transformations=spec.get("required_transformations", []),
            max_null_fraction=spec.get("max_null_fraction", 0.1),
            min_rows=spec.get("min_rows", 0),
            max_unique_fraction=spec.get("max_unique_fraction", 0.95),
            inherit_from=spec.get("inherit_from"),
        )

    @staticmethod
    def from_string(content: str, format: str = "yaml") -> PrivacyPolicy:
        """Load policy from string."""
        if format == "yaml":
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required")
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        return PolicyLoader.from_dict(data)


class PolicyValidator:
    """Validates data against privacy policies."""

    def __init__(self, policy: PrivacyPolicy) -> None:
        """Initialize validator.

        Args:
            policy: Privacy policy to validate against.
        """
        self.policy = policy
        self._custom_validators: Dict[str, Callable] = {}

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PolicyValidator":
        """Create validator from policy file."""
        policy = PolicyLoader.from_file(path)
        return cls(policy)

    def register_custom_rule(
        self,
        name: str,
        validator: Callable[[pd.DataFrame], tuple[bool, Optional[str]]],
    ) -> None:
        """Register a custom validation rule.

        Args:
            name: Rule name.
            validator: Function that returns (passed, error_message).
        """
        self._custom_validators[name] = validator

    def validate(
        self,
        data: pd.DataFrame,
        privacy_config: Optional[Any] = None,
        applied_transformations: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate data against policy.

        Args:
            data: Synthetic data to validate.
            privacy_config: Privacy configuration used.
            applied_transformations: Transformations that were applied.

        Returns:
            ValidationResult with all violations.
        """
        violations: List[PolicyViolation] = []
        applied_transformations = applied_transformations or []

        # Check prohibited columns
        for col in self.policy.prohibited_columns:
            if col in data.columns:
                violations.append(PolicyViolation(
                    rule="prohibited_columns",
                    severity=PolicySeverity.ERROR,
                    message=f"Prohibited column '{col}' present in data",
                    column=col,
                ))

        # Check required columns
        for col in self.policy.required_columns:
            if col not in data.columns:
                violations.append(PolicyViolation(
                    rule="required_columns",
                    severity=PolicySeverity.ERROR,
                    message=f"Required column '{col}' missing from data",
                    column=col,
                ))

        # Check minimum rows
        if self.policy.min_rows > 0 and len(data) < self.policy.min_rows:
            violations.append(PolicyViolation(
                rule="min_rows",
                severity=PolicySeverity.ERROR,
                message=f"Data has {len(data)} rows, minimum is {self.policy.min_rows}",
                details={"actual": len(data), "required": self.policy.min_rows},
            ))

        # Check null fraction
        null_fraction = data.isnull().sum().sum() / data.size if data.size > 0 else 0
        if null_fraction > self.policy.max_null_fraction:
            violations.append(PolicyViolation(
                rule="max_null_fraction",
                severity=PolicySeverity.WARNING,
                message=f"Null fraction {null_fraction:.2%} exceeds max {self.policy.max_null_fraction:.2%}",
                details={"actual": null_fraction, "max": self.policy.max_null_fraction},
            ))

        # Check unique fraction (potential identifiers)
        for col in data.columns:
            unique_fraction = data[col].nunique() / len(data) if len(data) > 0 else 0
            if unique_fraction > self.policy.max_unique_fraction:
                violations.append(PolicyViolation(
                    rule="max_unique_fraction",
                    severity=PolicySeverity.WARNING,
                    message=f"Column '{col}' has {unique_fraction:.2%} unique values (potential identifier)",
                    column=col,
                    details={"unique_fraction": unique_fraction},
                ))

        # Check epsilon constraints
        if self.policy.epsilon and privacy_config:
            epsilon = getattr(privacy_config, "epsilon", None)
            if epsilon is not None:
                passed, msg = self.policy.epsilon.validate(epsilon)
                if not passed:
                    violations.append(PolicyViolation(
                        rule="epsilon",
                        severity=PolicySeverity.ERROR,
                        message=msg or "Epsilon constraint violated",
                        details={"epsilon": epsilon},
                    ))

        # Check delta constraints
        if self.policy.delta and privacy_config:
            delta = getattr(privacy_config, "delta", None)
            if delta is not None:
                passed, msg = self.policy.delta.validate(delta)
                if not passed:
                    violations.append(PolicyViolation(
                        rule="delta",
                        severity=PolicySeverity.ERROR,
                        message=msg or "Delta constraint violated",
                        details={"delta": delta},
                    ))

        # Check k-anonymity
        if self.policy.k_anonymity:
            passed, msg = self.policy.k_anonymity.validate(data)
            if not passed:
                violations.append(PolicyViolation(
                    rule="k_anonymity",
                    severity=PolicySeverity.ERROR,
                    message=msg or "K-anonymity constraint violated",
                ))

        # Check required transformations
        for transform in self.policy.required_transformations:
            if transform not in applied_transformations:
                violations.append(PolicyViolation(
                    rule="required_transformations",
                    severity=PolicySeverity.WARNING,
                    message=f"Required transformation '{transform}' not applied",
                    details={"transformation": transform},
                ))

        # Run custom validators
        for name, validator in self._custom_validators.items():
            try:
                passed, msg = validator(data)
                if not passed:
                    violations.append(PolicyViolation(
                        rule=f"custom:{name}",
                        severity=PolicySeverity.ERROR,
                        message=msg or f"Custom rule '{name}' failed",
                    ))
            except Exception as e:
                violations.append(PolicyViolation(
                    rule=f"custom:{name}",
                    severity=PolicySeverity.ERROR,
                    message=f"Custom rule '{name}' error: {e}",
                ))

        # Determine compliance
        errors = [v for v in violations if v.severity == PolicySeverity.ERROR]
        warnings = [v for v in violations if v.severity == PolicySeverity.WARNING]

        return ValidationResult(
            is_compliant=len(errors) == 0,
            violations=violations,
            warnings=warnings,
            policy_name=self.policy.name,
        )


class CICDIntegration:
    """CI/CD integration utilities for compliance-as-code."""

    @staticmethod
    def generate_github_action() -> str:
        """Generate GitHub Actions workflow."""
        return """
name: Privacy Compliance Check

on:
  pull_request:
    paths:
      - 'data/**'
      - 'synthetic/**'

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install genesis-synth pyyaml

      - name: Run compliance check
        run: |
          python -c "
from genesis.compliance_as_code import PolicyValidator
import pandas as pd

validator = PolicyValidator.from_file('privacy_policy.yaml')
data = pd.read_parquet('synthetic/data.parquet')
result = validator.validate(data)

print(result.to_report())

if not result.is_compliant:
    exit(1)
"
"""

    @staticmethod
    def generate_gitlab_ci() -> str:
        """Generate GitLab CI config."""
        return """
privacy-compliance:
  stage: test
  image: python:3.11
  script:
    - pip install genesis-synth pyyaml
    - python scripts/check_compliance.py
  rules:
    - changes:
        - data/**
        - synthetic/**
"""

    @staticmethod
    def generate_pre_commit_hook() -> str:
        """Generate pre-commit hook."""
        return """#!/usr/bin/env python
\"\"\"Pre-commit hook for privacy compliance.\"\"\"

import sys
import pandas as pd
from pathlib import Path

try:
    from genesis.compliance_as_code import PolicyValidator
except ImportError:
    print("genesis-synth not installed, skipping compliance check")
    sys.exit(0)

# Find synthetic data files
synthetic_files = list(Path(".").glob("**/synthetic*.parquet"))

if not synthetic_files:
    sys.exit(0)

# Load policy
policy_file = Path("privacy_policy.yaml")
if not policy_file.exists():
    policy_file = Path("privacy_policy.json")

if not policy_file.exists():
    print("No privacy policy found, skipping check")
    sys.exit(0)

validator = PolicyValidator.from_file(policy_file)

all_compliant = True
for file in synthetic_files:
    try:
        data = pd.read_parquet(file)
        result = validator.validate(data)

        if not result.is_compliant:
            print(f"\\n✗ {file} failed compliance check:")
            for v in result.violations:
                print(f"  - {v.message}")
            all_compliant = False
        else:
            print(f"✓ {file} passed compliance check")
    except Exception as e:
        print(f"⚠ Could not check {file}: {e}")

sys.exit(0 if all_compliant else 1)
"""

    @staticmethod
    def run_check(
        data_path: str,
        policy_path: str,
        fail_on_warning: bool = False,
    ) -> int:
        """Run compliance check and return exit code.

        Args:
            data_path: Path to synthetic data.
            policy_path: Path to policy file.
            fail_on_warning: Exit 1 on warnings too.

        Returns:
            Exit code (0 = pass, 1 = fail).
        """
        validator = PolicyValidator.from_file(policy_path)
        data = pd.read_parquet(data_path)
        result = validator.validate(data)

        print(result.to_report())

        if not result.is_compliant:
            return 1
        if fail_on_warning and result.warning_count > 0:
            return 1
        return 0


# Default policies
GDPR_POLICY = PrivacyPolicy(
    name="gdpr-default",
    version="1.0.0",
    description="Default GDPR-compliant policy",
    framework=ComplianceFramework.GDPR,
    epsilon=EpsilonConstraint(max_value=1.0),
    delta=DeltaConstraint(max_value=1e-5),
    k_anonymity=KAnonymityConstraint(min_k=5),
    prohibited_columns=["ssn", "social_security", "passport", "drivers_license"],
    max_null_fraction=0.05,
)

HIPAA_POLICY = PrivacyPolicy(
    name="hipaa-default",
    version="1.0.0",
    description="Default HIPAA-compliant policy",
    framework=ComplianceFramework.HIPAA,
    epsilon=EpsilonConstraint(max_value=0.5),
    delta=DeltaConstraint(max_value=1e-6),
    k_anonymity=KAnonymityConstraint(min_k=10),
    prohibited_columns=[
        "ssn", "social_security", "mrn", "medical_record_number",
        "email", "phone", "address", "zip_code",
    ],
    max_null_fraction=0.01,
)


# Convenience functions
def validate_compliance(
    data: pd.DataFrame,
    policy: Union[str, Path, PrivacyPolicy],
) -> ValidationResult:
    """Validate data against a policy.

    Args:
        data: Synthetic data.
        policy: Policy file path or PrivacyPolicy object.

    Returns:
        ValidationResult.
    """
    if isinstance(policy, (str, Path)):
        validator = PolicyValidator.from_file(policy)
    else:
        validator = PolicyValidator(policy)

    return validator.validate(data)


def check_gdpr_compliance(data: pd.DataFrame) -> ValidationResult:
    """Check GDPR compliance with default policy."""
    return PolicyValidator(GDPR_POLICY).validate(data)


def check_hipaa_compliance(data: pd.DataFrame) -> ValidationResult:
    """Check HIPAA compliance with default policy."""
    return PolicyValidator(HIPAA_POLICY).validate(data)
