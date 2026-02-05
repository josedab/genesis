"""Tests for Compliance-as-Code module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genesis.compliance_as_code import (
    CICDIntegration,
    ComplianceFramework,
    DeltaConstraint,
    EpsilonConstraint,
    GDPR_POLICY,
    HIPAA_POLICY,
    KAnonymityConstraint,
    PolicyLoader,
    PolicySeverity,
    PolicyValidator,
    PolicyViolation,
    PrivacyPolicy,
    ValidationResult,
    check_gdpr_compliance,
    check_hipaa_compliance,
    validate_compliance,
)


class TestEpsilonConstraint:
    """Tests for EpsilonConstraint."""

    def test_validate_within_range(self):
        """Test epsilon within valid range."""
        constraint = EpsilonConstraint(min_value=0.1, max_value=1.0)
        passed, msg = constraint.validate(0.5)
        assert passed
        assert msg is None

    def test_validate_below_minimum(self):
        """Test epsilon below minimum."""
        constraint = EpsilonConstraint(min_value=0.1)
        passed, msg = constraint.validate(0.05)
        assert not passed
        assert "below minimum" in msg.lower()

    def test_validate_above_maximum(self):
        """Test epsilon above maximum."""
        constraint = EpsilonConstraint(max_value=1.0)
        passed, msg = constraint.validate(2.0)
        assert not passed
        assert "above maximum" in msg.lower()


class TestDeltaConstraint:
    """Tests for DeltaConstraint."""

    def test_validate_within_range(self):
        """Test delta within valid range."""
        constraint = DeltaConstraint(max_value=1e-5)
        passed, msg = constraint.validate(1e-6)
        assert passed

    def test_validate_above_maximum(self):
        """Test delta above maximum."""
        constraint = DeltaConstraint(max_value=1e-5)
        passed, msg = constraint.validate(1e-4)
        assert not passed


class TestKAnonymityConstraint:
    """Tests for KAnonymityConstraint."""

    def test_validate_satisfies_k(self):
        """Test data satisfying k-anonymity."""
        data = pd.DataFrame({
            "age_group": ["20-30"] * 10 + ["30-40"] * 10,
            "gender": ["M"] * 10 + ["F"] * 10,
        })

        constraint = KAnonymityConstraint(
            min_k=5,
            quasi_identifiers=["age_group", "gender"],
        )

        passed, msg = constraint.validate(data)
        assert passed

    def test_validate_violates_k(self):
        """Test data violating k-anonymity."""
        data = pd.DataFrame({
            "age_group": ["20-30"] * 10 + ["30-40"] * 2,
            "gender": ["M"] * 10 + ["F"] * 2,
        })

        constraint = KAnonymityConstraint(
            min_k=5,
            quasi_identifiers=["age_group", "gender"],
        )

        passed, msg = constraint.validate(data)
        assert not passed
        assert "violated" in msg.lower()


class TestPrivacyPolicy:
    """Tests for PrivacyPolicy."""

    def test_to_yaml(self):
        """Test policy YAML serialization."""
        policy = PrivacyPolicy(
            name="test-policy",
            version="1.0.0",
            epsilon=EpsilonConstraint(max_value=1.0),
            prohibited_columns=["ssn"],
        )

        yaml_str = policy.to_yaml()
        assert "test-policy" in yaml_str
        assert "ssn" in yaml_str

    def test_to_dict(self):
        """Test policy dictionary conversion."""
        policy = PrivacyPolicy(
            name="test-policy",
            framework=ComplianceFramework.GDPR,
        )

        data = policy.to_dict()
        assert data["metadata"]["name"] == "test-policy"
        assert data["spec"]["framework"] == "gdpr"


class TestPolicyLoader:
    """Tests for PolicyLoader."""

    def test_from_dict(self):
        """Test loading policy from dictionary."""
        data = {
            "metadata": {"name": "test"},
            "spec": {
                "epsilon": {"max": 1.0},
                "prohibited_columns": ["password"],
            },
        }

        policy = PolicyLoader.from_dict(data)
        assert policy.name == "test"
        assert policy.epsilon.max_value == 1.0
        assert "password" in policy.prohibited_columns

    def test_from_file_json(self):
        """Test loading policy from JSON file."""
        data = {
            "metadata": {"name": "file-test"},
            "spec": {"min_rows": 100},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            policy = PolicyLoader.from_file(f.name)
            assert policy.name == "file-test"
            assert policy.min_rows == 100

            Path(f.name).unlink()


class TestPolicyValidator:
    """Tests for PolicyValidator."""

    def test_validate_compliant_data(self):
        """Test validation of compliant data."""
        policy = PrivacyPolicy(
            name="test",
            prohibited_columns=["ssn"],
            min_rows=10,
        )

        data = pd.DataFrame({
            "age": range(100),
            "income": range(100),
        })

        validator = PolicyValidator(policy)
        result = validator.validate(data)

        assert result.is_compliant
        assert result.error_count == 0

    def test_validate_prohibited_column(self):
        """Test validation catches prohibited columns."""
        policy = PrivacyPolicy(
            name="test",
            prohibited_columns=["ssn", "credit_card"],
        )

        data = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "ssn": ["123-45-6789", "987-65-4321"],
        })

        validator = PolicyValidator(policy)
        result = validator.validate(data)

        assert not result.is_compliant
        assert result.error_count >= 1
        assert any("ssn" in v.message.lower() for v in result.violations)

    def test_validate_missing_required_column(self):
        """Test validation catches missing required columns."""
        policy = PrivacyPolicy(
            name="test",
            required_columns=["id", "timestamp"],
        )

        data = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        })

        validator = PolicyValidator(policy)
        result = validator.validate(data)

        assert not result.is_compliant
        assert any("timestamp" in v.message.lower() for v in result.violations)

    def test_validate_insufficient_rows(self):
        """Test validation catches insufficient rows."""
        policy = PrivacyPolicy(
            name="test",
            min_rows=100,
        )

        data = pd.DataFrame({
            "col": range(50),
        })

        validator = PolicyValidator(policy)
        result = validator.validate(data)

        assert not result.is_compliant

    def test_validate_high_null_fraction(self):
        """Test validation catches high null fraction."""
        policy = PrivacyPolicy(
            name="test",
            max_null_fraction=0.05,
        )

        data = pd.DataFrame({
            "col1": [1, None, None, None, 5],
            "col2": [1, 2, None, 4, 5],
        })

        validator = PolicyValidator(policy)
        result = validator.validate(data)

        # Should be a warning
        assert result.warning_count >= 1

    def test_register_custom_rule(self):
        """Test registering custom validation rule."""
        policy = PrivacyPolicy(name="test")
        validator = PolicyValidator(policy)

        def check_positive(data: pd.DataFrame):
            for col in data.select_dtypes(include=[np.number]).columns:
                if (data[col] < 0).any():
                    return False, f"Column {col} has negative values"
            return True, None

        validator.register_custom_rule("positive_values", check_positive)

        data = pd.DataFrame({
            "values": [-1, 2, 3, 4, 5],
        })

        result = validator.validate(data)
        assert not result.is_compliant
        assert any("negative" in v.message.lower() for v in result.violations)

    def test_from_file(self):
        """Test creating validator from file."""
        data = {
            "metadata": {"name": "file-validator"},
            "spec": {"min_rows": 10},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            validator = PolicyValidator.from_file(f.name)
            assert validator.policy.name == "file-validator"

            Path(f.name).unlink()


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_to_report(self):
        """Test generating human-readable report."""
        result = ValidationResult(
            is_compliant=False,
            violations=[
                PolicyViolation(
                    rule="prohibited_columns",
                    severity=PolicySeverity.ERROR,
                    message="Found SSN column",
                ),
            ],
            policy_name="test-policy",
        )

        report = result.to_report()
        assert "NON-COMPLIANT" in report
        assert "SSN" in report

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ValidationResult(
            is_compliant=True,
            policy_name="test",
        )

        data = result.to_dict()
        assert data["is_compliant"]
        assert data["policy_name"] == "test"


class TestCICDIntegration:
    """Tests for CI/CD integration utilities."""

    def test_generate_github_action(self):
        """Test GitHub Actions workflow generation."""
        workflow = CICDIntegration.generate_github_action()

        assert "name:" in workflow
        assert "privacy_policy.yaml" in workflow
        assert "pip install" in workflow

    def test_generate_gitlab_ci(self):
        """Test GitLab CI config generation."""
        config = CICDIntegration.generate_gitlab_ci()

        assert "privacy-compliance:" in config
        assert "pip install" in config

    def test_generate_pre_commit_hook(self):
        """Test pre-commit hook generation."""
        hook = CICDIntegration.generate_pre_commit_hook()

        assert "#!/usr/bin/env python" in hook
        assert "PolicyValidator" in hook


class TestDefaultPolicies:
    """Tests for built-in default policies."""

    def test_gdpr_policy(self):
        """Test GDPR default policy."""
        assert GDPR_POLICY.name == "gdpr-default"
        assert GDPR_POLICY.framework == ComplianceFramework.GDPR
        assert GDPR_POLICY.epsilon.max_value == 1.0
        assert "ssn" in GDPR_POLICY.prohibited_columns

    def test_hipaa_policy(self):
        """Test HIPAA default policy."""
        assert HIPAA_POLICY.name == "hipaa-default"
        assert HIPAA_POLICY.framework == ComplianceFramework.HIPAA
        assert HIPAA_POLICY.epsilon.max_value == 0.5
        assert HIPAA_POLICY.k_anonymity.min_k == 10


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_gdpr_compliance(self):
        """Test GDPR compliance check."""
        data = pd.DataFrame({
            "age": range(100),
            "income": range(100),
        })

        result = check_gdpr_compliance(data)
        assert result.policy_name == "gdpr-default"

    def test_check_hipaa_compliance(self):
        """Test HIPAA compliance check."""
        data = pd.DataFrame({
            "patient_id": range(100),
            "diagnosis_code": ["A00"] * 100,
        })

        result = check_hipaa_compliance(data)
        assert result.policy_name == "hipaa-default"

    def test_validate_compliance_with_policy_object(self):
        """Test validate_compliance with PrivacyPolicy."""
        policy = PrivacyPolicy(name="custom")
        data = pd.DataFrame({"col": range(10)})

        result = validate_compliance(data, policy)
        assert result.policy_name == "custom"
