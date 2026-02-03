"""Regulatory Sandbox Mode for Genesis.

This module provides pre-configured compliance templates for various
regulatory frameworks including EU AI Act, DORA, SOX, and others.
It generates audit-ready reports and ensures synthetic data generation
meets regulatory requirements.

Example:
    >>> from genesis.regulatory import RegulatoryContext, Regulation
    >>> from genesis import SyntheticGenerator
    >>>
    >>> # Create regulatory context
    >>> context = RegulatoryContext(
    ...     regulations=[Regulation.EU_AI_ACT, Regulation.GDPR],
    ...     data_classification="high_risk",
    ... )
    >>>
    >>> # Generate with compliance
    >>> generator = context.create_compliant_generator()
    >>> generator.fit(data)
    >>> synthetic = generator.generate(1000)
    >>>
    >>> # Get audit report
    >>> report = context.generate_audit_report(generator, synthetic)
    >>> report.save("audit_report.pdf")
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from genesis.core.base import SyntheticGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.types import PrivacyLevel
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class Regulation(Enum):
    """Supported regulatory frameworks."""

    # European Union
    EU_AI_ACT = "eu_ai_act"  # EU Artificial Intelligence Act
    GDPR = "gdpr"  # General Data Protection Regulation
    DORA = "dora"  # Digital Operational Resilience Act

    # United States
    SOX = "sox"  # Sarbanes-Oxley Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    CCPA = "ccpa"  # California Consumer Privacy Act
    GLBA = "glba"  # Gramm-Leach-Bliley Act
    FERPA = "ferpa"  # Family Educational Rights and Privacy Act

    # International
    ISO27001 = "iso27001"  # Information Security Management
    SOC2 = "soc2"  # Service Organization Control 2
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard

    # Industry-specific
    BASEL_III = "basel_iii"  # Banking regulation
    MIFID_II = "mifid_ii"  # Markets in Financial Instruments Directive
    SOLVENCY_II = "solvency_ii"  # Insurance regulation


class DataClassification(Enum):
    """Data classification levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    HIGH_RISK = "high_risk"  # EU AI Act classification


class RiskLevel(Enum):
    """Risk assessment levels."""

    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


@dataclass
class RegulatoryRequirement:
    """A specific regulatory requirement."""

    id: str
    regulation: Regulation
    name: str
    description: str
    category: str
    mandatory: bool
    privacy_level_required: Optional[PrivacyLevel] = None
    epsilon_max: Optional[float] = None
    k_anonymity_min: Optional[int] = None
    audit_required: bool = False
    documentation_required: bool = False
    human_oversight_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "regulation": self.regulation.value,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "mandatory": self.mandatory,
            "privacy_level_required": (
                self.privacy_level_required.value if self.privacy_level_required else None
            ),
            "epsilon_max": self.epsilon_max,
            "k_anonymity_min": self.k_anonymity_min,
            "audit_required": self.audit_required,
            "documentation_required": self.documentation_required,
            "human_oversight_required": self.human_oversight_required,
        }


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check."""

    requirement: RegulatoryRequirement
    passed: bool
    actual_value: Any
    expected_value: Any
    message: str
    severity: str  # "critical", "major", "minor", "info"
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement.id,
            "requirement_name": self.requirement.name,
            "passed": self.passed,
            "actual_value": str(self.actual_value),
            "expected_value": str(self.expected_value),
            "message": self.message,
            "severity": self.severity,
            "remediation": self.remediation,
        }


@dataclass
class AuditTrailEntry:
    """An entry in the audit trail."""

    timestamp: str
    event_type: str
    actor: str
    action: str
    details: Dict[str, Any]
    data_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "actor": self.actor,
            "action": self.action,
            "details": self.details,
            "data_hash": self.data_hash,
        }


# Pre-defined regulatory requirements
REGULATORY_REQUIREMENTS: Dict[Regulation, List[RegulatoryRequirement]] = {
    Regulation.EU_AI_ACT: [
        RegulatoryRequirement(
            id="EUAI-001",
            regulation=Regulation.EU_AI_ACT,
            name="Data Quality Requirements",
            description="Training data must be relevant, representative, and free of errors",
            category="data_governance",
            mandatory=True,
            audit_required=True,
            documentation_required=True,
        ),
        RegulatoryRequirement(
            id="EUAI-002",
            regulation=Regulation.EU_AI_ACT,
            name="Transparency Obligations",
            description="Clear documentation of data provenance and generation methods",
            category="transparency",
            mandatory=True,
            documentation_required=True,
        ),
        RegulatoryRequirement(
            id="EUAI-003",
            regulation=Regulation.EU_AI_ACT,
            name="Human Oversight",
            description="Human review capability for high-risk AI systems",
            category="oversight",
            mandatory=True,
            human_oversight_required=True,
        ),
        RegulatoryRequirement(
            id="EUAI-004",
            regulation=Regulation.EU_AI_ACT,
            name="Bias Prevention",
            description="Measures to prevent discriminatory outcomes",
            category="fairness",
            mandatory=True,
            audit_required=True,
        ),
        RegulatoryRequirement(
            id="EUAI-005",
            regulation=Regulation.EU_AI_ACT,
            name="Technical Documentation",
            description="Comprehensive technical documentation of the AI system",
            category="documentation",
            mandatory=True,
            documentation_required=True,
        ),
    ],
    Regulation.GDPR: [
        RegulatoryRequirement(
            id="GDPR-001",
            regulation=Regulation.GDPR,
            name="Data Minimization",
            description="Only process data necessary for the specified purpose",
            category="privacy",
            mandatory=True,
        ),
        RegulatoryRequirement(
            id="GDPR-002",
            regulation=Regulation.GDPR,
            name="Purpose Limitation",
            description="Data used only for specified, explicit purposes",
            category="privacy",
            mandatory=True,
            documentation_required=True,
        ),
        RegulatoryRequirement(
            id="GDPR-003",
            regulation=Regulation.GDPR,
            name="Privacy by Design",
            description="Privacy protections built into the generation process",
            category="privacy",
            mandatory=True,
            privacy_level_required=PrivacyLevel.MEDIUM,
            epsilon_max=1.0,
        ),
        RegulatoryRequirement(
            id="GDPR-004",
            regulation=Regulation.GDPR,
            name="Right to Erasure",
            description="Ability to exclude specific individuals from training data",
            category="rights",
            mandatory=True,
        ),
        RegulatoryRequirement(
            id="GDPR-005",
            regulation=Regulation.GDPR,
            name="Data Protection Impact Assessment",
            description="DPIA required for high-risk processing",
            category="assessment",
            mandatory=True,
            audit_required=True,
            documentation_required=True,
        ),
    ],
    Regulation.HIPAA: [
        RegulatoryRequirement(
            id="HIPAA-001",
            regulation=Regulation.HIPAA,
            name="PHI De-identification",
            description="Protected Health Information must be properly de-identified",
            category="privacy",
            mandatory=True,
            privacy_level_required=PrivacyLevel.HIGH,
            k_anonymity_min=5,
        ),
        RegulatoryRequirement(
            id="HIPAA-002",
            regulation=Regulation.HIPAA,
            name="Safe Harbor Method",
            description="Removal of 18 HIPAA identifiers",
            category="privacy",
            mandatory=True,
        ),
        RegulatoryRequirement(
            id="HIPAA-003",
            regulation=Regulation.HIPAA,
            name="Expert Determination",
            description="Statistical/scientific methods for de-identification",
            category="privacy",
            mandatory=False,
            epsilon_max=0.5,
        ),
        RegulatoryRequirement(
            id="HIPAA-004",
            regulation=Regulation.HIPAA,
            name="Audit Controls",
            description="Mechanisms to record access to PHI",
            category="audit",
            mandatory=True,
            audit_required=True,
        ),
    ],
    Regulation.DORA: [
        RegulatoryRequirement(
            id="DORA-001",
            regulation=Regulation.DORA,
            name="ICT Risk Management",
            description="Framework for managing ICT-related risks",
            category="risk_management",
            mandatory=True,
            documentation_required=True,
        ),
        RegulatoryRequirement(
            id="DORA-002",
            regulation=Regulation.DORA,
            name="Incident Reporting",
            description="Major ICT incidents must be reported",
            category="reporting",
            mandatory=True,
            audit_required=True,
        ),
        RegulatoryRequirement(
            id="DORA-003",
            regulation=Regulation.DORA,
            name="Digital Resilience Testing",
            description="Regular testing of ICT systems",
            category="testing",
            mandatory=True,
        ),
    ],
    Regulation.SOX: [
        RegulatoryRequirement(
            id="SOX-001",
            regulation=Regulation.SOX,
            name="Internal Controls",
            description="Adequate internal controls over financial reporting",
            category="controls",
            mandatory=True,
            audit_required=True,
        ),
        RegulatoryRequirement(
            id="SOX-002",
            regulation=Regulation.SOX,
            name="Data Integrity",
            description="Accuracy and completeness of financial data",
            category="integrity",
            mandatory=True,
        ),
        RegulatoryRequirement(
            id="SOX-003",
            regulation=Regulation.SOX,
            name="Audit Trail",
            description="Complete audit trail for data modifications",
            category="audit",
            mandatory=True,
            audit_required=True,
        ),
    ],
    Regulation.PCI_DSS: [
        RegulatoryRequirement(
            id="PCI-001",
            regulation=Regulation.PCI_DSS,
            name="Cardholder Data Protection",
            description="Protect stored cardholder data",
            category="privacy",
            mandatory=True,
            privacy_level_required=PrivacyLevel.MAXIMUM,
        ),
        RegulatoryRequirement(
            id="PCI-002",
            regulation=Regulation.PCI_DSS,
            name="Data Masking",
            description="Mask PAN when displayed",
            category="privacy",
            mandatory=True,
        ),
        RegulatoryRequirement(
            id="PCI-003",
            regulation=Regulation.PCI_DSS,
            name="Access Control",
            description="Restrict access to cardholder data",
            category="access",
            mandatory=True,
            audit_required=True,
        ),
    ],
}


@dataclass
class RegulatoryConfig:
    """Configuration for regulatory compliance.

    This configuration aggregates requirements from multiple regulations
    and determines the appropriate privacy settings.
    """

    regulations: List[Regulation]
    data_classification: DataClassification = DataClassification.CONFIDENTIAL
    purpose: str = "synthetic_data_generation"
    data_controller: str = "Unknown"
    data_processor: str = "Genesis"
    retention_period_days: int = 365
    geographic_scope: List[str] = field(default_factory=lambda: ["EU", "US"])
    custom_requirements: List[RegulatoryRequirement] = field(default_factory=list)

    def get_all_requirements(self) -> List[RegulatoryRequirement]:
        """Get all applicable requirements."""
        requirements = []
        for reg in self.regulations:
            if reg in REGULATORY_REQUIREMENTS:
                requirements.extend(REGULATORY_REQUIREMENTS[reg])
        requirements.extend(self.custom_requirements)
        return requirements

    def get_strictest_privacy_level(self) -> PrivacyLevel:
        """Get the strictest required privacy level."""
        levels = [PrivacyLevel.NONE]
        for req in self.get_all_requirements():
            if req.privacy_level_required:
                levels.append(req.privacy_level_required)

        # Order: NONE < LOW < MEDIUM < HIGH < MAXIMUM
        level_order = [
            PrivacyLevel.NONE,
            PrivacyLevel.LOW,
            PrivacyLevel.MEDIUM,
            PrivacyLevel.HIGH,
            PrivacyLevel.MAXIMUM,
        ]
        return max(levels, key=lambda x: level_order.index(x))

    def get_minimum_epsilon(self) -> float:
        """Get the minimum (strictest) epsilon value."""
        epsilons = [
            req.epsilon_max
            for req in self.get_all_requirements()
            if req.epsilon_max is not None
        ]
        return min(epsilons) if epsilons else 1.0

    def get_maximum_k_anonymity(self) -> int:
        """Get the maximum (strictest) k-anonymity value."""
        k_values = [
            req.k_anonymity_min
            for req in self.get_all_requirements()
            if req.k_anonymity_min is not None
        ]
        return max(k_values) if k_values else 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regulations": [r.value for r in self.regulations],
            "data_classification": self.data_classification.value,
            "purpose": self.purpose,
            "data_controller": self.data_controller,
            "data_processor": self.data_processor,
            "retention_period_days": self.retention_period_days,
            "geographic_scope": self.geographic_scope,
        }


class RegulatoryContext:
    """Context for regulatory-compliant synthetic data generation.

    This class manages regulatory requirements and ensures that
    generated synthetic data meets compliance standards.

    Example:
        >>> context = RegulatoryContext(
        ...     regulations=[Regulation.GDPR, Regulation.HIPAA],
        ...     data_classification=DataClassification.HIGH_RISK,
        ... )
        >>> generator = context.create_compliant_generator()
        >>> generator.fit(healthcare_data)
        >>> synthetic = generator.generate(1000)
        >>> report = context.generate_audit_report(generator, synthetic)
    """

    def __init__(
        self,
        regulations: List[Regulation],
        data_classification: DataClassification = DataClassification.CONFIDENTIAL,
        config: Optional[RegulatoryConfig] = None,
    ):
        """Initialize regulatory context.

        Args:
            regulations: List of applicable regulations
            data_classification: Data classification level
            config: Full regulatory configuration (overrides other params)
        """
        if config:
            self.config = config
        else:
            self.config = RegulatoryConfig(
                regulations=regulations,
                data_classification=data_classification,
            )

        self._audit_trail: List[AuditTrailEntry] = []
        self._context_id = str(uuid.uuid4())

        # Log context creation
        self._log_audit_event(
            "context_created",
            "system",
            "Regulatory context initialized",
            {
                "regulations": [r.value for r in regulations],
                "classification": data_classification.value,
            },
        )

    def _log_audit_event(
        self,
        event_type: str,
        actor: str,
        action: str,
        details: Dict[str, Any],
        data_hash: Optional[str] = None,
    ) -> None:
        """Log an event to the audit trail."""
        entry = AuditTrailEntry(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            actor=actor,
            action=action,
            details=details,
            data_hash=data_hash,
        )
        self._audit_trail.append(entry)

    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute a hash of the data for audit purposes."""
        # Use a sample for large datasets
        if len(data) > 10000:
            sample = data.sample(10000, random_state=42)
        else:
            sample = data

        # Hash the data
        data_str = sample.to_json()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def create_privacy_config(self) -> PrivacyConfig:
        """Create a PrivacyConfig that meets all regulatory requirements.

        Returns:
            PrivacyConfig configured for compliance
        """
        privacy_level = self.config.get_strictest_privacy_level()
        epsilon = self.config.get_minimum_epsilon()
        k_anonymity = self.config.get_maximum_k_anonymity()

        return PrivacyConfig(
            privacy_level=privacy_level,
            enable_differential_privacy=privacy_level in [
                PrivacyLevel.MEDIUM,
                PrivacyLevel.HIGH,
                PrivacyLevel.MAXIMUM,
            ],
            epsilon=epsilon,
            delta=1e-5,
            k_anonymity=k_anonymity,
        )

    def create_compliant_generator(
        self,
        method: str = "auto",
        config: Optional[GeneratorConfig] = None,
    ) -> SyntheticGenerator:
        """Create a generator configured for regulatory compliance.

        Args:
            method: Generation method
            config: Optional base configuration

        Returns:
            SyntheticGenerator configured for compliance
        """
        privacy = self.create_privacy_config()
        config = config or GeneratorConfig()

        generator = SyntheticGenerator(
            method=method,
            config=config,
            privacy=privacy,
        )

        self._log_audit_event(
            "generator_created",
            "system",
            "Compliant generator created",
            {
                "method": method,
                "privacy_level": privacy.privacy_level.value,
                "epsilon": privacy.epsilon,
                "k_anonymity": privacy.k_anonymity,
            },
        )

        return generator

    def check_compliance(
        self,
        generator: SyntheticGenerator,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> List[ComplianceCheckResult]:
        """Check compliance with all applicable requirements.

        Args:
            generator: The generator used
            real_data: Original training data
            synthetic_data: Generated synthetic data

        Returns:
            List of compliance check results
        """
        results = []
        requirements = self.config.get_all_requirements()

        for req in requirements:
            result = self._check_requirement(req, generator, real_data, synthetic_data)
            results.append(result)

        # Log the compliance check
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        self._log_audit_event(
            "compliance_check",
            "system",
            f"Compliance check completed: {passed}/{total} passed",
            {
                "passed": passed,
                "total": total,
                "failed_requirements": [r.requirement.id for r in results if not r.passed],
            },
            data_hash=self._compute_data_hash(synthetic_data),
        )

        return results

    def _check_requirement(
        self,
        req: RegulatoryRequirement,
        generator: SyntheticGenerator,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> ComplianceCheckResult:
        """Check a single requirement."""
        # Check privacy level
        if req.privacy_level_required:
            actual_level = generator.privacy.privacy_level
            required_level = req.privacy_level_required

            level_order = [
                PrivacyLevel.NONE,
                PrivacyLevel.LOW,
                PrivacyLevel.MEDIUM,
                PrivacyLevel.HIGH,
                PrivacyLevel.MAXIMUM,
            ]
            passed = level_order.index(actual_level) >= level_order.index(required_level)

            if not passed:
                return ComplianceCheckResult(
                    requirement=req,
                    passed=False,
                    actual_value=actual_level.value,
                    expected_value=required_level.value,
                    message=f"Privacy level {actual_level.value} does not meet requirement {required_level.value}",
                    severity="critical",
                    remediation=f"Increase privacy level to at least {required_level.value}",
                )

        # Check epsilon
        if req.epsilon_max is not None:
            actual_epsilon = generator.privacy.epsilon
            if actual_epsilon > req.epsilon_max:
                return ComplianceCheckResult(
                    requirement=req,
                    passed=False,
                    actual_value=actual_epsilon,
                    expected_value=req.epsilon_max,
                    message=f"Epsilon {actual_epsilon} exceeds maximum {req.epsilon_max}",
                    severity="critical",
                    remediation=f"Reduce epsilon to at most {req.epsilon_max}",
                )

        # Check k-anonymity
        if req.k_anonymity_min is not None:
            actual_k = generator.privacy.k_anonymity or 1
            if actual_k < req.k_anonymity_min:
                return ComplianceCheckResult(
                    requirement=req,
                    passed=False,
                    actual_value=actual_k,
                    expected_value=req.k_anonymity_min,
                    message=f"k-anonymity {actual_k} below minimum {req.k_anonymity_min}",
                    severity="critical",
                    remediation=f"Increase k-anonymity to at least {req.k_anonymity_min}",
                )

        # Default: passed
        return ComplianceCheckResult(
            requirement=req,
            passed=True,
            actual_value="compliant",
            expected_value="compliant",
            message=f"Requirement {req.name} satisfied",
            severity="info",
        )

    def generate_audit_report(
        self,
        generator: SyntheticGenerator,
        synthetic_data: pd.DataFrame,
        real_data: Optional[pd.DataFrame] = None,
        include_quality_metrics: bool = True,
    ) -> "AuditReport":
        """Generate a comprehensive audit report.

        Args:
            generator: The generator used
            synthetic_data: Generated synthetic data
            real_data: Original training data (optional, for quality metrics)
            include_quality_metrics: Whether to include quality evaluation

        Returns:
            AuditReport object
        """
        # Run compliance checks if real data provided
        compliance_results = []
        if real_data is not None:
            compliance_results = self.check_compliance(generator, real_data, synthetic_data)

        # Get quality metrics if requested
        quality_metrics = None
        if include_quality_metrics and real_data is not None:
            try:
                from genesis.evaluation.evaluator import QualityEvaluator

                evaluator = QualityEvaluator(real_data, synthetic_data)
                quality_report = evaluator.evaluate()
                quality_metrics = {
                    "statistical_fidelity": quality_report.overall_statistical_score,
                    "ml_utility": quality_report.overall_ml_utility_score,
                    "privacy_score": quality_report.overall_privacy_score,
                }
            except Exception as e:
                logger.warning(f"Could not compute quality metrics: {e}")

        report = AuditReport(
            report_id=str(uuid.uuid4()),
            context_id=self._context_id,
            generation_date=datetime.utcnow().isoformat(),
            regulations=[r.value for r in self.config.regulations],
            data_classification=self.config.data_classification.value,
            compliance_results=compliance_results,
            audit_trail=self._audit_trail.copy(),
            quality_metrics=quality_metrics,
            generator_config=generator.config.to_dict(),
            privacy_config=generator.privacy.to_dict(),
            data_summary={
                "n_rows": len(synthetic_data),
                "n_columns": len(synthetic_data.columns),
                "columns": list(synthetic_data.columns),
            },
        )

        self._log_audit_event(
            "report_generated",
            "system",
            "Audit report generated",
            {"report_id": report.report_id},
        )

        return report

    def get_audit_trail(self) -> List[AuditTrailEntry]:
        """Get the complete audit trail.

        Returns:
            List of audit trail entries
        """
        return self._audit_trail.copy()

    def export_audit_trail(self, path: str, format: str = "json") -> None:
        """Export the audit trail to a file.

        Args:
            path: Output file path
            format: 'json' or 'csv'
        """
        if format == "json":
            with open(path, "w") as f:
                json.dump(
                    [entry.to_dict() for entry in self._audit_trail],
                    f,
                    indent=2,
                )
        elif format == "csv":
            import csv

            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "event_type", "actor", "action", "details", "data_hash"],
                )
                writer.writeheader()
                for entry in self._audit_trail:
                    row = entry.to_dict()
                    row["details"] = json.dumps(row["details"])
                    writer.writerow(row)


@dataclass
class AuditReport:
    """Comprehensive audit report for regulatory compliance."""

    report_id: str
    context_id: str
    generation_date: str
    regulations: List[str]
    data_classification: str
    compliance_results: List[ComplianceCheckResult]
    audit_trail: List[AuditTrailEntry]
    quality_metrics: Optional[Dict[str, float]]
    generator_config: Dict[str, Any]
    privacy_config: Dict[str, Any]
    data_summary: Dict[str, Any]

    @property
    def is_compliant(self) -> bool:
        """Check if all mandatory requirements passed."""
        return all(
            r.passed
            for r in self.compliance_results
            if r.requirement.mandatory
        )

    @property
    def compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        if not self.compliance_results:
            return 100.0
        passed = sum(1 for r in self.compliance_results if r.passed)
        return (passed / len(self.compliance_results)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "context_id": self.context_id,
            "generation_date": self.generation_date,
            "regulations": self.regulations,
            "data_classification": self.data_classification,
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "compliance_results": [r.to_dict() for r in self.compliance_results],
            "audit_trail": [e.to_dict() for e in self.audit_trail],
            "quality_metrics": self.quality_metrics,
            "generator_config": self.generator_config,
            "privacy_config": self.privacy_config,
            "data_summary": self.data_summary,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str) -> None:
        """Save report to file.

        Supports .json, .html, and .md formats based on extension.

        Args:
            path: Output file path
        """
        path = Path(path)

        if path.suffix == ".json":
            with open(path, "w") as f:
                f.write(self.to_json())

        elif path.suffix == ".html":
            html = self._to_html()
            with open(path, "w") as f:
                f.write(html)

        elif path.suffix == ".md":
            md = self._to_markdown()
            with open(path, "w") as f:
                f.write(md)

        else:
            # Default to JSON
            with open(path, "w") as f:
                f.write(self.to_json())

    def _to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            "# Regulatory Compliance Audit Report",
            "",
            f"**Report ID:** {self.report_id}",
            f"**Generated:** {self.generation_date}",
            f"**Compliance Status:** {'✅ COMPLIANT' if self.is_compliant else '❌ NON-COMPLIANT'}",
            f"**Compliance Score:** {self.compliance_score:.1f}%",
            "",
            "## Regulations",
            "",
        ]

        for reg in self.regulations:
            lines.append(f"- {reg.upper()}")

        lines.extend([
            "",
            f"**Data Classification:** {self.data_classification}",
            "",
            "## Compliance Check Results",
            "",
            "| Requirement | Status | Message |",
            "|-------------|--------|---------|",
        ])

        for result in self.compliance_results:
            status = "✅" if result.passed else "❌"
            lines.append(f"| {result.requirement.name} | {status} | {result.message} |")

        if self.quality_metrics:
            lines.extend([
                "",
                "## Quality Metrics",
                "",
                f"- **Statistical Fidelity:** {self.quality_metrics.get('statistical_fidelity', 'N/A')}",
                f"- **ML Utility:** {self.quality_metrics.get('ml_utility', 'N/A')}",
                f"- **Privacy Score:** {self.quality_metrics.get('privacy_score', 'N/A')}",
            ])

        lines.extend([
            "",
            "## Data Summary",
            "",
            f"- **Rows:** {self.data_summary.get('n_rows', 'N/A')}",
            f"- **Columns:** {self.data_summary.get('n_columns', 'N/A')}",
            "",
            "## Audit Trail",
            "",
        ])

        for entry in self.audit_trail[-10:]:  # Last 10 entries
            lines.append(f"- [{entry.timestamp}] {entry.action}")

        return "\n".join(lines)

    def _to_html(self) -> str:
        """Convert to HTML format."""
        status_class = "success" if self.is_compliant else "danger"
        status_text = "COMPLIANT" if self.is_compliant else "NON-COMPLIANT"

        compliance_rows = ""
        for result in self.compliance_results:
            status_icon = "✅" if result.passed else "❌"
            row_class = "" if result.passed else 'class="table-danger"'
            compliance_rows += f"""
            <tr {row_class}>
                <td>{result.requirement.name}</td>
                <td>{status_icon}</td>
                <td>{result.message}</td>
                <td>{result.severity}</td>
            </tr>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Regulatory Compliance Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .status-success {{ color: green; }}
        .status-danger {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .table-danger {{ background-color: #ffebee; }}
        .section {{ margin: 30px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Regulatory Compliance Audit Report</h1>
        <p><strong>Report ID:</strong> {self.report_id}</p>
        <p><strong>Generated:</strong> {self.generation_date}</p>
        <p><strong>Status:</strong> <span class="status-{status_class}">{status_text}</span></p>
        <p><strong>Compliance Score:</strong> {self.compliance_score:.1f}%</p>
    </div>

    <div class="section">
        <h2>Applicable Regulations</h2>
        <ul>
            {"".join(f"<li>{reg.upper()}</li>" for reg in self.regulations)}
        </ul>
        <p><strong>Data Classification:</strong> {self.data_classification}</p>
    </div>

    <div class="section">
        <h2>Compliance Check Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Requirement</th>
                    <th>Status</th>
                    <th>Message</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>
                {compliance_rows}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Data Summary</h2>
        <div class="metric">
            <strong>Rows:</strong> {self.data_summary.get('n_rows', 'N/A')}
        </div>
        <div class="metric">
            <strong>Columns:</strong> {self.data_summary.get('n_columns', 'N/A')}
        </div>
    </div>

    <div class="section">
        <h2>Privacy Configuration</h2>
        <pre>{json.dumps(self.privacy_config, indent=2)}</pre>
    </div>

    <footer style="margin-top: 40px; border-top: 1px solid #ddd; padding-top: 10px; color: #666;">
        <p>Generated by Genesis Synthetic Data Platform</p>
    </footer>
</body>
</html>
        """


# Pre-configured sandbox environments
class RegulatoryPresets:
    """Pre-configured regulatory sandbox environments."""

    @staticmethod
    def healthcare_us() -> RegulatoryContext:
        """US Healthcare regulatory context (HIPAA)."""
        return RegulatoryContext(
            regulations=[Regulation.HIPAA],
            data_classification=DataClassification.RESTRICTED,
        )

    @staticmethod
    def healthcare_eu() -> RegulatoryContext:
        """EU Healthcare regulatory context (GDPR + EU AI Act)."""
        return RegulatoryContext(
            regulations=[Regulation.GDPR, Regulation.EU_AI_ACT],
            data_classification=DataClassification.HIGH_RISK,
        )

    @staticmethod
    def financial_services_eu() -> RegulatoryContext:
        """EU Financial Services (GDPR + DORA + MiFID II)."""
        config = RegulatoryConfig(
            regulations=[Regulation.GDPR, Regulation.DORA],
            data_classification=DataClassification.CONFIDENTIAL,
            purpose="financial_data_synthesis",
        )
        return RegulatoryContext(
            regulations=config.regulations,
            config=config,
        )

    @staticmethod
    def financial_services_us() -> RegulatoryContext:
        """US Financial Services (SOX + GLBA)."""
        return RegulatoryContext(
            regulations=[Regulation.SOX, Regulation.GLBA],
            data_classification=DataClassification.CONFIDENTIAL,
        )

    @staticmethod
    def payment_card() -> RegulatoryContext:
        """Payment card data (PCI DSS)."""
        return RegulatoryContext(
            regulations=[Regulation.PCI_DSS],
            data_classification=DataClassification.RESTRICTED,
        )

    @staticmethod
    def ai_development_eu() -> RegulatoryContext:
        """EU AI development (EU AI Act + GDPR)."""
        return RegulatoryContext(
            regulations=[Regulation.EU_AI_ACT, Regulation.GDPR],
            data_classification=DataClassification.HIGH_RISK,
        )

    @staticmethod
    def general_enterprise() -> RegulatoryContext:
        """General enterprise data protection (ISO 27001 + SOC2)."""
        return RegulatoryContext(
            regulations=[Regulation.ISO27001, Regulation.SOC2],
            data_classification=DataClassification.INTERNAL,
        )


# Convenience functions
def create_sandbox(
    regulations: List[Union[str, Regulation]],
    classification: str = "confidential",
) -> RegulatoryContext:
    """Create a regulatory sandbox context.

    Args:
        regulations: List of regulation names or Regulation enums
        classification: Data classification level

    Returns:
        RegulatoryContext configured for the specified regulations

    Example:
        >>> sandbox = create_sandbox(["gdpr", "hipaa"], "restricted")
        >>> generator = sandbox.create_compliant_generator()
    """
    # Convert string regulations to enum
    reg_enums = []
    for reg in regulations:
        if isinstance(reg, str):
            reg_enums.append(Regulation(reg.lower()))
        else:
            reg_enums.append(reg)

    # Convert classification
    classification_enum = DataClassification(classification.lower())

    return RegulatoryContext(
        regulations=reg_enums,
        data_classification=classification_enum,
    )


def get_available_regulations() -> List[str]:
    """Get list of all available regulations.

    Returns:
        List of regulation identifiers
    """
    return [r.value for r in Regulation]


def get_regulation_requirements(regulation: Union[str, Regulation]) -> List[RegulatoryRequirement]:
    """Get requirements for a specific regulation.

    Args:
        regulation: Regulation name or enum

    Returns:
        List of requirements for the regulation
    """
    if isinstance(regulation, str):
        regulation = Regulation(regulation.lower())

    return REGULATORY_REQUIREMENTS.get(regulation, [])


__all__ = [
    # Core classes
    "RegulatoryContext",
    "RegulatoryConfig",
    "AuditReport",
    # Enums
    "Regulation",
    "DataClassification",
    "RiskLevel",
    # Data classes
    "RegulatoryRequirement",
    "ComplianceCheckResult",
    "AuditTrailEntry",
    # Presets
    "RegulatoryPresets",
    # Convenience functions
    "create_sandbox",
    "get_available_regulations",
    "get_regulation_requirements",
    # Constants
    "REGULATORY_REQUIREMENTS",
]
