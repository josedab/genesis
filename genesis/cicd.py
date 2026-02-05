"""Synthetic Data CI/CD Integration.

GitHub Actions and GitLab CI integration for automated synthetic data
generation, validation, and deployment in CI/CD pipelines.

Features:
    - GitHub Actions reusable workflow
    - GitLab CI template
    - Schema drift detection and auto-regeneration
    - Quality gate enforcement
    - Artifact versioning and caching
    - Secrets management integration

Example:
    GitHub Actions workflow::

        # .github/workflows/synthetic-data.yml
        name: Synthetic Data Pipeline
        on:
          push:
            paths:
              - 'schemas/**'
              - 'config/synthetic.yaml'

        jobs:
          generate:
            uses: genesis-synth/genesis/.github/workflows/generate.yml@main
            with:
              schema-path: schemas/customer.yaml
              output-path: data/synthetic/customers.parquet
              n-samples: 10000
            secrets:
              GENESIS_LICENSE_KEY: ${{ secrets.GENESIS_LICENSE_KEY }}

    Programmatic CI integration::

        from genesis.cicd import SyntheticDataPipeline, CIConfig

        pipeline = SyntheticDataPipeline(
            config=CIConfig(
                schema_dir="schemas/",
                output_dir="data/synthetic/",
                quality_threshold=0.85,
            )
        )
        
        # Run in CI
        result = pipeline.run()
        if not result.passed:
            sys.exit(1)

Classes:
    CIConfig: Configuration for CI/CD pipeline.
    SchemaDriftDetector: Detects schema changes.
    QualityGate: Enforces quality thresholds.
    ArtifactManager: Manages synthetic data artifacts.
    SyntheticDataPipeline: Main CI/CD pipeline.
    GitHubActionsIntegration: GitHub-specific integration.
    GitLabCIIntegration: GitLab-specific integration.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class CIProvider(str, Enum):
    """Supported CI/CD providers."""

    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"
    LOCAL = "local"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CIConfig:
    """Configuration for CI/CD synthetic data pipeline.

    Attributes:
        schema_dir: Directory containing schema definitions
        output_dir: Directory for generated data
        cache_dir: Directory for caching models
        quality_threshold: Minimum quality score to pass (0-1)
        privacy_threshold: Minimum privacy score to pass (0-1)
        auto_regenerate: Regenerate on schema drift
        fail_on_drift: Fail pipeline if schema drifts
        n_samples_default: Default number of samples
        parallel_jobs: Number of parallel generation jobs
        timeout_seconds: Pipeline timeout
        artifact_retention_days: How long to keep artifacts
        enable_caching: Cache fitted models
        notify_on_failure: Send notifications on failure
        notification_webhook: Webhook URL for notifications
    """

    schema_dir: str = "schemas/"
    output_dir: str = "data/synthetic/"
    cache_dir: str = ".genesis-cache/"
    quality_threshold: float = 0.85
    privacy_threshold: float = 0.95
    auto_regenerate: bool = True
    fail_on_drift: bool = False
    n_samples_default: int = 1000
    parallel_jobs: int = 4
    timeout_seconds: int = 3600
    artifact_retention_days: int = 30
    enable_caching: bool = True
    notify_on_failure: bool = False
    notification_webhook: Optional[str] = None
    git_commit_results: bool = False
    git_branch: str = "synthetic-data"


@dataclass
class SchemaChange:
    """Represents a schema change."""

    file_path: str
    change_type: str  # added, removed, modified
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    columns_added: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    columns_modified: List[str] = field(default_factory=list)


@dataclass
class QualityGateResult:
    """Result of quality gate check."""

    passed: bool
    quality_score: float
    privacy_score: float
    threshold_quality: float
    threshold_privacy: float
    details: Dict[str, Any] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result from CI/CD pipeline execution."""

    status: PipelineStatus
    passed: bool
    duration_seconds: float
    schemas_processed: int
    files_generated: List[str]
    quality_results: Dict[str, QualityGateResult]
    schema_changes: List[SchemaChange]
    artifacts: List[str]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchemaDriftDetector:
    """Detects changes in schema definitions.

    Compares current schemas against cached baseline to detect
    additions, removals, and modifications.
    """

    def __init__(self, schema_dir: str, cache_dir: str):
        """Initialize detector.

        Args:
            schema_dir: Directory containing schema files
            cache_dir: Directory for cached schema hashes
        """
        self._schema_dir = Path(schema_dir)
        self._cache_dir = Path(cache_dir)
        self._hash_file = self._cache_dir / "schema_hashes.json"

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _load_cached_hashes(self) -> Dict[str, str]:
        """Load cached schema hashes."""
        if self._hash_file.exists():
            with open(self._hash_file) as f:
                return json.load(f)
        return {}

    def _save_hashes(self, hashes: Dict[str, str]) -> None:
        """Save schema hashes to cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._hash_file, "w") as f:
            json.dump(hashes, f, indent=2)

    def _parse_schema(self, file_path: Path) -> Dict[str, Any]:
        """Parse schema file (YAML or JSON)."""
        with open(file_path) as f:
            if file_path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
            return json.load(f)

    def detect_changes(self) -> List[SchemaChange]:
        """Detect schema changes since last run.

        Returns:
            List of schema changes
        """
        changes: List[SchemaChange] = []
        cached_hashes = self._load_cached_hashes()
        current_hashes: Dict[str, str] = {}

        # Find all schema files
        schema_files = list(self._schema_dir.glob("**/*.yaml"))
        schema_files.extend(self._schema_dir.glob("**/*.yml"))
        schema_files.extend(self._schema_dir.glob("**/*.json"))

        for file_path in schema_files:
            rel_path = str(file_path.relative_to(self._schema_dir))
            current_hash = self._compute_hash(file_path)
            current_hashes[rel_path] = current_hash

            if rel_path not in cached_hashes:
                # New schema
                changes.append(
                    SchemaChange(
                        file_path=rel_path,
                        change_type="added",
                        new_hash=current_hash,
                    )
                )
            elif cached_hashes[rel_path] != current_hash:
                # Modified schema - analyze column changes
                change = self._analyze_change(file_path, rel_path, cached_hashes[rel_path])
                changes.append(change)

        # Check for removed schemas
        for rel_path in cached_hashes:
            if rel_path not in current_hashes:
                changes.append(
                    SchemaChange(
                        file_path=rel_path,
                        change_type="removed",
                        old_hash=cached_hashes[rel_path],
                    )
                )

        # Update cache
        self._save_hashes(current_hashes)

        return changes

    def _analyze_change(
        self,
        file_path: Path,
        rel_path: str,
        old_hash: str,
    ) -> SchemaChange:
        """Analyze what changed in a schema file."""
        try:
            # Try to get old schema from git
            old_columns = self._get_old_columns_from_git(rel_path)
            new_schema = self._parse_schema(file_path)
            new_columns = set(c.get("name", c) for c in new_schema.get("columns", []))

            if old_columns:
                added = list(new_columns - old_columns)
                removed = list(old_columns - new_columns)
            else:
                added = []
                removed = []

            return SchemaChange(
                file_path=rel_path,
                change_type="modified",
                old_hash=old_hash,
                new_hash=self._compute_hash(file_path),
                columns_added=added,
                columns_removed=removed,
            )
        except Exception:
            return SchemaChange(
                file_path=rel_path,
                change_type="modified",
                old_hash=old_hash,
                new_hash=self._compute_hash(file_path),
            )

    def _get_old_columns_from_git(self, rel_path: str) -> Optional[set]:
        """Get columns from previous git commit."""
        try:
            result = subprocess.run(
                ["git", "show", f"HEAD~1:{self._schema_dir / rel_path}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                old_schema = yaml.safe_load(result.stdout)
                return set(c.get("name", c) for c in old_schema.get("columns", []))
        except Exception:
            pass
        return None


class QualityGate:
    """Enforces quality and privacy thresholds.

    Validates generated synthetic data meets minimum quality
    standards before passing the CI pipeline.
    """

    def __init__(
        self,
        quality_threshold: float = 0.85,
        privacy_threshold: float = 0.95,
    ):
        """Initialize quality gate.

        Args:
            quality_threshold: Minimum quality score (0-1)
            privacy_threshold: Minimum privacy score (0-1)
        """
        self._quality_threshold = quality_threshold
        self._privacy_threshold = privacy_threshold

    def check(
        self,
        synthetic_data: pd.DataFrame,
        real_data: Optional[pd.DataFrame] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> QualityGateResult:
        """Check if synthetic data passes quality gates.

        Args:
            synthetic_data: Generated data
            real_data: Reference data (optional)
            schema: Schema definition (optional)

        Returns:
            QualityGateResult with pass/fail status
        """
        violations: List[str] = []
        details: Dict[str, Any] = {}

        # Basic validation
        if len(synthetic_data) == 0:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                privacy_score=0.0,
                threshold_quality=self._quality_threshold,
                threshold_privacy=self._privacy_threshold,
                violations=["Empty dataset generated"],
            )

        # Compute quality score
        quality_score = self._compute_quality_score(synthetic_data, real_data, schema)
        details["quality_breakdown"] = quality_score

        # Compute privacy score
        privacy_score = self._compute_privacy_score(synthetic_data)
        details["privacy_breakdown"] = privacy_score

        overall_quality = quality_score.get("overall", 0.0)
        overall_privacy = privacy_score.get("overall", 0.0)

        # Check thresholds
        if overall_quality < self._quality_threshold:
            violations.append(
                f"Quality score {overall_quality:.2f} below threshold {self._quality_threshold}"
            )

        if overall_privacy < self._privacy_threshold:
            violations.append(
                f"Privacy score {overall_privacy:.2f} below threshold {self._privacy_threshold}"
            )

        return QualityGateResult(
            passed=len(violations) == 0,
            quality_score=overall_quality,
            privacy_score=overall_privacy,
            threshold_quality=self._quality_threshold,
            threshold_privacy=self._privacy_threshold,
            details=details,
            violations=violations,
        )

    def _compute_quality_score(
        self,
        synthetic_data: pd.DataFrame,
        real_data: Optional[pd.DataFrame],
        schema: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute quality score components."""
        scores: Dict[str, float] = {}

        # Schema compliance
        if schema:
            expected_cols = set(c.get("name", c) for c in schema.get("columns", []))
            actual_cols = set(synthetic_data.columns)
            scores["schema_compliance"] = len(expected_cols & actual_cols) / max(
                len(expected_cols), 1
            )

        # Data completeness (no nulls unless specified)
        null_ratio = synthetic_data.isnull().sum().sum() / synthetic_data.size
        scores["completeness"] = 1.0 - null_ratio

        # Distribution plausibility (basic checks)
        numeric_cols = synthetic_data.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            # Check for inf values
            inf_count = sum(synthetic_data[c].apply(lambda x: x == float("inf") or x == float("-inf")).sum() for c in numeric_cols)
            scores["validity"] = 1.0 - (inf_count / max(len(numeric_cols) * len(synthetic_data), 1))
        else:
            scores["validity"] = 1.0

        # Overall score
        scores["overall"] = sum(scores.values()) / len(scores) if scores else 0.0

        return scores

    def _compute_privacy_score(self, synthetic_data: pd.DataFrame) -> Dict[str, float]:
        """Compute privacy score components."""
        scores: Dict[str, float] = {}

        # Check for unique identifiers (high uniqueness = potential risk)
        for col in synthetic_data.columns:
            uniqueness = synthetic_data[col].nunique() / len(synthetic_data)
            if uniqueness > 0.99:
                scores[f"{col}_uniqueness_risk"] = 0.5  # Penalize high uniqueness
            else:
                scores[f"{col}_uniqueness_risk"] = 1.0

        # Overall privacy score
        scores["overall"] = sum(scores.values()) / len(scores) if scores else 1.0

        return scores


class ArtifactManager:
    """Manages synthetic data artifacts.

    Handles versioning, caching, and storage of generated data
    and trained models.
    """

    def __init__(self, output_dir: str, cache_dir: str):
        """Initialize artifact manager.

        Args:
            output_dir: Directory for output artifacts
            cache_dir: Directory for cached models
        """
        self._output_dir = Path(output_dir)
        self._cache_dir = Path(cache_dir)
        self._manifest_file = self._output_dir / "manifest.json"

    def save_artifact(
        self,
        data: pd.DataFrame,
        name: str,
        schema_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save synthetic data artifact.

        Args:
            data: Generated DataFrame
            name: Artifact name
            schema_hash: Hash of source schema
            metadata: Additional metadata

        Returns:
            Path to saved artifact
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Generate versioned filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.parquet"
        file_path = self._output_dir / filename

        # Save data
        data.to_parquet(file_path, index=False)

        # Update manifest
        self._update_manifest(
            name=name,
            file_path=str(file_path),
            schema_hash=schema_hash,
            n_rows=len(data),
            n_cols=len(data.columns),
            metadata=metadata or {},
        )

        logger.info(f"Saved artifact: {file_path}")
        return str(file_path)

    def _update_manifest(self, **entry: Any) -> None:
        """Update artifact manifest."""
        manifest = self._load_manifest()
        entry["created_at"] = datetime.utcnow().isoformat()
        manifest["artifacts"].append(entry)
        manifest["updated_at"] = datetime.utcnow().isoformat()

        with open(self._manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_manifest(self) -> Dict[str, Any]:
        """Load artifact manifest."""
        if self._manifest_file.exists():
            with open(self._manifest_file) as f:
                return json.load(f)
        return {"artifacts": [], "created_at": datetime.utcnow().isoformat()}

    def get_latest_artifact(self, name: str) -> Optional[str]:
        """Get path to latest artifact for a schema.

        Args:
            name: Schema/artifact name

        Returns:
            Path to latest artifact or None
        """
        manifest = self._load_manifest()
        matching = [a for a in manifest["artifacts"] if a["name"] == name]
        if matching:
            return matching[-1]["file_path"]
        return None

    def cache_model(self, model: Any, schema_hash: str) -> str:
        """Cache a fitted model.

        Args:
            model: Fitted generator model
            schema_hash: Hash of schema

        Returns:
            Path to cached model
        """
        import pickle

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_dir / f"model_{schema_hash}.pkl"

        with open(cache_path, "wb") as f:
            pickle.dump(model, f)

        return str(cache_path)

    def load_cached_model(self, schema_hash: str) -> Optional[Any]:
        """Load cached model if exists.

        Args:
            schema_hash: Hash of schema

        Returns:
            Cached model or None
        """
        import pickle

        cache_path = self._cache_dir / f"model_{schema_hash}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def cleanup_old_artifacts(self, retention_days: int) -> int:
        """Remove artifacts older than retention period.

        Args:
            retention_days: Days to retain artifacts

        Returns:
            Number of artifacts removed
        """
        removed = 0
        cutoff = datetime.utcnow().timestamp() - (retention_days * 86400)

        for file_path in self._output_dir.glob("*.parquet"):
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                removed += 1

        return removed


class SyntheticDataPipeline:
    """Main CI/CD pipeline for synthetic data generation.

    Orchestrates schema detection, generation, validation,
    and artifact management.
    """

    def __init__(self, config: Optional[CIConfig] = None):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or CIConfig()
        self._drift_detector = SchemaDriftDetector(
            self.config.schema_dir, self.config.cache_dir
        )
        self._quality_gate = QualityGate(
            self.config.quality_threshold,
            self.config.privacy_threshold,
        )
        self._artifact_manager = ArtifactManager(
            self.config.output_dir, self.config.cache_dir
        )

    def run(
        self,
        force_regenerate: bool = False,
        schemas: Optional[List[str]] = None,
    ) -> PipelineResult:
        """Run the synthetic data pipeline.

        Args:
            force_regenerate: Force regeneration even without drift
            schemas: Specific schemas to process (None = all)

        Returns:
            PipelineResult with status and artifacts
        """
        start_time = time.time()
        errors: List[str] = []
        warnings: List[str] = []
        files_generated: List[str] = []
        quality_results: Dict[str, QualityGateResult] = {}

        try:
            # Detect schema changes
            changes = self._drift_detector.detect_changes()

            if not force_regenerate and not changes:
                logger.info("No schema changes detected, skipping generation")
                return PipelineResult(
                    status=PipelineStatus.SKIPPED,
                    passed=True,
                    duration_seconds=time.time() - start_time,
                    schemas_processed=0,
                    files_generated=[],
                    quality_results={},
                    schema_changes=[],
                    artifacts=[],
                    errors=[],
                    warnings=["No schema changes detected"],
                )

            # Get schemas to process
            schema_dir = Path(self.config.schema_dir)
            if schemas:
                schema_files = [schema_dir / s for s in schemas]
            else:
                schema_files = list(schema_dir.glob("**/*.yaml"))
                schema_files.extend(schema_dir.glob("**/*.yml"))

            # Process each schema
            for schema_file in schema_files:
                if not schema_file.exists():
                    warnings.append(f"Schema file not found: {schema_file}")
                    continue

                try:
                    result = self._process_schema(schema_file)
                    if result["success"]:
                        files_generated.append(result["output_path"])
                        quality_results[str(schema_file)] = result["quality_result"]
                    else:
                        errors.append(f"Failed to process {schema_file}: {result.get('error')}")
                except Exception as e:
                    errors.append(f"Error processing {schema_file}: {str(e)}")

            # Check if all quality gates passed
            all_passed = all(qr.passed for qr in quality_results.values())
            if errors:
                all_passed = False

            # Cleanup old artifacts
            removed = self._artifact_manager.cleanup_old_artifacts(
                self.config.artifact_retention_days
            )
            if removed > 0:
                logger.info(f"Cleaned up {removed} old artifacts")

            # Notify on failure
            if not all_passed and self.config.notify_on_failure:
                self._send_notification(errors, warnings)

            return PipelineResult(
                status=PipelineStatus.PASSED if all_passed else PipelineStatus.FAILED,
                passed=all_passed,
                duration_seconds=time.time() - start_time,
                schemas_processed=len(schema_files),
                files_generated=files_generated,
                quality_results=quality_results,
                schema_changes=changes,
                artifacts=files_generated,
                errors=errors,
                warnings=warnings,
                metadata={
                    "config": {
                        "quality_threshold": self.config.quality_threshold,
                        "privacy_threshold": self.config.privacy_threshold,
                    }
                },
            )

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return PipelineResult(
                status=PipelineStatus.FAILED,
                passed=False,
                duration_seconds=time.time() - start_time,
                schemas_processed=0,
                files_generated=[],
                quality_results={},
                schema_changes=[],
                artifacts=[],
                errors=[str(e)],
                warnings=[],
            )

    def _process_schema(self, schema_file: Path) -> Dict[str, Any]:
        """Process a single schema file."""
        with open(schema_file) as f:
            schema = yaml.safe_load(f)

        schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]
        name = schema.get("name", schema_file.stem)
        n_samples = schema.get("n_samples", self.config.n_samples_default)

        # Check for cached model
        cached_model = None
        if self.config.enable_caching:
            cached_model = self._artifact_manager.load_cached_model(schema_hash)

        # Generate synthetic data
        synthetic_data = self._generate_from_schema(schema, n_samples, cached_model)

        # Run quality gate
        quality_result = self._quality_gate.check(synthetic_data, schema=schema)

        if not quality_result.passed:
            return {
                "success": False,
                "error": f"Quality gate failed: {quality_result.violations}",
                "quality_result": quality_result,
            }

        # Save artifact
        output_path = self._artifact_manager.save_artifact(
            data=synthetic_data,
            name=name,
            schema_hash=schema_hash,
            metadata={"schema_file": str(schema_file)},
        )

        return {
            "success": True,
            "output_path": output_path,
            "quality_result": quality_result,
        }

    def _generate_from_schema(
        self,
        schema: Dict[str, Any],
        n_samples: int,
        cached_model: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data from schema definition."""
        import numpy as np

        columns = schema.get("columns", [])
        data: Dict[str, Any] = {}

        for col_def in columns:
            name = col_def.get("name", col_def) if isinstance(col_def, dict) else col_def
            col_type = col_def.get("type", "string") if isinstance(col_def, dict) else "string"
            constraints = col_def.get("constraints", {}) if isinstance(col_def, dict) else {}

            if col_type == "integer":
                min_val = constraints.get("min", 0)
                max_val = constraints.get("max", 100)
                data[name] = np.random.randint(min_val, max_val + 1, n_samples)

            elif col_type == "float":
                min_val = constraints.get("min", 0.0)
                max_val = constraints.get("max", 1000.0)
                data[name] = np.random.uniform(min_val, max_val, n_samples)

            elif col_type == "categorical":
                values = constraints.get("values", ["A", "B", "C"])
                data[name] = np.random.choice(values, n_samples)

            elif col_type == "boolean":
                data[name] = np.random.choice([True, False], n_samples)

            elif col_type == "datetime":
                start = pd.Timestamp(constraints.get("min", "2020-01-01"))
                end = pd.Timestamp(constraints.get("max", "2024-12-31"))
                data[name] = pd.to_datetime(
                    np.random.randint(start.value, end.value, n_samples)
                )

            else:  # string
                prefix = constraints.get("prefix", name)
                if constraints.get("unique", False):
                    data[name] = [f"{prefix}_{i:06d}" for i in range(n_samples)]
                else:
                    data[name] = [f"{prefix}_{np.random.randint(10000)}" for _ in range(n_samples)]

        return pd.DataFrame(data)

    def _send_notification(self, errors: List[str], warnings: List[str]) -> None:
        """Send failure notification."""
        if not self.config.notification_webhook:
            return

        try:
            import urllib.request

            payload = {
                "text": f"Synthetic Data Pipeline Failed\nErrors: {len(errors)}\nWarnings: {len(warnings)}",
                "errors": errors[:5],  # Limit errors
            }

            req = urllib.request.Request(
                self.config.notification_webhook,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")


def generate_github_workflow(
    schema_path: str = "schemas/",
    output_path: str = "data/synthetic/",
    quality_threshold: float = 0.85,
    python_version: str = "3.11",
) -> str:
    """Generate GitHub Actions workflow YAML.

    Args:
        schema_path: Path to schema directory
        output_path: Path for output data
        quality_threshold: Quality threshold
        python_version: Python version to use

    Returns:
        YAML workflow content
    """
    workflow = {
        "name": "Synthetic Data Generation",
        "on": {
            "push": {"paths": [f"{schema_path}**", "config/synthetic*.yaml"]},
            "pull_request": {"paths": [f"{schema_path}**"]},
            "workflow_dispatch": {
                "inputs": {
                    "force_regenerate": {
                        "description": "Force regeneration",
                        "type": "boolean",
                        "default": False,
                    }
                }
            },
        },
        "jobs": {
            "generate": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {
                        "uses": "actions/setup-python@v5",
                        "with": {"python-version": python_version},
                    },
                    {
                        "name": "Install Genesis",
                        "run": "pip install genesis-synth",
                    },
                    {
                        "name": "Generate Synthetic Data",
                        "run": f"genesis cicd run --schema-dir {schema_path} --output-dir {output_path} --quality-threshold {quality_threshold}",
                        "env": {
                            "GENESIS_LICENSE_KEY": "${{ secrets.GENESIS_LICENSE_KEY }}"
                        },
                    },
                    {
                        "name": "Upload Artifacts",
                        "uses": "actions/upload-artifact@v4",
                        "with": {
                            "name": "synthetic-data",
                            "path": output_path,
                            "retention-days": 30,
                        },
                    },
                ],
            }
        },
    }

    return yaml.dump(workflow, default_flow_style=False, sort_keys=False)


def generate_gitlab_ci(
    schema_path: str = "schemas/",
    output_path: str = "data/synthetic/",
    quality_threshold: float = 0.85,
) -> str:
    """Generate GitLab CI configuration YAML.

    Args:
        schema_path: Path to schema directory
        output_path: Path for output data
        quality_threshold: Quality threshold

    Returns:
        YAML CI configuration
    """
    config = {
        "stages": ["generate", "validate"],
        "variables": {
            "SCHEMA_PATH": schema_path,
            "OUTPUT_PATH": output_path,
            "QUALITY_THRESHOLD": str(quality_threshold),
        },
        "generate_synthetic_data": {
            "stage": "generate",
            "image": "python:3.11",
            "script": [
                "pip install genesis-synth",
                "genesis cicd run --schema-dir $SCHEMA_PATH --output-dir $OUTPUT_PATH --quality-threshold $QUALITY_THRESHOLD",
            ],
            "artifacts": {
                "paths": [output_path],
                "expire_in": "30 days",
            },
            "rules": [
                {"changes": [f"{schema_path}**/*"], "when": "always"},
                {"when": "manual"},
            ],
        },
    }

    return yaml.dump(config, default_flow_style=False, sort_keys=False)


# CLI integration
def run_cicd_cli(args: List[str]) -> int:
    """Run CI/CD pipeline from CLI.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 = success)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Genesis CI/CD Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline")
    run_parser.add_argument("--schema-dir", default="schemas/")
    run_parser.add_argument("--output-dir", default="data/synthetic/")
    run_parser.add_argument("--quality-threshold", type=float, default=0.85)
    run_parser.add_argument("--force", action="store_true")

    # Generate workflow command
    workflow_parser = subparsers.add_parser("generate-workflow", help="Generate CI workflow")
    workflow_parser.add_argument("--provider", choices=["github", "gitlab"], default="github")
    workflow_parser.add_argument("--output", default="-")

    parsed = parser.parse_args(args)

    if parsed.command == "run":
        config = CIConfig(
            schema_dir=parsed.schema_dir,
            output_dir=parsed.output_dir,
            quality_threshold=parsed.quality_threshold,
        )
        pipeline = SyntheticDataPipeline(config)
        result = pipeline.run(force_regenerate=parsed.force)

        print(f"Status: {result.status.value}")
        print(f"Schemas processed: {result.schemas_processed}")
        print(f"Files generated: {len(result.files_generated)}")
        print(f"Duration: {result.duration_seconds:.2f}s")

        if result.errors:
            print("\nErrors:")
            for err in result.errors:
                print(f"  - {err}")

        return 0 if result.passed else 1

    elif parsed.command == "generate-workflow":
        if parsed.provider == "github":
            content = generate_github_workflow()
        else:
            content = generate_gitlab_ci()

        if parsed.output == "-":
            print(content)
        else:
            with open(parsed.output, "w") as f:
                f.write(content)

        return 0

    parser.print_help()
    return 1
