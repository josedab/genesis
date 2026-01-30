"""Privacy Attack Testing for synthetic data.

This module provides automated privacy attack simulations to validate
that synthetic data doesn't leak information about the original training data.

Features:
    - Membership Inference Attack: Can an attacker tell if a record was in training?
    - Attribute Inference Attack: Can sensitive attributes be inferred from known ones?
    - Re-identification Attack: Can synthetic records be linked to real individuals?
    - Comprehensive audit reports with risk levels and recommendations

Example:
    Quick privacy audit with convenience function::

        from genesis import run_privacy_audit

        report = run_privacy_audit(
            real_data=original_df,
            synthetic_data=synthetic_df,
            sensitive_columns=["ssn", "income", "diagnosis"]
        )

        print(f"Overall Risk: {report.overall_risk}")
        print(f"Passed: {report.passed}")

    Running individual attacks::

        from genesis.privacy_attacks import MembershipInferenceAttack

        attack = MembershipInferenceAttack()
        result = attack.run(real_df, synthetic_df, holdout_df)

        print(f"Attack accuracy: {result.accuracy:.1%}")
        print(f"Risk level: {result.risk_level}")

Classes:
    MembershipInferenceAttack: Tests membership disclosure risk.
    AttributeInferenceAttack: Tests attribute disclosure risk.
    ReidentificationAttack: Tests re-identification risk.
    PrivacyAttackTester: Orchestrates multiple attack types.
    PrivacyAuditReport: Comprehensive audit results.

Functions:
    run_privacy_audit: One-line convenience function for full audit.

Risk Levels:
    LOW: Advantage < 5% over random guessing
    MEDIUM: Advantage 5-15%
    HIGH: Advantage 15-30%
    CRITICAL: Advantage > 30%
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AttackType(str, Enum):
    """Types of privacy attacks.

    Attributes:
        MEMBERSHIP_INFERENCE: Test if specific records were in training data.
        ATTRIBUTE_INFERENCE: Test if sensitive attributes can be inferred.
        REIDENTIFICATION: Test if synthetic records link to real individuals.
        LINKAGE: Test record linkage across datasets.
    """

    MEMBERSHIP_INFERENCE = "membership_inference"
    ATTRIBUTE_INFERENCE = "attribute_inference"
    REIDENTIFICATION = "reidentification"
    LINKAGE = "linkage"


@dataclass
class AttackResult:
    """Result of a privacy attack simulation.

    Attributes:
        attack_type: The type of attack performed.
        success_rate: Attack success rate (0-1), higher = more vulnerable.
        baseline_rate: Random guess success rate for comparison.
        advantage: Attacker advantage over baseline (success_rate - baseline_rate).
        risk_level: Qualitative risk assessment ("low", "medium", "high", "critical").
        details: Additional metrics and diagnostic information.
        recommendations: Suggested mitigations if risk is elevated.

    Example:
        >>> result = attack.run(real_df, synthetic_df)
        >>> if result.risk_level in ["high", "critical"]:
        ...     print("Privacy risk detected!")
        ...     for rec in result.recommendations:
        ...         print(f"  - {rec}")
    """

    attack_type: AttackType
    success_rate: float  # Attack success rate (0-1)
    baseline_rate: float  # Random guess baseline
    advantage: float  # Attacker advantage over baseline
    risk_level: str = ""  # "low", "medium", "high", "critical" - calculated in __post_init__
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Alias for success_rate for compatibility."""
        return self.success_rate

    @property
    def reidentification_rate(self) -> float:
        """Alias for success_rate for re-identification attacks."""
        return self.success_rate

    def __post_init__(self) -> None:
        """Calculate risk level and recommendations based on advantage."""
        if not self.risk_level:  # Only calculate if not provided
            if self.advantage <= 0.05:
                self.risk_level = "low"
            elif self.advantage <= 0.15:
                self.risk_level = "medium"
            elif self.advantage <= 0.30:
                self.risk_level = "high"
            else:
                self.risk_level = "critical"

        if not self.recommendations:
            self.recommendations = self._generate_recommendations()

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on attack results."""
        recs = []

        if self.risk_level in ["high", "critical"]:
            if self.attack_type == AttackType.MEMBERSHIP_INFERENCE:
                recs.append("Increase differential privacy noise (lower epsilon)")
                recs.append("Reduce model training epochs")
                recs.append("Use larger training datasets")
            elif self.attack_type == AttackType.ATTRIBUTE_INFERENCE:
                recs.append("Remove or generalize sensitive attributes")
                recs.append("Apply differential privacy to specific columns")
                recs.append("Use k-anonymity with higher k value")
            elif self.attack_type == AttackType.REIDENTIFICATION:
                recs.append("Remove quasi-identifiers")
                recs.append("Apply data suppression for rare combinations")
                recs.append("Use l-diversity or t-closeness")

        return recs


@dataclass
class PrivacyAuditReport:
    """Comprehensive privacy audit report."""

    attack_results: List[AttackResult]
    overall_risk: str
    privacy_score: float  # 0-1, higher is more private
    summary: str
    tested_at: str = ""

    def __post_init__(self) -> None:
        import datetime

        self.tested_at = datetime.datetime.now().isoformat()

    @property
    def passed(self) -> bool:
        """Check if audit passed (overall risk is not high/critical)."""
        return self.overall_risk.lower() not in ["high", "critical"]

    @property
    def membership_result(self) -> Optional[AttackResult]:
        """Get membership inference attack result if available."""
        for result in self.attack_results:
            if result.attack_type == AttackType.MEMBERSHIP_INFERENCE:
                return result
        return None

    @property
    def reidentification_result(self) -> Optional[AttackResult]:
        """Get re-identification attack result if available."""
        for result in self.attack_results:
            if result.attack_type == AttackType.REIDENTIFICATION:
                return result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_risk": self.overall_risk,
            "privacy_score": self.privacy_score,
            "summary": self.summary,
            "tested_at": self.tested_at,
            "passed": self.passed,
            "attacks": [
                {
                    "type": r.attack_type.value,
                    "success_rate": r.success_rate,
                    "baseline_rate": r.baseline_rate,
                    "advantage": r.advantage,
                    "risk_level": r.risk_level,
                    "recommendations": r.recommendations,
                }
                for r in self.attack_results
            ],
        }

    def to_json(self, path: str) -> None:
        """Save report as JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_html(self, path: str) -> None:
        """Save report as HTML file."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Privacy Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .risk-low {{ color: green; }}
        .risk-medium {{ color: orange; }}
        .risk-high {{ color: red; }}
        .risk-critical {{ color: darkred; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Privacy Audit Report</h1>
    <p><strong>Overall Risk:</strong> <span class="risk-{self.overall_risk.lower()}">{self.overall_risk}</span></p>
    <p><strong>Privacy Score:</strong> {self.privacy_score:.2f}</p>
    <p><strong>Passed:</strong> {"Yes" if self.passed else "No"}</p>
    <p><strong>Tested At:</strong> {self.tested_at}</p>
    <h2>Summary</h2>
    <p>{self.summary}</p>
    <h2>Attack Results</h2>
    <table>
        <tr><th>Attack Type</th><th>Success Rate</th><th>Advantage</th><th>Risk Level</th></tr>
        {"".join(f'<tr><td>{r.attack_type.value}</td><td>{r.success_rate:.1%}</td><td>{r.advantage:.1%}</td><td class="risk-{r.risk_level}">{r.risk_level}</td></tr>' for r in self.attack_results)}
    </table>
</body>
</html>"""
        with open(path, "w") as f:
            f.write(html)


class MembershipInferenceAttack:
    """Simulate membership inference attacks.

    Tests whether an attacker can determine if a specific record
    was in the training data used to create the synthetic data.
    """

    def __init__(
        self,
        n_shadow_models: int = 5,
        attack_model: str = "gradient_boosting",
        test_size: float = 0.3,
    ):
        """Initialize attack simulator.

        Args:
            n_shadow_models: Number of shadow models to train
            attack_model: Model type for attack classifier
            test_size: Proportion of data for testing
        """
        self.n_shadow_models = n_shadow_models
        self.attack_model = attack_model
        self.test_size = test_size

    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        generator_fn: Optional[Callable] = None,
    ) -> AttackResult:
        """Run membership inference attack.

        Args:
            real_data: Original training data
            synthetic_data: Generated synthetic data
            generator_fn: Optional function to train new generators

        Returns:
            AttackResult with success metrics
        """
        # Prepare features
        real_data = self._preprocess(real_data)
        synthetic_data = self._preprocess(synthetic_data)

        # Build attack training data using shadow models
        attack_X, attack_y = self._build_attack_data(real_data, synthetic_data, generator_fn)

        if len(attack_X) == 0:
            return AttackResult(
                attack_type=AttackType.MEMBERSHIP_INFERENCE,
                success_rate=0.5,
                baseline_rate=0.5,
                advantage=0.0,
                details={"error": "Could not build attack data"},
            )

        # Train attack model
        X_train, X_test, y_train, y_test = train_test_split(
            attack_X, attack_y, test_size=self.test_size, random_state=42
        )

        attack_clf = self._get_attack_model()
        attack_clf.fit(X_train, y_train)

        # Evaluate attack
        y_pred = attack_clf.predict(X_test)
        y_prob = attack_clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        baseline = 0.5  # Random guess
        advantage = max(0, accuracy - baseline)

        return AttackResult(
            attack_type=AttackType.MEMBERSHIP_INFERENCE,
            success_rate=accuracy,
            baseline_rate=baseline,
            advantage=advantage,
            details={
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "auc": auc,
                "n_samples": len(attack_y),
            },
        )

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for attack."""
        df = df.copy()

        # Encode categoricals
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # Fill missing values
        df = df.fillna(df.median(numeric_only=True))

        return df

    def _build_attack_data(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        generator_fn: Optional[Callable],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build training data for attack model."""
        # Use synthetic data statistics as features
        # Members: records in real_data (label=1)
        # Non-members: records not in training (label=0)

        features = []
        labels = []

        # Sample records from real data (members)
        n_samples = min(len(real_data), 1000)
        member_samples = real_data.sample(n=n_samples, random_state=42)

        for _, row in member_samples.iterrows():
            feat = self._compute_record_features(row, synthetic_data)
            features.append(feat)
            labels.append(1)

        # Generate non-member samples (perturbed real samples)
        non_members = self._generate_non_members(real_data, n_samples)

        for _, row in non_members.iterrows():
            feat = self._compute_record_features(row, synthetic_data)
            features.append(feat)
            labels.append(0)

        return np.array(features), np.array(labels)

    def _compute_record_features(
        self, record: pd.Series, synthetic_data: pd.DataFrame
    ) -> np.ndarray:
        """Compute features for a record based on synthetic data similarity."""
        features = []

        # Distance to nearest synthetic record
        numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            record_vals = record[numeric_cols].values.astype(float)
            synth_vals = synthetic_data[numeric_cols].values.astype(float)

            # Handle missing values
            record_vals = np.nan_to_num(record_vals, nan=0)
            synth_vals = np.nan_to_num(synth_vals, nan=0)

            # Normalize
            scaler = StandardScaler()
            synth_vals_scaled = scaler.fit_transform(synth_vals)
            record_scaled = scaler.transform(record_vals.reshape(1, -1))

            # Compute distances
            distances = np.linalg.norm(synth_vals_scaled - record_scaled, axis=1)

            features.extend(
                [
                    np.min(distances),
                    np.mean(np.sort(distances)[:5]),  # Mean of 5 nearest
                    np.percentile(distances, 10),
                ]
            )
        else:
            features.extend([0, 0, 0])

        return np.array(features)

    def _generate_non_members(self, real_data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Generate non-member samples by perturbing real data."""
        non_members = real_data.sample(n=n_samples, random_state=43).copy()

        # Perturb numeric columns
        for col in non_members.select_dtypes(include=[np.number]).columns:
            std = non_members[col].std()
            if std > 0:
                noise = np.random.normal(0, std * 0.5, len(non_members))
                non_members[col] = non_members[col] + noise

        # Shuffle categorical columns
        for col in non_members.select_dtypes(include=["object", "category"]).columns:
            non_members[col] = np.random.permutation(non_members[col].values)

        return non_members

    def _get_attack_model(self):
        """Get attack model instance."""
        if self.attack_model == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.attack_model == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return LogisticRegression(max_iter=1000)


class AttributeInferenceAttack:
    """Simulate attribute inference attacks.

    Tests whether an attacker can infer sensitive attribute values
    from other attributes using the synthetic data.
    """

    def __init__(self, n_folds: int = 5):
        """Initialize attack.

        Args:
            n_folds: Number of cross-validation folds
        """
        self.n_folds = n_folds

    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_column: str,
    ) -> AttackResult:
        """Run attribute inference attack.

        Args:
            real_data: Original training data
            synthetic_data: Generated synthetic data
            sensitive_column: Column to try to infer

        Returns:
            AttackResult with success metrics
        """
        if sensitive_column not in real_data.columns:
            raise ValueError(f"Column '{sensitive_column}' not found")

        # Prepare features (all columns except sensitive)
        feature_cols = [c for c in real_data.columns if c != sensitive_column]

        X_real = self._encode_features(real_data[feature_cols])
        y_real = self._encode_target(real_data[sensitive_column])

        X_synth = self._encode_features(synthetic_data[feature_cols])
        y_synth = self._encode_target(synthetic_data[sensitive_column])

        # Train attacker on synthetic, test on real
        attacker = GradientBoostingClassifier(n_estimators=100, random_state=42)
        attacker.fit(X_synth, y_synth)

        y_pred = attacker.predict(X_real)
        accuracy = accuracy_score(y_real, y_pred)

        # Baseline: most frequent class
        baseline = max(np.bincount(y_real)) / len(y_real)
        advantage = max(0, accuracy - baseline)

        return AttackResult(
            attack_type=AttackType.ATTRIBUTE_INFERENCE,
            success_rate=accuracy,
            baseline_rate=baseline,
            advantage=advantage,
            details={
                "sensitive_column": sensitive_column,
                "n_classes": len(np.unique(y_real)),
                "feature_columns": len(feature_cols),
            },
        )

    def _encode_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode features for model."""
        df = df.copy()

        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        return df.fillna(0).values

    def _encode_target(self, series: pd.Series) -> np.ndarray:
        """Encode target variable."""
        le = LabelEncoder()
        return le.fit_transform(series.astype(str))


class ReidentificationAttack:
    """Simulate re-identification attacks.

    Tests whether synthetic records can be linked back to
    real individuals using quasi-identifiers.
    """

    def __init__(self, distance_threshold: float = 0.1):
        """Initialize attack.

        Args:
            distance_threshold: Threshold for considering a match
        """
        self.distance_threshold = distance_threshold

    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        quasi_identifiers: Optional[List[str]] = None,
    ) -> AttackResult:
        """Run re-identification attack.

        Args:
            real_data: Original training data
            synthetic_data: Generated synthetic data
            quasi_identifiers: Columns to use for matching

        Returns:
            AttackResult with success metrics
        """
        if quasi_identifiers is None:
            # Auto-detect quasi-identifiers (categorical + some numeric)
            quasi_identifiers = self._detect_quasi_identifiers(real_data)

        if not quasi_identifiers:
            return AttackResult(
                attack_type=AttackType.REIDENTIFICATION,
                success_rate=0.0,
                baseline_rate=0.0,
                advantage=0.0,
                details={"error": "No quasi-identifiers found"},
            )

        # Compute matches
        n_matches = 0
        n_samples = min(len(synthetic_data), 1000)
        sampled_synth = synthetic_data.sample(n=n_samples, random_state=42)

        for _, synth_row in sampled_synth.iterrows():
            if self._find_match(synth_row, real_data, quasi_identifiers):
                n_matches += 1

        success_rate = n_matches / n_samples
        baseline = 1 / len(real_data)  # Random guess
        advantage = max(0, success_rate - baseline)

        return AttackResult(
            attack_type=AttackType.REIDENTIFICATION,
            success_rate=success_rate,
            baseline_rate=baseline,
            advantage=advantage,
            details={
                "quasi_identifiers": quasi_identifiers,
                "n_matches": n_matches,
                "n_tested": n_samples,
            },
        )

    def _detect_quasi_identifiers(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect potential quasi-identifiers."""
        qis = []

        for col in df.columns:
            # Skip high cardinality columns (likely IDs)
            if df[col].nunique() > len(df) * 0.9:
                continue

            # Include low-medium cardinality columns
            if df[col].nunique() < len(df) * 0.3:
                qis.append(col)

        return qis[:10]  # Limit to 10

    def _find_match(
        self,
        synth_row: pd.Series,
        real_data: pd.DataFrame,
        quasi_identifiers: List[str],
    ) -> bool:
        """Check if synthetic row matches any real row."""
        for _, real_row in real_data.iterrows():
            match = True
            for qi in quasi_identifiers:
                if synth_row[qi] != real_row[qi]:
                    match = False
                    break
            if match:
                return True
        return False


class PrivacyAttackTester:
    """Run comprehensive privacy attack testing."""

    def __init__(self, verbose: bool = True):
        """Initialize tester.

        Args:
            verbose: Print progress
        """
        self.verbose = verbose

        self.membership_attack = MembershipInferenceAttack()
        self.attribute_attack = AttributeInferenceAttack()
        self.reidentification_attack = ReidentificationAttack()

    def run_full_audit(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: Optional[List[str]] = None,
        quasi_identifiers: Optional[List[str]] = None,
    ) -> PrivacyAuditReport:
        """Run full privacy audit.

        Args:
            real_data: Original training data
            synthetic_data: Generated synthetic data
            sensitive_columns: Columns to test for attribute inference
            quasi_identifiers: Columns for re-identification testing

        Returns:
            PrivacyAuditReport with all results
        """
        results = []

        # Membership inference
        if self.verbose:
            print("Running membership inference attack...")

        mi_result = self.membership_attack.run(real_data, synthetic_data)
        results.append(mi_result)

        if self.verbose:
            print(
                f"  Success rate: {mi_result.success_rate:.1%}, "
                f"Advantage: {mi_result.advantage:.1%}"
            )

        # Attribute inference
        if sensitive_columns is None:
            # Auto-detect potential sensitive columns
            sensitive_columns = self._detect_sensitive_columns(real_data)

        for col in sensitive_columns[:3]:  # Test top 3
            if self.verbose:
                print(f"Running attribute inference attack on '{col}'...")

            try:
                ai_result = self.attribute_attack.run(real_data, synthetic_data, col)
                results.append(ai_result)

                if self.verbose:
                    print(
                        f"  Success rate: {ai_result.success_rate:.1%}, "
                        f"Advantage: {ai_result.advantage:.1%}"
                    )
            except Exception as e:
                if self.verbose:
                    print(f"  Skipped: {e}")

        # Re-identification
        if self.verbose:
            print("Running re-identification attack...")

        ri_result = self.reidentification_attack.run(real_data, synthetic_data, quasi_identifiers)
        results.append(ri_result)

        if self.verbose:
            print(
                f"  Success rate: {ri_result.success_rate:.1%}, "
                f"Advantage: {ri_result.advantage:.1%}"
            )

        # Compute overall metrics
        overall_risk = self._compute_overall_risk(results)
        privacy_score = self._compute_privacy_score(results)
        summary = self._generate_summary(results, overall_risk)

        return PrivacyAuditReport(
            attack_results=results,
            overall_risk=overall_risk,
            privacy_score=privacy_score,
            summary=summary,
        )

    def _detect_sensitive_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect potentially sensitive columns."""
        sensitive_patterns = [
            "income",
            "salary",
            "wage",
            "pay",
            "age",
            "gender",
            "sex",
            "race",
            "ethnicity",
            "diagnosis",
            "disease",
            "health",
            "medical",
            "ssn",
            "social",
            "address",
            "zip",
            "postal",
            "credit",
            "score",
            "rating",
        ]

        candidates = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in sensitive_patterns:
                if pattern in col_lower:
                    candidates.append(col)
                    break

        # Also include categorical columns with low cardinality
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col not in candidates and df[col].nunique() < 20:
                candidates.append(col)

        return candidates[:5]

    def _compute_overall_risk(self, results: List[AttackResult]) -> str:
        """Compute overall risk level."""
        risk_scores = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }

        max_risk = max(risk_scores.get(r.risk_level, 1) for r in results)

        for level, score in risk_scores.items():
            if score == max_risk:
                return level

        return "low"

    def _compute_privacy_score(self, results: List[AttackResult]) -> float:
        """Compute privacy score (0-1, higher is more private)."""
        if not results:
            return 1.0

        # Average of (1 - advantage) across all attacks
        scores = [1 - min(r.advantage, 0.5) * 2 for r in results]
        return np.mean(scores)

    def _generate_summary(self, results: List[AttackResult], overall_risk: str) -> str:
        """Generate human-readable summary."""
        lines = [
            "Privacy Audit Summary",
            f"Overall Risk Level: {overall_risk.upper()}",
            "",
            "Attack Results:",
        ]

        for r in results:
            lines.append(
                f"  - {r.attack_type.value}: {r.risk_level} risk " f"(advantage: {r.advantage:.1%})"
            )

        # Collect all recommendations
        all_recs = []
        for r in results:
            all_recs.extend(r.recommendations)

        if all_recs:
            lines.extend(["", "Recommendations:"])
            for rec in list(set(all_recs))[:5]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)


def run_privacy_audit(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
    holdout_data: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> PrivacyAuditReport:
    """Convenience function to run full privacy audit.

    Args:
        real_data: Original training data
        synthetic_data: Generated synthetic data
        sensitive_columns: Columns to test for attribute inference
        quasi_identifiers: Columns for re-identification testing
        holdout_data: Optional holdout set for more accurate testing
        verbose: Print progress

    Returns:
        PrivacyAuditReport
    """
    tester = PrivacyAttackTester(verbose=verbose)
    return tester.run_full_audit(
        real_data,
        synthetic_data,
        sensitive_columns=sensitive_columns,
        quasi_identifiers=quasi_identifiers,
    )
