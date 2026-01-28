"""Privacy analysis for data."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PrivacyRiskAssessment:
    """Assessment of privacy risks in data."""

    overall_risk_score: float  # 0 (low risk) to 1 (high risk)
    quasi_identifiers: List[str]
    sensitive_columns: List[str]
    k_anonymity_estimate: int
    reidentification_risk: float
    uniqueness_risk: float
    rare_category_risk: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_risk_score": self.overall_risk_score,
            "quasi_identifiers": self.quasi_identifiers,
            "sensitive_columns": self.sensitive_columns,
            "k_anonymity_estimate": self.k_anonymity_estimate,
            "reidentification_risk": self.reidentification_risk,
            "uniqueness_risk": self.uniqueness_risk,
            "rare_category_risk": self.rare_category_risk,
            "details": self.details,
        }


class PrivacyAnalyzer:
    """Analyzer for assessing privacy risks in data."""

    # Common quasi-identifier patterns
    QUASI_IDENTIFIER_PATTERNS = [
        "age",
        "birth",
        "gender",
        "sex",
        "zip",
        "postal",
        "location",
        "city",
        "state",
        "country",
        "region",
        "address",
        "ethnicity",
        "race",
        "religion",
        "education",
        "occupation",
        "income",
        "marital",
        "nationality",
    ]

    # Common sensitive attribute patterns
    SENSITIVE_PATTERNS = [
        "disease",
        "diagnosis",
        "health",
        "medical",
        "condition",
        "salary",
        "income",
        "wage",
        "credit",
        "debt",
        "balance",
        "ssn",
        "social_security",
        "tax_id",
        "passport",
        "password",
        "secret",
        "token",
        "key",
        "political",
        "religion",
        "sexual",
        "orientation",
    ]

    def __init__(
        self,
        rare_threshold: float = 0.01,
        uniqueness_threshold: float = 0.8,
    ) -> None:
        """Initialize the privacy analyzer.

        Args:
            rare_threshold: Threshold for considering a category rare
            uniqueness_threshold: Threshold for high uniqueness risk
        """
        self.rare_threshold = rare_threshold
        self.uniqueness_threshold = uniqueness_threshold

    def analyze(
        self,
        data: pd.DataFrame,
        quasi_identifiers: Optional[List[str]] = None,
        sensitive_columns: Optional[List[str]] = None,
    ) -> PrivacyRiskAssessment:
        """Analyze privacy risks in a DataFrame.

        Args:
            data: DataFrame to analyze
            quasi_identifiers: Optional explicit list of quasi-identifiers
            sensitive_columns: Optional explicit list of sensitive columns

        Returns:
            PrivacyRiskAssessment with detailed risk analysis
        """
        # Detect or use provided quasi-identifiers
        if quasi_identifiers is None:
            quasi_identifiers = self._detect_quasi_identifiers(data)
        else:
            # Validate provided columns exist
            quasi_identifiers = [c for c in quasi_identifiers if c in data.columns]

        # Detect or use provided sensitive columns
        if sensitive_columns is None:
            sensitive_columns = self._detect_sensitive_columns(data)
        else:
            sensitive_columns = [c for c in sensitive_columns if c in data.columns]

        # Compute k-anonymity estimate
        k_anonymity = self._estimate_k_anonymity(data, quasi_identifiers)

        # Compute uniqueness risk
        uniqueness_risk = self._compute_uniqueness_risk(data, quasi_identifiers)

        # Compute rare category risk
        rare_risk, rare_details = self._compute_rare_category_risk(data)

        # Compute re-identification risk
        reidentification_risk = self._compute_reidentification_risk(
            data, quasi_identifiers, k_anonymity
        )

        # Compute overall risk score
        overall_risk = self._compute_overall_risk(
            uniqueness_risk, rare_risk, reidentification_risk, k_anonymity
        )

        return PrivacyRiskAssessment(
            overall_risk_score=overall_risk,
            quasi_identifiers=quasi_identifiers,
            sensitive_columns=sensitive_columns,
            k_anonymity_estimate=k_anonymity,
            reidentification_risk=reidentification_risk,
            uniqueness_risk=uniqueness_risk,
            rare_category_risk=rare_risk,
            details={
                "rare_categories": rare_details,
                "column_uniqueness": self._compute_column_uniqueness(data),
            },
        )

    def _detect_quasi_identifiers(self, data: pd.DataFrame) -> List[str]:
        """Detect potential quasi-identifier columns."""
        quasi_identifiers = []

        for col in data.columns:
            col_lower = col.lower()

            # Check against known patterns
            if any(pattern in col_lower for pattern in self.QUASI_IDENTIFIER_PATTERNS):
                quasi_identifiers.append(col)
                continue

            # Heuristic: columns with moderate cardinality (not too unique, not constant)
            cardinality_ratio = data[col].nunique() / len(data)
            if 0.01 < cardinality_ratio < 0.5:
                # Check if it's categorical or discrete numeric
                if not pd.api.types.is_float_dtype(data[col]):
                    quasi_identifiers.append(col)

        return quasi_identifiers

    def _detect_sensitive_columns(self, data: pd.DataFrame) -> List[str]:
        """Detect potential sensitive attribute columns."""
        sensitive = []

        for col in data.columns:
            col_lower = col.lower()

            if any(pattern in col_lower for pattern in self.SENSITIVE_PATTERNS):
                sensitive.append(col)

        return sensitive

    def _estimate_k_anonymity(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
    ) -> int:
        """Estimate the k-anonymity level of the data.

        k-anonymity means each combination of quasi-identifiers appears
        at least k times.
        """
        if not quasi_identifiers:
            return len(data)  # No quasi-identifiers means k = n

        # Filter to existing columns
        qi_cols = [c for c in quasi_identifiers if c in data.columns]
        if not qi_cols:
            return len(data)

        # Group by quasi-identifiers and find minimum group size
        group_sizes = data.groupby(qi_cols).size()

        if len(group_sizes) == 0:
            return len(data)

        return int(group_sizes.min())

    def _compute_uniqueness_risk(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
    ) -> float:
        """Compute risk from unique records based on quasi-identifiers."""
        if not quasi_identifiers:
            return 0.0

        qi_cols = [c for c in quasi_identifiers if c in data.columns]
        if not qi_cols:
            return 0.0

        # Count unique combinations
        group_sizes = data.groupby(qi_cols).size()

        # Risk is the proportion of records in unique groups
        n_unique_records = (group_sizes == 1).sum()
        total_records = len(data)

        if total_records == 0:
            return 0.0

        return n_unique_records / total_records

    def _compute_rare_category_risk(
        self,
        data: pd.DataFrame,
    ) -> Tuple[float, Dict[str, List[str]]]:
        """Compute risk from rare categories in categorical columns."""
        rare_details: Dict[str, List[str]] = {}
        total_rare_risk = 0.0
        n_categorical = 0

        for col in data.columns:
            if data[col].dtype == object or str(data[col].dtype) == "category":
                value_counts = data[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < self.rare_threshold].index.tolist()

                if rare_categories:
                    rare_details[col] = rare_categories

                # Risk is the proportion of records in rare categories
                rare_ratio = value_counts[value_counts < self.rare_threshold].sum()
                total_rare_risk += rare_ratio
                n_categorical += 1

        if n_categorical == 0:
            return 0.0, rare_details

        return total_rare_risk / n_categorical, rare_details

    def _compute_reidentification_risk(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        k_anonymity: int,
    ) -> float:
        """Compute re-identification risk estimate."""
        if k_anonymity >= len(data):
            return 0.0

        # Simple model: risk is inversely proportional to k-anonymity
        # With adjustment for number of quasi-identifiers
        base_risk = 1.0 / max(k_anonymity, 1)

        # More quasi-identifiers increase risk
        n_qi = len([c for c in quasi_identifiers if c in data.columns])
        qi_factor = min(1.0, 0.1 * n_qi)

        return min(1.0, base_risk + qi_factor)

    def _compute_overall_risk(
        self,
        uniqueness_risk: float,
        rare_risk: float,
        reidentification_risk: float,
        k_anonymity: int,
    ) -> float:
        """Compute overall privacy risk score."""
        # Weighted combination of risk factors
        weights = {
            "uniqueness": 0.3,
            "rare": 0.2,
            "reidentification": 0.3,
            "k_anonymity": 0.2,
        }

        # Convert k-anonymity to a 0-1 risk (lower k = higher risk)
        k_risk = 1.0 / max(k_anonymity, 1)
        k_risk = min(1.0, k_risk * 10)  # Scale so k=10 gives risk ~1

        overall = (
            weights["uniqueness"] * uniqueness_risk
            + weights["rare"] * rare_risk
            + weights["reidentification"] * reidentification_risk
            + weights["k_anonymity"] * k_risk
        )

        return min(1.0, overall)

    def _compute_column_uniqueness(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute uniqueness ratio for each column."""
        result = {}
        for col in data.columns:
            n_unique = data[col].nunique()
            n_total = len(data[col].dropna())
            result[col] = n_unique / n_total if n_total > 0 else 0.0
        return result


def assess_privacy_risk(
    data: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    sensitive_columns: Optional[List[str]] = None,
) -> PrivacyRiskAssessment:
    """Convenience function to assess privacy risk in data.

    Args:
        data: DataFrame to analyze
        quasi_identifiers: Optional list of quasi-identifier columns
        sensitive_columns: Optional list of sensitive columns

    Returns:
        PrivacyRiskAssessment with risk analysis
    """
    analyzer = PrivacyAnalyzer()
    return analyzer.analyze(data, quasi_identifiers, sensitive_columns)
