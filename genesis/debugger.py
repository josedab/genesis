"""Synthetic data debugger for diagnosing quality issues.

This module provides interactive tools to diagnose and fix synthetic data
quality problems, including distribution mismatches, correlation issues,
and constraint violations.

Example:
    >>> from genesis.debugger import SyntheticDebugger
    >>>
    >>> debugger = SyntheticDebugger(real_data, synthetic_data)
    >>> report = debugger.diagnose()
    >>>
    >>> # View problematic columns
    >>> for issue in report.issues:
    ...     print(f"{issue.column}: {issue.description}")
    ...     print(f"  Suggestion: {issue.suggestion}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class IssueSeverity(Enum):
    """Severity levels for quality issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of quality issues."""

    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    CONSTRAINT = "constraint"
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    CARDINALITY = "cardinality"
    RANGE = "range"
    PATTERN = "pattern"


@dataclass
class QualityIssue:
    """A single quality issue identified by the debugger."""

    column: Optional[str]
    category: IssueCategory
    severity: IssueSeverity
    description: str
    details: Dict[str, Any]
    suggestion: str
    fix_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "details": self.details,
            "suggestion": self.suggestion,
            "fix_available": self.fix_available,
        }


@dataclass
class ColumnDiagnostic:
    """Diagnostic information for a single column."""

    column: str
    dtype: str
    real_stats: Dict[str, Any]
    synthetic_stats: Dict[str, Any]
    similarity_score: float
    issues: List[QualityIssue]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "dtype": self.dtype,
            "real_stats": self.real_stats,
            "synthetic_stats": self.synthetic_stats,
            "similarity_score": self.similarity_score,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class DebugReport:
    """Complete debug report for synthetic data."""

    overall_score: float
    n_issues: int
    critical_issues: int
    column_diagnostics: List[ColumnDiagnostic]
    correlation_issues: List[QualityIssue]
    global_issues: List[QualityIssue]
    suggestions: List[str]

    @property
    def issues(self) -> List[QualityIssue]:
        """Get all issues across columns."""
        all_issues = list(self.global_issues)
        all_issues.extend(self.correlation_issues)
        for cd in self.column_diagnostics:
            all_issues.extend(cd.issues)
        return all_issues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "n_issues": self.n_issues,
            "critical_issues": self.critical_issues,
            "column_diagnostics": [c.to_dict() for c in self.column_diagnostics],
            "correlation_issues": [i.to_dict() for i in self.correlation_issues],
            "global_issues": [i.to_dict() for i in self.global_issues],
            "suggestions": self.suggestions,
        }

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        print("\n" + "=" * 60)
        print("SYNTHETIC DATA QUALITY DEBUG REPORT")
        print("=" * 60)
        print(f"\nOverall Quality Score: {self.overall_score:.2%}")
        print(f"Total Issues Found: {self.n_issues}")
        print(f"Critical Issues: {self.critical_issues}")

        if self.critical_issues > 0:
            print("\n⚠️  CRITICAL ISSUES:")
            for issue in self.issues:
                if issue.severity == IssueSeverity.CRITICAL:
                    print(f"  • [{issue.column or 'GLOBAL'}] {issue.description}")

        print("\n📊 COLUMN SUMMARY:")
        for cd in sorted(self.column_diagnostics, key=lambda x: x.similarity_score):
            status = (
                "✅" if cd.similarity_score > 0.8 else "⚠️" if cd.similarity_score > 0.6 else "❌"
            )
            print(f"  {status} {cd.column}: {cd.similarity_score:.2%} similarity")
            if cd.issues:
                for issue in cd.issues[:2]:  # Show top 2 issues
                    print(f"     └─ {issue.description}")

        if self.suggestions:
            print("\n💡 TOP SUGGESTIONS:")
            for i, suggestion in enumerate(self.suggestions[:5], 1):
                print(f"  {i}. {suggestion}")

        print("\n" + "=" * 60)


class SyntheticDebugger:
    """Debugger for synthetic data quality issues.

    Analyzes synthetic data against real data to identify problems
    and provide actionable suggestions for improvement.
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
    ):
        """Initialize debugger.

        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            discrete_columns: List of categorical columns
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.discrete_columns = discrete_columns or []

        # Infer discrete columns if not provided
        if not self.discrete_columns:
            self.discrete_columns = self._infer_discrete_columns()

    def diagnose(self) -> DebugReport:
        """Run full diagnostic analysis.

        Returns:
            DebugReport with all identified issues
        """
        logger.info("Starting synthetic data diagnosis...")

        # Analyze each column
        column_diagnostics = []
        for col in self.real_data.columns:
            if col in self.synthetic_data.columns:
                diagnostic = self._diagnose_column(col)
                column_diagnostics.append(diagnostic)

        # Analyze correlations
        correlation_issues = self._diagnose_correlations()

        # Analyze global issues
        global_issues = self._diagnose_global()

        # Calculate overall score
        col_scores = [cd.similarity_score for cd in column_diagnostics]
        overall_score = np.mean(col_scores) if col_scores else 0.0

        # Count issues
        all_issues = list(global_issues) + list(correlation_issues)
        for cd in column_diagnostics:
            all_issues.extend(cd.issues)

        n_issues = len(all_issues)
        critical_issues = sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL)

        # Generate suggestions
        suggestions = self._generate_suggestions(
            column_diagnostics, correlation_issues, global_issues
        )

        report = DebugReport(
            overall_score=overall_score,
            n_issues=n_issues,
            critical_issues=critical_issues,
            column_diagnostics=column_diagnostics,
            correlation_issues=correlation_issues,
            global_issues=global_issues,
            suggestions=suggestions,
        )

        logger.info(f"Diagnosis complete: {n_issues} issues found")
        return report

    def diagnose_column(self, column: str) -> ColumnDiagnostic:
        """Diagnose a single column.

        Args:
            column: Column name to diagnose

        Returns:
            ColumnDiagnostic for the column
        """
        return self._diagnose_column(column)

    def compare_distributions(
        self,
        column: str,
    ) -> Dict[str, Any]:
        """Compare distributions between real and synthetic for a column.

        Args:
            column: Column to compare

        Returns:
            Dictionary with comparison statistics
        """
        real_col = self.real_data[column]
        syn_col = self.synthetic_data[column]

        result = {
            "column": column,
            "real": {},
            "synthetic": {},
            "comparison": {},
        }

        if pd.api.types.is_numeric_dtype(real_col):
            result["real"] = {
                "mean": float(real_col.mean()),
                "std": float(real_col.std()),
                "min": float(real_col.min()),
                "max": float(real_col.max()),
                "median": float(real_col.median()),
                "skew": float(real_col.skew()),
            }
            result["synthetic"] = {
                "mean": float(syn_col.mean()),
                "std": float(syn_col.std()),
                "min": float(syn_col.min()),
                "max": float(syn_col.max()),
                "median": float(syn_col.median()),
                "skew": float(syn_col.skew()),
            }

            # KS test
            ks_stat, ks_pval = stats.ks_2samp(real_col.dropna(), syn_col.dropna())
            result["comparison"] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "mean_diff": abs(result["real"]["mean"] - result["synthetic"]["mean"]),
                "std_diff": abs(result["real"]["std"] - result["synthetic"]["std"]),
            }
        else:
            real_counts = real_col.value_counts(normalize=True)
            syn_counts = syn_col.value_counts(normalize=True)

            result["real"] = {
                "n_unique": int(real_col.nunique()),
                "top_categories": dict(real_counts.head(5)),
            }
            result["synthetic"] = {
                "n_unique": int(syn_col.nunique()),
                "top_categories": dict(syn_counts.head(5)),
            }

            # Category overlap
            real_cats = set(real_counts.index)
            syn_cats = set(syn_counts.index)
            overlap = len(real_cats & syn_cats) / len(real_cats | syn_cats)

            result["comparison"] = {
                "category_overlap": float(overlap),
                "missing_categories": list(real_cats - syn_cats),
                "extra_categories": list(syn_cats - real_cats),
            }

        return result

    def suggest_fixes(self, issue: QualityIssue) -> List[str]:
        """Get detailed fix suggestions for an issue.

        Args:
            issue: The issue to get fixes for

        Returns:
            List of specific fix suggestions
        """
        fixes = []

        if issue.category == IssueCategory.DISTRIBUTION:
            fixes.extend(
                [
                    "Increase training epochs for better distribution learning",
                    "Try a different generator method (e.g., TVAE instead of CTGAN)",
                    f"Mark '{issue.column}' as discrete if it has few unique values",
                ]
            )

        elif issue.category == IssueCategory.CORRELATION:
            fixes.extend(
                [
                    "Increase embedding dimension for better relationship capture",
                    "Try Gaussian Copula which explicitly models correlations",
                    "Check for missing values that may break correlations",
                ]
            )

        elif issue.category == IssueCategory.RANGE:
            fixes.extend(
                [
                    f"Add range constraint: Constraint.range('{issue.column}', min, max)",
                    "Use clipping in post-processing",
                    "Normalize data before training",
                ]
            )

        elif issue.category == IssueCategory.CARDINALITY:
            fixes.extend(
                [
                    f"Mark '{issue.column}' as discrete column",
                    "Increase batch size for better category learning",
                    "Check for rare categories that may need binning",
                ]
            )

        elif issue.category == IssueCategory.MISSING_VALUES:
            fixes.extend(
                [
                    "Check if synthetic data handles nulls correctly",
                    "Consider imputing missing values before training",
                    "Verify discrete column specification",
                ]
            )

        return fixes

    def _infer_discrete_columns(self) -> List[str]:
        """Infer which columns are discrete/categorical."""
        discrete = []
        for col in self.real_data.columns:
            if not pd.api.types.is_numeric_dtype(self.real_data[col]):
                discrete.append(col)
            elif self.real_data[col].nunique() < 20:
                discrete.append(col)
        return discrete

    def _diagnose_column(self, column: str) -> ColumnDiagnostic:
        """Diagnose a single column."""
        real_col = self.real_data[column]
        syn_col = self.synthetic_data[column]
        issues = []

        is_numeric = pd.api.types.is_numeric_dtype(real_col)

        # Compute statistics
        if is_numeric:
            real_stats = {
                "mean": float(real_col.mean()),
                "std": float(real_col.std()),
                "min": float(real_col.min()),
                "max": float(real_col.max()),
                "null_rate": float(real_col.isna().mean()),
            }
            syn_stats = {
                "mean": float(syn_col.mean()),
                "std": float(syn_col.std()),
                "min": float(syn_col.min()),
                "max": float(syn_col.max()),
                "null_rate": float(syn_col.isna().mean()),
            }

            # KS test for similarity
            try:
                ks_stat, _ = stats.ks_2samp(real_col.dropna(), syn_col.dropna())
                similarity = 1 - ks_stat
            except Exception:
                similarity = 0.5

            # Check for issues

            # Range violations
            if (
                syn_stats["min"] < real_stats["min"] * 0.9
                or syn_stats["max"] > real_stats["max"] * 1.1
            ):
                issues.append(
                    QualityIssue(
                        column=column,
                        category=IssueCategory.RANGE,
                        severity=IssueSeverity.WARNING,
                        description=f"Range mismatch: synthetic [{syn_stats['min']:.2f}, {syn_stats['max']:.2f}] vs real [{real_stats['min']:.2f}, {real_stats['max']:.2f}]",
                        details={
                            "real_range": [real_stats["min"], real_stats["max"]],
                            "syn_range": [syn_stats["min"], syn_stats["max"]],
                        },
                        suggestion="Add range constraint to enforce bounds",
                        fix_available=True,
                    )
                )

            # Mean shift
            mean_diff = abs(syn_stats["mean"] - real_stats["mean"])
            if real_stats["std"] > 0 and mean_diff / real_stats["std"] > 0.5:
                issues.append(
                    QualityIssue(
                        column=column,
                        category=IssueCategory.DISTRIBUTION,
                        severity=IssueSeverity.WARNING,
                        description=f"Mean shift: {mean_diff:.2f} ({mean_diff/real_stats['std']:.1f} std deviations)",
                        details={"real_mean": real_stats["mean"], "syn_mean": syn_stats["mean"]},
                        suggestion="Increase training epochs or try different generator",
                    )
                )

            # Variance mismatch
            if real_stats["std"] > 0:
                std_ratio = syn_stats["std"] / real_stats["std"]
                if std_ratio < 0.5 or std_ratio > 2.0:
                    issues.append(
                        QualityIssue(
                            column=column,
                            category=IssueCategory.DISTRIBUTION,
                            severity=IssueSeverity.WARNING,
                            description=f"Variance mismatch: ratio {std_ratio:.2f}",
                            details={"real_std": real_stats["std"], "syn_std": syn_stats["std"]},
                            suggestion="Generator may be mode-collapsing; try different architecture",
                        )
                    )

        else:
            # Categorical column
            real_counts = real_col.value_counts(normalize=True)
            syn_counts = syn_col.value_counts(normalize=True)

            real_stats = {
                "n_unique": int(real_col.nunique()),
                "top_value": str(real_counts.index[0]) if len(real_counts) > 0 else None,
                "null_rate": float(real_col.isna().mean()),
            }
            syn_stats = {
                "n_unique": int(syn_col.nunique()),
                "top_value": str(syn_counts.index[0]) if len(syn_counts) > 0 else None,
                "null_rate": float(syn_col.isna().mean()),
            }

            # Category overlap
            real_cats = set(real_counts.index)
            syn_cats = set(syn_counts.index)
            overlap = len(real_cats & syn_cats) / max(len(real_cats | syn_cats), 1)
            similarity = overlap

            # Missing categories
            missing = real_cats - syn_cats
            if missing:
                issues.append(
                    QualityIssue(
                        column=column,
                        category=IssueCategory.CARDINALITY,
                        severity=IssueSeverity.WARNING if len(missing) < 3 else IssueSeverity.ERROR,
                        description=f"Missing {len(missing)} categories: {list(missing)[:5]}",
                        details={"missing": list(missing)},
                        suggestion="Increase batch size or training data for rare categories",
                    )
                )

            # Extra categories
            extra = syn_cats - real_cats
            if extra:
                issues.append(
                    QualityIssue(
                        column=column,
                        category=IssueCategory.CARDINALITY,
                        severity=IssueSeverity.ERROR,
                        description=f"Generated {len(extra)} invalid categories: {list(extra)[:5]}",
                        details={"extra": list(extra)},
                        suggestion="Mark column as discrete to constrain categories",
                        fix_available=True,
                    )
                )

        # Null rate mismatch
        null_diff = abs(syn_stats.get("null_rate", 0) - real_stats.get("null_rate", 0))
        if null_diff > 0.1:
            issues.append(
                QualityIssue(
                    column=column,
                    category=IssueCategory.MISSING_VALUES,
                    severity=IssueSeverity.WARNING,
                    description=f"Null rate mismatch: {null_diff:.1%} difference",
                    details={
                        "real_null_rate": real_stats.get("null_rate"),
                        "syn_null_rate": syn_stats.get("null_rate"),
                    },
                    suggestion="Check null handling in generator configuration",
                )
            )

        return ColumnDiagnostic(
            column=column,
            dtype="numeric" if is_numeric else "categorical",
            real_stats=real_stats,
            synthetic_stats=syn_stats,
            similarity_score=similarity,
            issues=issues,
        )

    def _diagnose_correlations(self) -> List[QualityIssue]:
        """Diagnose correlation structure issues."""
        issues = []

        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in self.synthetic_data.columns]

        if len(common_cols) < 2:
            return issues

        try:
            real_corr = self.real_data[common_cols].corr()
            syn_corr = self.synthetic_data[common_cols].corr()

            # Find significant correlation differences
            diff = np.abs(real_corr - syn_corr)

            for i, col1 in enumerate(common_cols):
                for j, col2 in enumerate(common_cols):
                    if i < j and diff.iloc[i, j] > 0.2:
                        real_val = real_corr.iloc[i, j]
                        syn_val = syn_corr.iloc[i, j]

                        issues.append(
                            QualityIssue(
                                column=f"{col1} × {col2}",
                                category=IssueCategory.CORRELATION,
                                severity=(
                                    IssueSeverity.WARNING
                                    if diff.iloc[i, j] < 0.4
                                    else IssueSeverity.ERROR
                                ),
                                description=f"Correlation mismatch: real={real_val:.2f}, synthetic={syn_val:.2f}",
                                details={
                                    "real_corr": float(real_val),
                                    "syn_corr": float(syn_val),
                                    "diff": float(diff.iloc[i, j]),
                                },
                                suggestion="Try Gaussian Copula for better correlation preservation",
                            )
                        )
        except Exception as e:
            logger.warning(f"Failed to analyze correlations: {e}")

        return issues

    def _diagnose_global(self) -> List[QualityIssue]:
        """Diagnose global/dataset-level issues."""
        issues = []

        # Row count check
        real_rows = len(self.real_data)
        syn_rows = len(self.synthetic_data)

        if syn_rows < real_rows * 0.1:
            issues.append(
                QualityIssue(
                    column=None,
                    category=IssueCategory.PATTERN,
                    severity=IssueSeverity.WARNING,
                    description=f"Synthetic data is much smaller ({syn_rows} vs {real_rows} rows)",
                    details={"real_rows": real_rows, "syn_rows": syn_rows},
                    suggestion="Generate more samples for statistical validity",
                )
            )

        # Duplicate check
        syn_dupes = self.synthetic_data.duplicated().sum()
        dupe_rate = syn_dupes / len(self.synthetic_data)
        if dupe_rate > 0.1:
            issues.append(
                QualityIssue(
                    column=None,
                    category=IssueCategory.PATTERN,
                    severity=IssueSeverity.ERROR,
                    description=f"High duplicate rate: {dupe_rate:.1%} of rows are duplicates",
                    details={"duplicate_count": int(syn_dupes), "duplicate_rate": float(dupe_rate)},
                    suggestion="Increase generator diversity (larger embedding, more epochs)",
                )
            )

        # Column mismatch
        missing_cols = set(self.real_data.columns) - set(self.synthetic_data.columns)
        if missing_cols:
            issues.append(
                QualityIssue(
                    column=None,
                    category=IssueCategory.PATTERN,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Missing columns in synthetic data: {missing_cols}",
                    details={"missing_columns": list(missing_cols)},
                    suggestion="Check generator output includes all columns",
                )
            )

        return issues

    def _generate_suggestions(
        self,
        column_diagnostics: List[ColumnDiagnostic],
        correlation_issues: List[QualityIssue],
        global_issues: List[QualityIssue],
    ) -> List[str]:
        """Generate prioritized suggestions based on issues."""
        suggestions = []

        # Critical issues first
        for issue in global_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                suggestions.append(f"CRITICAL: {issue.suggestion}")

        # Correlation issues
        if len(correlation_issues) > 3:
            suggestions.append(
                "Multiple correlation issues detected. Consider using Gaussian Copula "
                "generator which explicitly preserves correlation structure."
            )

        # Column-specific patterns
        range_issues = sum(
            1 for cd in column_diagnostics for i in cd.issues if i.category == IssueCategory.RANGE
        )
        if range_issues > 2:
            suggestions.append(
                f"Multiple range violations detected ({range_issues} columns). "
                "Add range constraints to enforce valid bounds."
            )

        cardinality_issues = sum(
            1
            for cd in column_diagnostics
            for i in cd.issues
            if i.category == IssueCategory.CARDINALITY
        )
        if cardinality_issues > 2:
            suggestions.append(
                f"Multiple category issues ({cardinality_issues} columns). "
                "Verify all categorical columns are marked as discrete."
            )

        # Low similarity scores
        low_sim_cols = [cd for cd in column_diagnostics if cd.similarity_score < 0.6]
        if low_sim_cols:
            cols = [cd.column for cd in low_sim_cols[:3]]
            suggestions.append(
                f"Columns with poor similarity: {', '.join(cols)}. "
                "Consider increasing epochs or trying TVAE generator."
            )

        # Default suggestion if no issues
        if not suggestions:
            suggestions.append(
                "Synthetic data quality looks good! Consider running quality evaluation "
                "for detailed metrics."
            )

        return suggestions


__all__ = [
    "SyntheticDebugger",
    "DebugReport",
    "ColumnDiagnostic",
    "QualityIssue",
    "IssueSeverity",
    "IssueCategory",
]
