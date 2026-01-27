"""Quality evaluator for synthetic data."""

from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.core.config import EvaluationConfig
from genesis.evaluation.ml_utility import compute_ml_utility
from genesis.evaluation.privacy import compute_privacy_metrics
from genesis.evaluation.report import QualityReport
from genesis.evaluation.statistical import compute_statistical_fidelity


class QualityEvaluator:
    """Evaluator for comparing synthetic data quality against real data.

    Computes comprehensive metrics including statistical fidelity,
    ML utility, and privacy metrics.

    Example:
        >>> evaluator = QualityEvaluator(real_data, synthetic_data)
        >>> report = evaluator.evaluate()
        >>> print(report.summary())
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        config: Optional[EvaluationConfig] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            real_data: Real/original DataFrame
            synthetic_data: Synthetic/generated DataFrame
            config: Evaluation configuration
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.config = config or EvaluationConfig()

        self._report: Optional[QualityReport] = None

    def evaluate(
        self,
        compute_statistical: bool = True,
        compute_ml: bool = True,
        compute_privacy: bool = True,
        target_column: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
        quasi_identifiers: Optional[List[str]] = None,
    ) -> QualityReport:
        """Run comprehensive evaluation.

        Args:
            compute_statistical: Whether to compute statistical fidelity
            compute_ml: Whether to compute ML utility metrics
            compute_privacy: Whether to compute privacy metrics
            target_column: Target column for ML evaluation
            sensitive_columns: Columns with sensitive data
            quasi_identifiers: Columns that could identify individuals

        Returns:
            QualityReport with all computed metrics
        """
        report = QualityReport(
            metadata={
                "n_real_samples": len(self.real_data),
                "n_synthetic_samples": len(self.synthetic_data),
                "n_columns": len(self.real_data.columns),
            }
        )

        # Statistical fidelity
        if compute_statistical:
            discrete_cols = self.real_data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            report.statistical_fidelity = compute_statistical_fidelity(
                self.real_data,
                self.synthetic_data,
                discrete_columns=discrete_cols,
            )

        # ML utility
        if compute_ml:
            target = target_column or self.config.target_column
            if target is None and len(self.real_data.columns) > 1:
                # Use last column as default target
                target = self.real_data.columns[-1]

            if target:
                report.ml_utility = compute_ml_utility(
                    self.real_data,
                    self.synthetic_data,
                    target_column=target,
                    random_state=self.config.random_seed,
                )

        # Privacy metrics
        if compute_privacy:
            report.privacy_metrics = compute_privacy_metrics(
                self.real_data,
                self.synthetic_data,
                sensitive_columns=sensitive_columns,
                quasi_identifiers=quasi_identifiers,
            )

        self._report = report
        return report

    def get_column_report(self, column: str) -> Dict[str, Any]:
        """Get detailed report for a specific column.

        Args:
            column: Column name

        Returns:
            Dictionary with column-specific metrics
        """
        if self._report is None:
            self.evaluate()

        result = {}

        if "column_metrics" in self._report.statistical_fidelity:
            result["statistical"] = self._report.statistical_fidelity["column_metrics"].get(
                column, {}
            )

        return result

    def compare_distributions(
        self,
        column: str,
    ) -> Dict[str, Any]:
        """Compare distributions for a specific column.

        Args:
            column: Column name

        Returns:
            Dictionary with distribution comparison
        """
        if column not in self.real_data.columns:
            return {"error": f"Column '{column}' not found in real data"}

        if column not in self.synthetic_data.columns:
            return {"error": f"Column '{column}' not found in synthetic data"}

        real_col = self.real_data[column]
        syn_col = self.synthetic_data[column]

        result = {
            "column": column,
            "dtype": str(real_col.dtype),
        }

        if pd.api.types.is_numeric_dtype(real_col):
            result["real_stats"] = {
                "mean": float(real_col.mean()),
                "std": float(real_col.std()),
                "min": float(real_col.min()),
                "max": float(real_col.max()),
                "median": float(real_col.median()),
            }
            result["synthetic_stats"] = {
                "mean": float(syn_col.mean()),
                "std": float(syn_col.std()),
                "min": float(syn_col.min()),
                "max": float(syn_col.max()),
                "median": float(syn_col.median()),
            }
        else:
            result["real_value_counts"] = real_col.value_counts().head(10).to_dict()
            result["synthetic_value_counts"] = syn_col.value_counts().head(10).to_dict()

        return result


def evaluate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: Optional[str] = None,
) -> QualityReport:
    """Convenience function to evaluate synthetic data.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        target_column: Target column for ML evaluation

    Returns:
        QualityReport with evaluation results
    """
    evaluator = QualityEvaluator(real_data, synthetic_data)
    return evaluator.evaluate(target_column=target_column)
