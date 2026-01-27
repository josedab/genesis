"""Quality report generation."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class QualityReport:
    """Comprehensive quality report for synthetic data.

    Contains statistical fidelity, ML utility, and privacy metrics
    with support for multiple export formats.
    """

    statistical_fidelity: Dict[str, Any] = field(default_factory=dict)
    ml_utility: Dict[str, Any] = field(default_factory=dict)
    privacy_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metadata:
            self.metadata = {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
            }

    @property
    def overall_score(self) -> float:
        """Get overall quality score (0-100)."""
        scores = []

        if "overall" in self.statistical_fidelity:
            fidelity = self.statistical_fidelity["overall"].get("fidelity_score", 0)
            scores.append(fidelity * 100)

        if "utility_score" in self.ml_utility:
            scores.append(self.ml_utility["utility_score"] * 100)

        if "overall_privacy_score" in self.privacy_metrics:
            scores.append(self.privacy_metrics["overall_privacy_score"] * 100)

        return sum(scores) / len(scores) if scores else 0.0

    @property
    def fidelity_score(self) -> float:
        """Get statistical fidelity score (0-1)."""
        return self.statistical_fidelity.get("overall", {}).get("fidelity_score", 0.0)

    @property
    def utility_score(self) -> float:
        """Get ML utility score (0-1)."""
        return self.ml_utility.get("utility_score", 0.0)

    @property
    def privacy_score(self) -> float:
        """Get privacy score (0-1)."""
        return self.privacy_metrics.get("overall_privacy_score", 0.0)

    def summary(self) -> str:
        """Get a text summary of the report."""
        lines = [
            "=" * 50,
            "SYNTHETIC DATA QUALITY REPORT",
            "=" * 50,
            "",
            f"Overall Score: {self.overall_score:.1f}%",
            "",
            "--- Statistical Fidelity ---",
            f"Score: {self.fidelity_score * 100:.1f}%",
        ]

        # Column scores
        if "column_metrics" in self.statistical_fidelity:
            lines.append("\nColumn-wise scores:")
            for col, metrics in list(self.statistical_fidelity["column_metrics"].items())[:5]:
                score = metrics.get("score", 0) * 100
                lines.append(f"  {col}: {score:.1f}%")
            if len(self.statistical_fidelity["column_metrics"]) > 5:
                lines.append(
                    f"  ... and {len(self.statistical_fidelity['column_metrics']) - 5} more columns"
                )

        lines.extend(
            [
                "",
                "--- ML Utility ---",
                f"Score: {self.utility_score * 100:.1f}%",
            ]
        )

        if "tstr" in self.ml_utility:
            tstr = self.ml_utility["tstr"]
            if "metrics" in tstr:
                metric_name = "accuracy" if "accuracy" in tstr["metrics"] else "r2"
                metric_val = tstr["metrics"].get(metric_name, 0)
                lines.append(f"Train-Synthetic-Test-Real {metric_name}: {metric_val:.3f}")

        lines.extend(
            [
                "",
                "--- Privacy ---",
                f"Score: {self.privacy_score * 100:.1f}%",
            ]
        )

        if "reidentification" in self.privacy_metrics:
            risk = self.privacy_metrics["reidentification"].get("reidentification_risk", 0)
            lines.append(f"Re-identification Risk: {risk * 100:.2f}%")

        if "dcr" in self.privacy_metrics:
            mean_dcr = self.privacy_metrics["dcr"].get("mean_dcr", 0)
            lines.append(f"Mean Distance to Closest Record: {mean_dcr:.4f}")

        lines.extend(["", "=" * 50])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "overall_score": self.overall_score,
            "fidelity_score": self.fidelity_score,
            "utility_score": self.utility_score,
            "privacy_score": self.privacy_score,
            "statistical_fidelity": self.statistical_fidelity,
            "ml_utility": self.ml_utility,
            "privacy_metrics": self.privacy_metrics,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    def to_html(self) -> str:
        """Export report to HTML string."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Synthetic Data Quality Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #333; }",
            "h2 { color: #666; border-bottom: 1px solid #ccc; }",
            ".score { font-size: 24px; font-weight: bold; }",
            ".score.high { color: #28a745; }",
            ".score.medium { color: #ffc107; }",
            ".score.low { color: #dc3545; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f4f4f4; }",
            ".metric-bar { background: #e9ecef; border-radius: 4px; height: 20px; }",
            ".metric-fill { background: #007bff; height: 100%; border-radius: 4px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Synthetic Data Quality Report</h1>",
            f"<p>Generated: {self.metadata.get('generated_at', 'N/A')}</p>",
            "",
            "<h2>Overall Scores</h2>",
            "<table>",
            "<tr><th>Metric</th><th>Score</th><th>Visual</th></tr>",
        ]

        # Add score rows
        for name, score in [
            ("Overall", self.overall_score),
            ("Statistical Fidelity", self.fidelity_score * 100),
            ("ML Utility", self.utility_score * 100),
            ("Privacy", self.privacy_score * 100),
        ]:
            score_class = "high" if score >= 80 else "medium" if score >= 60 else "low"
            html_parts.extend(
                [
                    "<tr>",
                    f"<td>{name}</td>",
                    f"<td class='score {score_class}'>{score:.1f}%</td>",
                    f"<td><div class='metric-bar'><div class='metric-fill' style='width:{score}%'></div></div></td>",
                    "</tr>",
                ]
            )

        html_parts.extend(
            [
                "</table>",
                "",
                "<h2>Statistical Fidelity Details</h2>",
            ]
        )

        # Column metrics table
        if "column_metrics" in self.statistical_fidelity:
            html_parts.extend(
                [
                    "<table>",
                    "<tr><th>Column</th><th>Score</th><th>Details</th></tr>",
                ]
            )
            for col, metrics in self.statistical_fidelity["column_metrics"].items():
                score = metrics.get("score", 0) * 100
                details = ""
                if "ks_test" in metrics:
                    details = f"KS stat: {metrics['ks_test']['statistic']:.4f}"
                elif "chi_squared" in metrics:
                    details = f"Chi² stat: {metrics['chi_squared']['statistic']:.2f}"

                html_parts.append(f"<tr><td>{col}</td><td>{score:.1f}%</td><td>{details}</td></tr>")

            html_parts.append("</table>")

        html_parts.extend(
            [
                "",
                "<h2>ML Utility Details</h2>",
            ]
        )

        if "tstr" in self.ml_utility:
            tstr = self.ml_utility["tstr"]
            html_parts.append("<h3>Train-on-Synthetic, Test-on-Real (TSTR)</h3>")
            if "metrics" in tstr:
                html_parts.append("<ul>")
                for k, v in tstr["metrics"].items():
                    html_parts.append(f"<li>{k}: {v:.4f}</li>")
                html_parts.append("</ul>")

        html_parts.extend(
            [
                "",
                "<h2>Privacy Metrics</h2>",
            ]
        )

        if "reidentification" in self.privacy_metrics:
            reid = self.privacy_metrics["reidentification"]
            risk = reid.get("reidentification_risk", 0) * 100
            html_parts.append(f"<p><strong>Re-identification Risk:</strong> {risk:.2f}%</p>")

        if "dcr" in self.privacy_metrics:
            dcr = self.privacy_metrics["dcr"]
            html_parts.append(f"<p><strong>Mean DCR:</strong> {dcr.get('mean_dcr', 0):.4f}</p>")

        html_parts.extend(
            [
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html_parts)

    def save_html(self, path: str) -> None:
        """Save report to HTML file."""
        with open(path, "w") as f:
            f.write(self.to_html())

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"QualityReport(overall={self.overall_score:.1f}%, fidelity={self.fidelity_score:.2f}, utility={self.utility_score:.2f}, privacy={self.privacy_score:.2f})"
