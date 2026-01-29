"""Data Quality Dashboard for visualizing synthetic data quality.

This module provides an interactive web dashboard for comparing
real and synthetic data quality metrics.

Example:
    >>> from genesis.dashboard import QualityDashboard
    >>>
    >>> dashboard = QualityDashboard(real_data, synthetic_data)
    >>> dashboard.run(port=8050)  # Opens browser to localhost:8050
"""

import base64
import io
from typing import Any, Dict, Optional, Union

import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class QualityDashboard:
    """Interactive dashboard for synthetic data quality visualization.

    Features:
    - Distribution comparisons (histograms, KDE plots)
    - Correlation matrix comparison
    - Statistical metrics summary
    - Privacy risk heatmap
    - ML utility comparison
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        title: str = "Genesis Quality Dashboard",
    ) -> None:
        """Initialize the dashboard.

        Args:
            real_data: Original/real DataFrame
            synthetic_data: Synthetic DataFrame
            title: Dashboard title
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.title = title
        self._metrics: Optional[Dict[str, Any]] = None
        self._app: Optional[Any] = None

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all quality metrics.

        Returns:
            Dictionary of computed metrics
        """
        if self._metrics is not None:
            return self._metrics

        from genesis.evaluation.evaluator import QualityEvaluator

        evaluator = QualityEvaluator(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
        )
        report = evaluator.evaluate()
        self._metrics = report.to_dict()

        return self._metrics

    def generate_html_report(self) -> str:
        """Generate a static HTML report.

        Returns:
            HTML string with complete report
        """
        metrics = self.compute_metrics()

        # Generate plots as base64
        distribution_plots = self._generate_distribution_plots()
        correlation_plot = self._generate_correlation_plot()

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; margin-bottom: 30px; border-radius: 8px; }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card h2 {{ color: #667eea; margin-bottom: 15px; font-size: 1.4em; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        .metric-score {{ padding: 5px 10px; border-radius: 20px; display: inline-block; margin-top: 5px; font-size: 0.8em; }}
        .score-good {{ background: #d4edda; color: #155724; }}
        .score-medium {{ background: #fff3cd; color: #856404; }}
        .score-bad {{ background: #f8d7da; color: #721c24; }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{ max-width: 100%; height: auto; border-radius: 6px; }}
        .columns {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .column {{ flex: 1; min-width: 300px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .progress-bar {{ height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.title}</h1>
            <p class="subtitle">Synthetic Data Quality Assessment Report</p>
        </header>

        <div class="card">
            <h2>📊 Overall Quality Score</h2>
            <div class="metrics-grid">
                {self._render_overall_score(metrics)}
            </div>
        </div>

        <div class="columns">
            <div class="column">
                <div class="card">
                    <h2>📈 Statistical Fidelity</h2>
                    {self._render_statistical_metrics(metrics)}
                </div>
            </div>
            <div class="column">
                <div class="card">
                    <h2>🤖 ML Utility</h2>
                    {self._render_ml_utility(metrics)}
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🔒 Privacy Metrics</h2>
            {self._render_privacy_metrics(metrics)}
        </div>

        <div class="card">
            <h2>📊 Distribution Comparisons</h2>
            <div class="plot-container">
                {distribution_plots}
            </div>
        </div>

        <div class="card">
            <h2>🔗 Correlation Comparison</h2>
            <div class="plot-container">
                {correlation_plot}
            </div>
        </div>

        <div class="card">
            <h2>📋 Column Summary</h2>
            {self._render_column_summary()}
        </div>

        <footer style="text-align: center; padding: 20px; color: #666;">
            Generated by Genesis Synthetic Data Platform
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _render_overall_score(self, metrics: Dict[str, Any]) -> str:
        """Render overall score section."""
        overall = metrics.get("overall_score", 0)

        return f"""
        <div class="metric">
            <div class="metric-value">{overall:.1%}</div>
            <div class="metric-label">Overall Quality Score</div>
            <div class="progress-bar"><div class="progress-fill" style="width: {overall*100}%"></div></div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(self.real_data):,}</div>
            <div class="metric-label">Real Data Rows</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(self.synthetic_data):,}</div>
            <div class="metric-label">Synthetic Data Rows</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(self.real_data.columns)}</div>
            <div class="metric-label">Columns</div>
        </div>
        """

    def _render_statistical_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render statistical metrics."""
        stats = metrics.get("statistical", {})
        rows = ""
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                rows += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"

        return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"

    def _render_ml_utility(self, metrics: Dict[str, Any]) -> str:
        """Render ML utility metrics."""
        ml = metrics.get("ml_utility", {})
        rows = ""
        for key, value in ml.items():
            if isinstance(value, (int, float)):
                rows += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"

        if not rows:
            return "<p>ML utility metrics not computed</p>"

        return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"

    def _render_privacy_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render privacy metrics."""
        privacy = metrics.get("privacy", {})
        rows = ""
        for key, value in privacy.items():
            if isinstance(value, (int, float)):
                score_class = (
                    "score-good" if value < 0.1 else "score-medium" if value < 0.3 else "score-bad"
                )
                rows += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value:.4f}</td>
                    <td><span class="metric-score {score_class}">{'Low Risk' if value < 0.1 else 'Medium Risk' if value < 0.3 else 'High Risk'}</span></td>
                </tr>"""

        if not rows:
            return "<p>Privacy metrics not computed</p>"

        return f"<table><thead><tr><th>Metric</th><th>Value</th><th>Risk Level</th></tr></thead><tbody>{rows}</tbody></table>"

    def _render_column_summary(self) -> str:
        """Render column-by-column summary."""
        rows = ""
        for col in self.real_data.columns:
            if col not in self.synthetic_data.columns:
                continue

            real_col = self.real_data[col]
            synth_col = self.synthetic_data[col]

            dtype = str(real_col.dtype)
            real_nulls = real_col.isna().sum()
            synth_nulls = synth_col.isna().sum()
            real_unique = real_col.nunique()
            synth_unique = synth_col.nunique()

            rows += f"""
            <tr>
                <td>{col}</td>
                <td>{dtype}</td>
                <td>{real_unique}</td>
                <td>{synth_unique}</td>
                <td>{real_nulls}</td>
                <td>{synth_nulls}</td>
            </tr>"""

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Real Unique</th>
                    <th>Synth Unique</th>
                    <th>Real Nulls</th>
                    <th>Synth Nulls</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        """

    def _generate_distribution_plots(self) -> str:
        """Generate distribution comparison plots."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            numeric_cols = self.real_data.select_dtypes(include=["number"]).columns[:6]

            if len(numeric_cols) == 0:
                return "<p>No numeric columns to visualize</p>"

            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]
            elif n_cols == 1:
                axes = [[ax] for ax in axes]

            for idx, col in enumerate(numeric_cols):
                row_idx = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row_idx][col_idx]

                ax.hist(
                    self.real_data[col].dropna(), bins=30, alpha=0.5, label="Real", density=True
                )
                ax.hist(
                    self.synthetic_data[col].dropna(),
                    bins=30,
                    alpha=0.5,
                    label="Synthetic",
                    density=True,
                )
                ax.set_title(col)
                ax.legend()

            # Hide empty subplots
            for idx in range(len(numeric_cols), n_rows * n_cols):
                row_idx = idx // n_cols
                col_idx = idx % n_cols
                axes[row_idx][col_idx].set_visible(False)

            plt.tight_layout()

            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close()

            return f'<img src="data:image/png;base64,{img_base64}" alt="Distribution Comparison">'

        except ImportError:
            return "<p>Matplotlib required for plots. Install with: pip install matplotlib</p>"

    def _generate_correlation_plot(self) -> str:
        """Generate correlation matrix comparison."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            numeric_cols = self.real_data.select_dtypes(include=["number"]).columns

            if len(numeric_cols) < 2:
                return "<p>Need at least 2 numeric columns for correlation</p>"

            real_corr = self.real_data[numeric_cols].corr()
            synth_corr = self.synthetic_data[numeric_cols].corr()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(real_corr, cmap="coolwarm", vmin=-1, vmax=1)
            axes[0].set_title("Real Data Correlation")
            axes[0].set_xticks(range(len(numeric_cols)))
            axes[0].set_yticks(range(len(numeric_cols)))
            axes[0].set_xticklabels(numeric_cols, rotation=45, ha="right")
            axes[0].set_yticklabels(numeric_cols)

            im2 = axes[1].imshow(synth_corr, cmap="coolwarm", vmin=-1, vmax=1)
            axes[1].set_title("Synthetic Data Correlation")
            axes[1].set_xticks(range(len(numeric_cols)))
            axes[1].set_yticks(range(len(numeric_cols)))
            axes[1].set_xticklabels(numeric_cols, rotation=45, ha="right")
            axes[1].set_yticklabels(numeric_cols)

            fig.colorbar(im2, ax=axes, shrink=0.8)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close()

            return f'<img src="data:image/png;base64,{img_base64}" alt="Correlation Comparison">'

        except ImportError:
            return "<p>Matplotlib required for plots. Install with: pip install matplotlib</p>"

    def save_report(self, path: str) -> None:
        """Save HTML report to file.

        Args:
            path: Output file path
        """
        html = self.generate_html_report()
        with open(path, "w") as f:
            f.write(html)
        logger.info(f"Saved report to {path}")

    def save_pdf(self, path: str) -> None:
        """Save report as PDF.

        Args:
            path: Output PDF file path
        """
        try:
            from weasyprint import HTML

            html = self.generate_html_report()
            HTML(string=html).write_pdf(path)
            logger.info(f"Saved PDF report to {path}")
        except ImportError:
            logger.warning("weasyprint not installed. Install with: pip install weasyprint")
            # Fallback: save as HTML
            html_path = path.replace(".pdf", ".html")
            self.save_report(html_path)
            logger.info(f"Saved HTML report to {html_path} (PDF requires weasyprint)")

    def generate_plotly_figures(self) -> Dict[str, Any]:
        """Generate interactive Plotly figures.

        Returns:
            Dict of figure names to Plotly figure objects
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly not installed. Install with: pip install plotly")
            return {}

        figures = {}

        # Distribution comparison figures
        numeric_cols = self.real_data.select_dtypes(include=["number"]).columns[:8]

        if len(numeric_cols) > 0:
            n_cols = min(2, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig = make_subplots(
                rows=n_rows, cols=n_cols, subplot_titles=[str(c) for c in numeric_cols]
            )

            for idx, col in enumerate(numeric_cols):
                row = idx // n_cols + 1
                col_idx = idx % n_cols + 1

                fig.add_trace(
                    go.Histogram(
                        x=self.real_data[col].dropna(),
                        name="Real",
                        opacity=0.6,
                        marker_color="#667eea",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Histogram(
                        x=self.synthetic_data[col].dropna(),
                        name="Synthetic",
                        opacity=0.6,
                        marker_color="#764ba2",
                        histnorm="probability density",
                    ),
                    row=row,
                    col=col_idx,
                )

            fig.update_layout(
                title="Distribution Comparison",
                barmode="overlay",
                height=300 * n_rows,
                showlegend=True,
            )
            figures["distributions"] = fig

        # Correlation heatmaps
        if len(numeric_cols) >= 2:
            real_corr = self.real_data[numeric_cols].corr()
            synth_corr = self.synthetic_data[numeric_cols].corr()

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Real Data Correlation", "Synthetic Data Correlation"],
            )

            fig.add_trace(
                go.Heatmap(
                    z=real_corr.values,
                    x=list(numeric_cols),
                    y=list(numeric_cols),
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=synth_corr.values,
                    x=list(numeric_cols),
                    y=list(numeric_cols),
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                ),
                row=1,
                col=2,
            )

            fig.update_layout(title="Correlation Comparison", height=500)
            figures["correlation"] = fig

        # Metrics gauge chart
        metrics = self.compute_metrics()
        overall_score = metrics.get("overall_score", 0)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_score * 100,
                title={"text": "Overall Quality Score"},
                delta={"reference": 80},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [0, 50], "color": "#f8d7da"},
                        {"range": [50, 80], "color": "#fff3cd"},
                        {"range": [80, 100], "color": "#d4edda"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80,
                    },
                },
            )
        )
        fig.update_layout(height=300)
        figures["quality_gauge"] = fig

        return figures

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        open_browser: bool = True,
    ) -> None:
        """Run interactive dashboard server.

        Args:
            host: Host to bind to
            port: Port to bind to
            open_browser: Whether to open browser automatically
        """
        try:
            import webbrowser

            from flask import Flask

            # Save report to temp file
            html = self.generate_html_report()

            app = Flask(__name__)

            @app.route("/")
            def index():
                return html

            if open_browser:
                webbrowser.open(f"http://{host}:{port}")

            logger.info(f"Dashboard running at http://{host}:{port}")
            app.run(host=host, port=port, debug=False)

        except ImportError:
            logger.warning("Flask not installed. Saving static HTML instead.")
            self.save_report("quality_report.html")
            print("Report saved to quality_report.html")


class InteractiveDashboard:
    """Full-featured interactive dashboard with FastAPI backend.

    This provides a more sophisticated dashboard experience with:
    - Real-time metric updates
    - Interactive Plotly charts
    - Column-level drill-down
    - Privacy parameter simulation
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        title: str = "Genesis Interactive Dashboard",
    ) -> None:
        """Initialize interactive dashboard."""
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.title = title
        self._basic_dashboard = QualityDashboard(real_data, synthetic_data, title)

    def run(
        self,
        host: str = "127.0.0.1",
        port: int = 8050,
        open_browser: bool = True,
    ) -> None:
        """Run the interactive dashboard server."""
        try:
            import webbrowser

            import uvicorn
            from fastapi import FastAPI
            from fastapi.responses import HTMLResponse, JSONResponse
        except ImportError:
            logger.warning("FastAPI/uvicorn not installed. Falling back to basic dashboard.")
            self._basic_dashboard.run(host, port, open_browser)
            return

        app = FastAPI(title=self.title)

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return self._basic_dashboard.generate_html_report()

        @app.get("/api/metrics")
        async def get_metrics():
            return JSONResponse(self._basic_dashboard.compute_metrics())

        @app.get("/api/column/{column_name}")
        async def get_column_details(column_name: str):
            if column_name not in self.real_data.columns:
                return JSONResponse({"error": "Column not found"}, status_code=404)

            real_col = self.real_data[column_name]
            synth_col = self.synthetic_data[column_name]

            details = {
                "column": column_name,
                "dtype": str(real_col.dtype),
                "real": {
                    "count": int(len(real_col)),
                    "null_count": int(real_col.isna().sum()),
                    "unique": int(real_col.nunique()),
                },
                "synthetic": {
                    "count": int(len(synth_col)),
                    "null_count": int(synth_col.isna().sum()),
                    "unique": int(synth_col.nunique()),
                },
            }

            if pd.api.types.is_numeric_dtype(real_col):
                details["real"].update(
                    {
                        "mean": float(real_col.mean()),
                        "std": float(real_col.std()),
                        "min": float(real_col.min()),
                        "max": float(real_col.max()),
                    }
                )
                details["synthetic"].update(
                    {
                        "mean": float(synth_col.mean()),
                        "std": float(synth_col.std()),
                        "min": float(synth_col.min()),
                        "max": float(synth_col.max()),
                    }
                )

            return JSONResponse(details)

        @app.get("/api/figures/{figure_name}")
        async def get_figure(figure_name: str):
            figures = self._basic_dashboard.generate_plotly_figures()
            if figure_name not in figures:
                return JSONResponse({"error": "Figure not found"}, status_code=404)
            return JSONResponse(figures[figure_name].to_json())

        if open_browser:
            import threading

            threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()

        logger.info(f"Interactive dashboard running at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning")


def create_dashboard(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    output_path: Optional[str] = None,
    run_server: bool = False,
    interactive: bool = False,
) -> Union[str, None]:
    """Convenience function to create a quality dashboard.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        output_path: Path to save HTML report
        run_server: Whether to run interactive server
        interactive: Use full interactive dashboard (requires FastAPI)

    Returns:
        HTML string if output_path is None, else None
    """
    if interactive and run_server:
        dashboard = InteractiveDashboard(real_data, synthetic_data)
        dashboard.run()
        return None

    dashboard = QualityDashboard(real_data, synthetic_data)

    if run_server:
        dashboard.run()
        return None
    elif output_path:
        if output_path.endswith(".pdf"):
            dashboard.save_pdf(output_path)
        else:
            dashboard.save_report(output_path)
        return None
    else:
        return dashboard.generate_html_report()
