# Quality Dashboard

Genesis provides interactive quality dashboards for visualizing and comparing real vs. synthetic data quality.

## Overview

| Class/Function | Output | Use Case |
|----------------|--------|----------|
| **QualityDashboard** | HTML report | Static reports |
| **create_dashboard()** | HTML string | Quick visualization |
| **generate_plotly_figures()** | Plotly figures | Custom dashboards |
| **InteractiveDashboard** | REST API | Real-time monitoring |

## Quick Start

```python
from genesis.dashboard import create_dashboard

# Generate HTML dashboard
html = create_dashboard(
    real_data=original_df,
    synthetic_data=synthetic_df,
    output_path="quality_report.html"
)
```

## QualityDashboard

Full-featured dashboard with comprehensive metrics.

```python
from genesis.dashboard import QualityDashboard

# Create dashboard
dashboard = QualityDashboard(
    real_data=original_df,
    synthetic_data=synthetic_df,
    name="Customer Data Quality Report",
)

# Compute all metrics
metrics = dashboard.compute_metrics()
print(f"Overall Score: {metrics['overall_score']:.2f}")
print(f"Statistical Fidelity: {metrics['statistical_fidelity']:.2f}")
print(f"ML Utility: {metrics['ml_utility']:.2f}")

# Generate HTML report
html = dashboard.generate_html_report()

# Save to file
dashboard.save_report("quality_dashboard.html")
```

### Computed Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `overall_score` | 0-1 | Weighted composite score |
| `statistical_fidelity` | 0-1 | Distribution similarity |
| `ml_utility` | 0-1 | Predictive performance |
| `privacy_score` | 0-1 | Re-identification risk |
| `coverage` | 0-1 | Value space coverage |

### Column-Level Metrics

```python
metrics = dashboard.compute_metrics()

for col, stats in metrics['column_metrics'].items():
    print(f"\n{col}:")
    print(f"  KS Statistic: {stats['ks_statistic']:.4f}")
    print(f"  Correlation: {stats['correlation']:.4f}")
    if 'js_divergence' in stats:
        print(f"  JS Divergence: {stats['js_divergence']:.4f}")
```

## Plotly Interactive Figures

Generate interactive Plotly visualizations:

```python
from genesis.dashboard import QualityDashboard

dashboard = QualityDashboard(real_data, synthetic_data)
figures = dashboard.generate_plotly_figures()

# Available figures
print(figures.keys())
# dict_keys(['distributions', 'correlations', 'pca', 'metrics_summary'])

# Display in Jupyter
figures['distributions'].show()
figures['correlations'].show()
figures['pca'].show()
```

### Distribution Comparison

```python
fig = figures['distributions']
# Shows overlaid histograms for each numeric column
# Real data in blue, synthetic in orange
```

### Correlation Heatmaps

```python
fig = figures['correlations']
# Side-by-side heatmaps of correlation matrices
# Left: real data, Right: synthetic data
```

### PCA Visualization

```python
fig = figures['pca']
# 2D PCA projection showing data overlap
# Points colored by source (real vs synthetic)
```

### Custom Plotly Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dashboard = QualityDashboard(real_data, synthetic_data)
figures = dashboard.generate_plotly_figures()

# Combine into single dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Distributions', 'Correlations', 'PCA', 'Metrics']
)

# Add traces from each figure
# ... customize as needed

fig.write_html("custom_dashboard.html")
```

## PDF Export

Export dashboard as PDF (requires `weasyprint`):

```python
from genesis.dashboard import QualityDashboard

dashboard = QualityDashboard(real_data, synthetic_data)

# Generate and save PDF
dashboard.save_pdf("quality_report.pdf")
```

Install dependency:
```bash
pip install weasyprint
```

## InteractiveDashboard

Run a live dashboard server for real-time monitoring:

```python
from genesis.dashboard import InteractiveDashboard

# Create interactive dashboard
interactive = InteractiveDashboard(
    real_data=original_df,
    synthetic_data=synthetic_df,
    host="0.0.0.0",
    port=8050,
)

# Start server
interactive.run()
# Dashboard available at http://localhost:8050
```

### REST API Endpoints

The interactive dashboard exposes REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML page |
| `/api/metrics` | GET | JSON metrics |
| `/api/columns` | GET | Column list |
| `/api/distribution/{column}` | GET | Column distribution data |
| `/api/refresh` | POST | Refresh with new data |

### API Usage

```python
import requests

# Get metrics
response = requests.get("http://localhost:8050/api/metrics")
metrics = response.json()

# Get column distribution
response = requests.get("http://localhost:8050/api/distribution/age")
dist_data = response.json()
```

### Update Data Dynamically

```python
# Update synthetic data
requests.post(
    "http://localhost:8050/api/refresh",
    json={"synthetic_data": new_synthetic_df.to_dict()}
)
```

## CLI Dashboard Command

Launch dashboard from command line:

```bash
# Basic usage
genesis dashboard real_data.csv synthetic_data.csv

# With output file
genesis dashboard real.csv synthetic.csv --output report.html

# Launch interactive server
genesis dashboard real.csv synthetic.csv --serve --port 8050
```

## Integration Examples

### With Generator Pipeline

```python
from genesis import SyntheticGenerator
from genesis.dashboard import QualityDashboard

# Generate synthetic data
generator = SyntheticGenerator(method="ctgan")
generator.fit(real_data)
synthetic = generator.generate(len(real_data))

# Create quality dashboard
dashboard = QualityDashboard(real_data, synthetic)
dashboard.save_report("generation_quality.html")

# Check quality threshold
metrics = dashboard.compute_metrics()
if metrics['overall_score'] < 0.8:
    print("Warning: Quality below threshold!")
```

### Batch Comparison

```python
from genesis.dashboard import QualityDashboard
import os

methods = ["gaussian_copula", "ctgan", "tvae"]
results = {}

for method in methods:
    generator = SyntheticGenerator(method=method)
    generator.fit(real_data)
    synthetic = generator.generate(10000)
    
    dashboard = QualityDashboard(real_data, synthetic, name=f"{method} Quality")
    metrics = dashboard.compute_metrics()
    results[method] = metrics['overall_score']
    
    dashboard.save_report(f"quality_{method}.html")

# Summary
print("\nMethod Comparison:")
for method, score in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {method}: {score:.3f}")
```

### Continuous Monitoring

```python
from genesis.dashboard import InteractiveDashboard
import schedule
import time

dashboard = InteractiveDashboard(real_data, initial_synthetic)

def refresh_synthetic():
    new_synthetic = generator.generate(10000)
    dashboard.update_synthetic(new_synthetic)
    print(f"Refreshed at {time.strftime('%H:%M:%S')}")

# Refresh every hour
schedule.every().hour.do(refresh_synthetic)

# Run dashboard in background
import threading
threading.Thread(target=dashboard.run, daemon=True).start()

# Keep refreshing
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Customization

### Custom Metrics

```python
from genesis.dashboard import QualityDashboard

class CustomDashboard(QualityDashboard):
    def compute_metrics(self):
        metrics = super().compute_metrics()
        
        # Add custom metric
        metrics['custom_score'] = self._compute_custom_metric()
        
        return metrics
    
    def _compute_custom_metric(self):
        # Your custom logic
        return 0.95

dashboard = CustomDashboard(real_data, synthetic_data)
```

### Custom Styling

```python
dashboard = QualityDashboard(
    real_data, 
    synthetic_data,
    theme="dark",  # or "light"
    colors={
        "real": "#1f77b4",
        "synthetic": "#ff7f0e",
        "good": "#2ca02c",
        "warning": "#ffbb00",
        "bad": "#d62728",
    }
)
```

## Configuration Reference

### QualityDashboard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `real_data` | DataFrame | required | Original dataset |
| `synthetic_data` | DataFrame | required | Synthetic dataset |
| `name` | str | "Quality Report" | Dashboard title |
| `include_privacy` | bool | True | Include privacy metrics |
| `include_ml_utility` | bool | True | Include ML utility tests |

### InteractiveDashboard

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "0.0.0.0" | Server bind address |
| `port` | int | 8050 | Server port |
| `debug` | bool | False | Enable debug mode |
| `auto_refresh` | int | None | Auto-refresh interval (seconds) |

## Best Practices

### 1. Compare Same-Size Samples
```python
# Ensure fair comparison
n_samples = min(len(real_data), len(synthetic_data))
dashboard = QualityDashboard(
    real_data.sample(n_samples),
    synthetic_data.sample(n_samples),
)
```

### 2. Set Quality Thresholds
```python
metrics = dashboard.compute_metrics()

thresholds = {
    'overall_score': 0.8,
    'statistical_fidelity': 0.85,
    'privacy_score': 0.95,
}

for metric, threshold in thresholds.items():
    if metrics[metric] < threshold:
        print(f"⚠️  {metric} ({metrics[metric]:.3f}) below threshold ({threshold})")
```

### 3. Archive Reports
```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dashboard.save_report(f"quality_reports/report_{timestamp}.html")
```
