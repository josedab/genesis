# Synthetic Data Observability

Genesis provides comprehensive observability features for monitoring synthetic data generation in production environments.

## Overview

| Component | Purpose |
|-----------|---------|
| **GenesisTracer** | Distributed tracing with OpenTelemetry |
| **MetricsCollector** | Metrics emission (Prometheus, StatsD) |
| **GenerationObserver** | Combined tracing + metrics |
| **DashboardExporter** | Export to Grafana, CloudWatch |

## Installation

```bash
pip install genesis-synth[observability]
```

## Quick Start

```python
from genesis.observability import create_observability_stack

# Create complete observability stack
stack = create_observability_stack(
    service_name="genesis-prod",
    enable_prometheus=True,
    prometheus_port=9090,
)

tracer = stack["tracer"]
metrics = stack["metrics"]
observer = stack["observer"]

# Use observer for automatic metrics
with observer.observe("generate_users", table="users") as ctx:
    data = generator.generate(10000)
    ctx.set_rows(len(data))

# Metrics automatically recorded!
```

## Tracing

Track generation operations with distributed tracing:

```python
from genesis.observability import GenesisTracer

tracer = GenesisTracer(
    service_name="genesis",
    enable_otel=True,  # Enable OpenTelemetry
    otel_endpoint="http://jaeger:4317",
)

# Create spans
with tracer.span("generate_dataset", table="customers") as span:
    span.attributes["method"] = "ctgan"
    
    # Nested operations
    with tracer.span("fit_model"):
        generator.fit(data)
    
    with tracer.span("generate_rows"):
        synthetic = generator.generate(10000)
        span.attributes["rows_generated"] = len(synthetic)
    
    # Add events
    tracer.add_event("validation_complete", {"passed": True})
```

### Console Tracing (Development)

```python
# Default: logs to console
tracer = GenesisTracer(service_name="genesis")

with tracer.span("test_op"):
    pass
# Output: [TRACE] test_op trace_id=abc123 span_id=def456 duration=1.23ms status=OK
```

### OpenTelemetry Integration

```python
from genesis.observability import OpenTelemetryBackend

backend = OpenTelemetryBackend(
    service_name="genesis",
    endpoint="http://otel-collector:4317",
)

tracer = GenesisTracer(backend=backend)
```

## Metrics Collection

Record generation metrics for monitoring:

```python
from genesis.observability import MetricsCollector, InMemoryMetricsBackend

collector = MetricsCollector(prefix="genesis")

# Record generation
collector.record_generation(
    table="users",
    rows=10000,
    duration_seconds=5.2,
    generator_type="ctgan",
)

# Record quality scores
collector.record_quality_score(
    table="users",
    metric="statistical_similarity",
    score=0.95,
)

# Record SLA checks
collector.record_sla_check(
    table="users",
    passed=True,
    metric="uniqueness",
)

# Record errors
collector.record_error(
    error_type="GenerationError",
    table="users",
)
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rows_generated_total` | Counter | Total rows generated |
| `generation_duration_seconds` | Histogram | Generation time |
| `generation_throughput` | Gauge | Rows per second |
| `quality_score` | Gauge | Quality metric value |
| `sla_checks_total` | Counter | SLA check results |
| `errors_total` | Counter | Error counts |
| `memory_usage_bytes` | Gauge | Memory consumption |

## Prometheus Integration

Expose metrics for Prometheus scraping:

```python
from genesis.observability import PrometheusBackend, MetricsCollector

backend = PrometheusBackend(port=9090)
backend.start_server()

collector = MetricsCollector(backend=backend)

# Record metrics
collector.record_generation("users", 10000, 5.0)

# Metrics available at http://localhost:9090/metrics
```

### Prometheus Output Format

```
# HELP genesis_rows_generated_total Total rows generated
# TYPE genesis_rows_generated_total counter
genesis_rows_generated_total{table="users",generator="ctgan"} 10000

# HELP genesis_generation_duration_seconds Generation duration
# TYPE genesis_generation_duration_seconds histogram
genesis_generation_duration_seconds_bucket{table="users",le="1"} 0
genesis_generation_duration_seconds_bucket{table="users",le="5"} 1
genesis_generation_duration_seconds_sum{table="users"} 5.0
genesis_generation_duration_seconds_count{table="users"} 1
```

## StatsD Integration

Send metrics to StatsD/DataDog:

```python
from genesis.observability import StatsDBackend, MetricsCollector

backend = StatsDBackend(
    host="localhost",
    port=8125,
    prefix="genesis",
)

collector = MetricsCollector(backend=backend)
collector.record_generation("users", 10000, 5.0)
```

## Generation Observer

Combined tracing and metrics in one API:

```python
from genesis.observability import GenerationObserver

observer = GenerationObserver()

# Automatic tracing and metrics
with observer.observe("generate", table="users", generator_type="ctgan") as ctx:
    data = generator.generate(10000)
    ctx.set_rows(len(data))

# Automatically records:
# - Span with duration
# - rows_generated_total metric
# - generation_duration_seconds metric  
# - generation_throughput metric
# - Error metrics (if exception occurs)
```

## Dashboard Export

Export metrics for dashboards:

```python
from genesis.observability import DashboardExporter, InMemoryMetricsBackend

backend = InMemoryMetricsBackend()
# ... record metrics ...

exporter = DashboardExporter(backend)

# Grafana JSON format
grafana_data = exporter.to_grafana_json()

# CloudWatch format
cloudwatch_data = exporter.to_cloudwatch_format()
```

### Grafana Dashboard

```json
{
  "metrics": [
    {
      "target": "genesis_rows_generated_total_table=users",
      "datapoints": [[10000, "2026-01-15T10:30:00Z"]]
    }
  ]
}
```

## Complete Production Setup

```python
import atexit
from genesis import SyntheticGenerator
from genesis.observability import (
    create_observability_stack,
    PrometheusBackend,
)

# Initialize observability
stack = create_observability_stack(
    service_name="genesis-production",
    enable_otel=True,
    enable_prometheus=True,
    prometheus_port=9090,
)

tracer = stack["tracer"]
metrics = stack["metrics"]
observer = stack["observer"]

def generate_synthetic_data(table_name: str, n_rows: int):
    """Generate synthetic data with full observability."""
    
    with observer.observe(
        f"generate_{table_name}",
        table=table_name,
        generator_type="ctgan",
    ) as ctx:
        # Load and fit
        with tracer.span("load_data"):
            data = load_training_data(table_name)
        
        with tracer.span("fit_generator"):
            generator = SyntheticGenerator(method="ctgan")
            generator.fit(data)
        
        # Generate
        with tracer.span("generate"):
            synthetic = generator.generate(n_rows)
            ctx.set_rows(len(synthetic))
        
        # Validate
        with tracer.span("validate"):
            report = generator.quality_report()
            
            for metric_name, score in report.scores.items():
                metrics.record_quality_score(
                    table=table_name,
                    metric=metric_name,
                    score=score,
                )
        
        return synthetic

# Usage
try:
    data = generate_synthetic_data("customers", 100000)
except Exception as e:
    metrics.record_error(type(e).__name__, "customers")
    raise
```

## Alerting Integration

Set up alerts based on metrics:

```python
# Example: Alert on low quality scores
def check_and_alert(table: str, synthetic: pd.DataFrame):
    evaluator = QualityEvaluator()
    scores = evaluator.evaluate(original, synthetic)
    
    for metric, score in scores.items():
        metrics.record_quality_score(table, metric, score)
        
        if score < 0.8:
            metrics.record_error(
                f"quality_threshold_violation_{metric}",
                table,
            )
            # Trigger alert via your alerting system
            send_alert(f"Quality score {metric} below threshold: {score}")
```

## Configuration Reference

### GenesisTracer

| Parameter | Type | Description |
|-----------|------|-------------|
| `service_name` | str | Service name for tracing |
| `backend` | TracingBackend | Custom tracing backend |
| `enable_otel` | bool | Enable OpenTelemetry |
| `otel_endpoint` | str | OTLP endpoint URL |

### MetricsCollector

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | MetricsBackend | Metrics backend |
| `prefix` | str | Metric name prefix |

### PrometheusBackend

| Parameter | Type | Description |
|-----------|------|-------------|
| `port` | int | HTTP port for /metrics endpoint |

### create_observability_stack

| Parameter | Type | Description |
|-----------|------|-------------|
| `service_name` | str | Service name |
| `enable_otel` | bool | Enable OpenTelemetry |
| `enable_prometheus` | bool | Enable Prometheus server |
| `prometheus_port` | int | Prometheus port |
| `statsd_host` | str | StatsD host (enables StatsD) |
