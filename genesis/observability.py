"""Synthetic Data Observability for Genesis.

This module provides comprehensive observability for synthetic data
generation, including:

- OpenTelemetry integration for distributed tracing
- Metrics emission for generation performance
- Quality metrics tracking
- Dashboard-ready metric formats
- Alerting hooks

Example:
    >>> from genesis.observability import GenesisTracer, MetricsCollector
    >>>
    >>> tracer = GenesisTracer(service_name="genesis-prod")
    >>> metrics = MetricsCollector()
    >>>
    >>> with tracer.span("generate_users"):
    ...     data = generator.generate(1000)
    ...     metrics.record_generation(
    ...         table="users",
    ...         rows=1000,
    ...         duration=elapsed
    ...     )
"""

import atexit
import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from genesis.core.exceptions import GenesisError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """A single metric measurement."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class SpanContext:
    """Context for a tracing span."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


@dataclass
class Span:
    """A tracing span representing a unit of work."""

    name: str
    context: SpanContext
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    status_message: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


class TracingBackend:
    """Base class for tracing backends."""

    def export_span(self, span: Span) -> None:
        """Export a completed span."""
        pass

    def flush(self) -> None:
        """Flush any buffered spans."""
        pass


class ConsoleTracingBackend(TracingBackend):
    """Tracing backend that outputs to console/logs."""

    def export_span(self, span: Span) -> None:
        """Log span to console."""
        logger.info(
            f"[TRACE] {span.name} "
            f"trace_id={span.context.trace_id[:8]} "
            f"span_id={span.context.span_id[:8]} "
            f"duration={span.duration_ms:.2f}ms "
            f"status={span.status}"
        )


class OpenTelemetryBackend(TracingBackend):
    """OpenTelemetry-compatible tracing backend.

    Requires: opentelemetry-api, opentelemetry-sdk
    """

    def __init__(
        self,
        service_name: str,
        endpoint: Optional[str] = None,
    ) -> None:
        self.service_name = service_name
        self.endpoint = endpoint
        self._tracer = None
        self._setup_otel()

    def _setup_otel(self) -> None:
        """Set up OpenTelemetry SDK."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()

            if self.endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                exporter = OTLPSpanExporter(endpoint=self.endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(self.service_name)

        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )

    def export_span(self, span: Span) -> None:
        """Export span via OpenTelemetry."""
        if not self._tracer:
            return

        from opentelemetry import trace

        with self._tracer.start_as_current_span(span.name) as otel_span:
            for key, value in span.attributes.items():
                otel_span.set_attribute(key, str(value))

            for event in span.events:
                otel_span.add_event(
                    event.get("name", "event"),
                    attributes=event.get("attributes", {}),
                )

            if span.status != "OK":
                otel_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, span.status_message)
                )


class GenesisTracer:
    """Tracer for Genesis operations.

    Example:
        >>> tracer = GenesisTracer(service_name="genesis")
        >>>
        >>> with tracer.span("generate_table", table="users") as span:
        ...     data = generator.generate(1000)
        ...     span.set_attribute("rows_generated", len(data))
    """

    def __init__(
        self,
        service_name: str = "genesis",
        backend: Optional[TracingBackend] = None,
        enable_otel: bool = False,
        otel_endpoint: Optional[str] = None,
    ) -> None:
        """Initialize tracer.

        Args:
            service_name: Name of the service for tracing
            backend: Custom tracing backend
            enable_otel: Enable OpenTelemetry integration
            otel_endpoint: OTLP endpoint for OpenTelemetry
        """
        self.service_name = service_name

        if backend:
            self.backend = backend
        elif enable_otel:
            self.backend = OpenTelemetryBackend(service_name, otel_endpoint)
        else:
            self.backend = ConsoleTracingBackend()

        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []

    def _generate_id(self) -> str:
        """Generate a random ID for spans/traces."""
        import secrets
        return secrets.token_hex(16)

    @contextmanager
    def span(
        self,
        name: str,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Create a new span context.

        Args:
            name: Span name
            **attributes: Initial attributes

        Yields:
            The created span
        """
        parent_span = self._current_span

        context = SpanContext(
            trace_id=parent_span.context.trace_id if parent_span else self._generate_id(),
            span_id=self._generate_id(),
            parent_span_id=parent_span.context.span_id if parent_span else None,
        )

        span = Span(
            name=name,
            context=context,
            start_time=time.time(),
            attributes=dict(attributes),
        )

        self._span_stack.append(span)
        self._current_span = span

        try:
            yield span
        except Exception as e:
            span.status = "ERROR"
            span.status_message = str(e)
            raise
        finally:
            span.end_time = time.time()
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None
            self.backend.export_span(span)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the current span."""
        if self._current_span:
            self._current_span.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            })


class MetricsBackend:
    """Base class for metrics backends."""

    def emit(self, metric: MetricPoint) -> None:
        """Emit a metric point."""
        pass

    def flush(self) -> None:
        """Flush any buffered metrics."""
        pass


class InMemoryMetricsBackend(MetricsBackend):
    """In-memory metrics storage for testing/development."""

    def __init__(self) -> None:
        self._metrics: List[MetricPoint] = []
        self._lock = threading.Lock()

    def emit(self, metric: MetricPoint) -> None:
        with self._lock:
            self._metrics.append(metric)

    def get_metrics(self) -> List[MetricPoint]:
        """Get all stored metrics."""
        return list(self._metrics)

    def clear(self) -> None:
        """Clear stored metrics."""
        with self._lock:
            self._metrics.clear()


class PrometheusBackend(MetricsBackend):
    """Prometheus-compatible metrics backend.

    Exposes metrics in Prometheus format via HTTP.
    """

    def __init__(self, port: int = 9090) -> None:
        self.port = port
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._server = None

    def emit(self, metric: MetricPoint) -> None:
        """Store metric for Prometheus scraping."""
        key = self._metric_key(metric)

        with self._lock:
            if metric.metric_type == MetricType.COUNTER:
                self._metrics[key] = self._metrics.get(key, 0) + metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self._metrics[key] = metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                if key not in self._metrics:
                    self._metrics[key] = {
                        "count": 0,
                        "sum": 0,
                        "buckets": defaultdict(int),
                    }
                hist = self._metrics[key]
                hist["count"] += 1
                hist["sum"] += metric.value

                # Default buckets
                for bucket in [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]:
                    if metric.value <= bucket:
                        hist["buckets"][bucket] += 1

    def _metric_key(self, metric: MetricPoint) -> str:
        """Generate unique key for metric."""
        labels = ",".join(f'{k}="{v}"' for k, v in sorted(metric.labels.items()))
        return f"{metric.name}{{{labels}}}" if labels else metric.name

    def get_prometheus_output(self) -> str:
        """Generate Prometheus-format output."""
        lines = []

        with self._lock:
            for key, value in self._metrics.items():
                if isinstance(value, dict):
                    # Histogram
                    name = key.split("{")[0]
                    labels = key[len(name):] if "{" in key else ""

                    for bucket, count in sorted(value["buckets"].items()):
                        bucket_labels = labels.rstrip("}") + f',le="{bucket}"}}'
                        lines.append(f"{name}_bucket{bucket_labels} {count}")

                    lines.append(f'{name}_sum{labels} {value["sum"]}')
                    lines.append(f'{name}_count{labels} {value["count"]}')
                else:
                    lines.append(f"{key} {value}")

        return "\n".join(lines)

    def start_server(self) -> None:
        """Start HTTP server for Prometheus scraping."""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        backend = self

        class PrometheusHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/metrics":
                    output = backend.get_prometheus_output()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(output.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress logging

        self._server = HTTPServer(("0.0.0.0", self.port), PrometheusHandler)

        def serve() -> None:
            self._server.serve_forever()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        logger.info(f"Prometheus metrics server started on port {self.port}")

    def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()


class StatsDBackend(MetricsBackend):
    """StatsD-compatible metrics backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "genesis",
    ) -> None:
        self.host = host
        self.port = port
        self.prefix = prefix
        self._socket = None

    def _get_socket(self) -> Any:
        """Get or create UDP socket."""
        if self._socket is None:
            import socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self._socket

    def emit(self, metric: MetricPoint) -> None:
        """Send metric to StatsD."""
        name = f"{self.prefix}.{metric.name}"

        type_char = {
            MetricType.COUNTER: "c",
            MetricType.GAUGE: "g",
            MetricType.HISTOGRAM: "ms",
            MetricType.SUMMARY: "ms",
        }.get(metric.metric_type, "g")

        message = f"{name}:{metric.value}|{type_char}"

        if metric.labels:
            tags = ",".join(f"{k}:{v}" for k, v in metric.labels.items())
            message += f"|#{tags}"

        try:
            sock = self._get_socket()
            sock.sendto(message.encode(), (self.host, self.port))
        except Exception as e:
            logger.warning(f"Failed to send metric to StatsD: {e}")

    def flush(self) -> None:
        """Close socket."""
        if self._socket:
            self._socket.close()
            self._socket = None


class MetricsCollector:
    """Collects and emits metrics for Genesis operations.

    Example:
        >>> collector = MetricsCollector()
        >>>
        >>> collector.record_generation(
        ...     table="users",
        ...     rows=10000,
        ...     duration_seconds=5.2
        ... )
        >>>
        >>> collector.record_quality_score(
        ...     table="users",
        ...     metric="statistical_similarity",
        ...     score=0.95
        ... )
    """

    def __init__(
        self,
        backend: Optional[MetricsBackend] = None,
        prefix: str = "genesis",
    ) -> None:
        """Initialize metrics collector.

        Args:
            backend: Metrics backend to use
            prefix: Prefix for all metric names
        """
        self.backend = backend or InMemoryMetricsBackend()
        self.prefix = prefix

    def _emit(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
        unit: str = "",
    ) -> None:
        """Emit a metric."""
        metric = MetricPoint(
            name=f"{self.prefix}_{name}" if self.prefix else name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            unit=unit,
        )
        self.backend.emit(metric)

    def record_generation(
        self,
        table: str,
        rows: int,
        duration_seconds: float,
        generator_type: str = "unknown",
    ) -> None:
        """Record a generation operation.

        Args:
            table: Table name
            rows: Number of rows generated
            duration_seconds: Generation duration
            generator_type: Type of generator used
        """
        labels = {"table": table, "generator": generator_type}

        self._emit(
            "rows_generated_total",
            rows,
            MetricType.COUNTER,
            labels,
        )

        self._emit(
            "generation_duration_seconds",
            duration_seconds,
            MetricType.HISTOGRAM,
            labels,
            unit="seconds",
        )

        rows_per_second = rows / duration_seconds if duration_seconds > 0 else 0
        self._emit(
            "generation_throughput",
            rows_per_second,
            MetricType.GAUGE,
            labels,
            unit="rows/s",
        )

    def record_quality_score(
        self,
        table: str,
        metric: str,
        score: float,
    ) -> None:
        """Record a quality metric score.

        Args:
            table: Table name
            metric: Quality metric name
            score: Score value (0-1)
        """
        self._emit(
            "quality_score",
            score,
            MetricType.GAUGE,
            {"table": table, "metric": metric},
        )

    def record_sla_check(
        self,
        table: str,
        passed: bool,
        metric: str,
    ) -> None:
        """Record an SLA check result.

        Args:
            table: Table name
            passed: Whether check passed
            metric: Metric that was checked
        """
        self._emit(
            "sla_checks_total",
            1,
            MetricType.COUNTER,
            {"table": table, "metric": metric, "result": "pass" if passed else "fail"},
        )

    def record_error(
        self,
        error_type: str,
        table: Optional[str] = None,
    ) -> None:
        """Record an error.

        Args:
            error_type: Type of error
            table: Table name if applicable
        """
        labels = {"type": error_type}
        if table:
            labels["table"] = table

        self._emit(
            "errors_total",
            1,
            MetricType.COUNTER,
            labels,
        )

    def record_memory_usage(
        self,
        bytes_used: int,
        table: Optional[str] = None,
    ) -> None:
        """Record memory usage.

        Args:
            bytes_used: Memory used in bytes
            table: Table name if applicable
        """
        labels = {"table": table} if table else {}

        self._emit(
            "memory_usage_bytes",
            bytes_used,
            MetricType.GAUGE,
            labels,
            unit="bytes",
        )


class GenerationObserver:
    """Observer that combines tracing and metrics.

    Example:
        >>> observer = GenerationObserver()
        >>>
        >>> with observer.observe("generate_users", table="users"):
        ...     data = generator.generate(1000)
        ...
        >>> # Automatically records duration, throughput, etc.
    """

    def __init__(
        self,
        tracer: Optional[GenesisTracer] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> None:
        self.tracer = tracer or GenesisTracer()
        self.metrics = metrics or MetricsCollector()
        self._rows_generated = 0

    @contextmanager
    def observe(
        self,
        name: str,
        table: str = "unknown",
        generator_type: str = "unknown",
    ) -> Iterator["ObservationContext"]:
        """Observe a generation operation.

        Args:
            name: Operation name
            table: Table being generated
            generator_type: Type of generator

        Yields:
            Observation context for setting additional info
        """
        context = ObservationContext(table=table, generator_type=generator_type)
        start_time = time.time()

        with self.tracer.span(name, table=table, generator=generator_type):
            try:
                yield context
            except Exception as e:
                self.metrics.record_error(type(e).__name__, table)
                raise
            finally:
                duration = time.time() - start_time

                if context.rows_generated > 0:
                    self.metrics.record_generation(
                        table=table,
                        rows=context.rows_generated,
                        duration_seconds=duration,
                        generator_type=generator_type,
                    )


@dataclass
class ObservationContext:
    """Context for generation observation."""

    table: str
    generator_type: str
    rows_generated: int = 0

    def set_rows(self, rows: int) -> None:
        """Set number of rows generated."""
        self.rows_generated = rows


class DashboardExporter:
    """Exports metrics in formats suitable for dashboards.

    Supports:
    - Grafana JSON
    - DataDog metrics
    - CloudWatch format
    """

    def __init__(self, metrics_backend: MetricsBackend) -> None:
        self.backend = metrics_backend

    def to_grafana_json(self) -> Dict[str, Any]:
        """Export metrics in Grafana-compatible JSON format."""
        if not isinstance(self.backend, InMemoryMetricsBackend):
            return {}

        metrics = self.backend.get_metrics()

        data_points: Dict[str, List[Any]] = defaultdict(list)
        for m in metrics:
            key = f"{m.name}_{','.join(f'{k}={v}' for k,v in m.labels.items())}"
            data_points[key].append({
                "timestamp": m.timestamp.isoformat(),
                "value": m.value,
            })

        return {
            "metrics": [
                {
                    "target": name,
                    "datapoints": [[p["value"], p["timestamp"]] for p in points]
                }
                for name, points in data_points.items()
            ]
        }

    def to_cloudwatch_format(self) -> List[Dict[str, Any]]:
        """Export metrics in CloudWatch format."""
        if not isinstance(self.backend, InMemoryMetricsBackend):
            return []

        metrics = self.backend.get_metrics()

        return [
            {
                "MetricName": m.name,
                "Dimensions": [
                    {"Name": k, "Value": v} for k, v in m.labels.items()
                ],
                "Timestamp": m.timestamp.isoformat(),
                "Value": m.value,
                "Unit": m.unit or "None",
            }
            for m in metrics
        ]


def create_observability_stack(
    service_name: str = "genesis",
    enable_otel: bool = False,
    enable_prometheus: bool = False,
    prometheus_port: int = 9090,
    statsd_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a complete observability stack.

    Args:
        service_name: Name of the service
        enable_otel: Enable OpenTelemetry
        enable_prometheus: Enable Prometheus metrics server
        prometheus_port: Port for Prometheus server
        statsd_host: Host for StatsD (enables StatsD backend)

    Returns:
        Dictionary with tracer, metrics, and observer
    """
    tracer = GenesisTracer(
        service_name=service_name,
        enable_otel=enable_otel,
    )

    if enable_prometheus:
        metrics_backend = PrometheusBackend(port=prometheus_port)
        metrics_backend.start_server()
        atexit.register(metrics_backend.stop_server)
    elif statsd_host:
        metrics_backend = StatsDBackend(host=statsd_host)
    else:
        metrics_backend = InMemoryMetricsBackend()

    metrics = MetricsCollector(backend=metrics_backend)
    observer = GenerationObserver(tracer=tracer, metrics=metrics)

    return {
        "tracer": tracer,
        "metrics": metrics,
        "observer": observer,
        "backend": metrics_backend,
    }
