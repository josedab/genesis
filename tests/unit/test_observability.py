"""Tests for synthetic data observability."""

import time

import pytest

from genesis.observability import (
    ConsoleTracingBackend,
    DashboardExporter,
    GenerationObserver,
    GenesisTracer,
    InMemoryMetricsBackend,
    MetricPoint,
    MetricsCollector,
    MetricType,
    ObservationContext,
    PrometheusBackend,
    Span,
    SpanContext,
    StatsDBackend,
    create_observability_stack,
)


class TestMetricPoint:
    """Tests for MetricPoint dataclass."""

    def test_default_values(self) -> None:
        """Test default metric point values."""
        point = MetricPoint(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.COUNTER,
        )

        assert point.name == "test_metric"
        assert point.value == 42.0
        assert point.metric_type == MetricType.COUNTER
        assert point.labels == {}
        assert point.unit == ""

    def test_with_labels(self) -> None:
        """Test metric point with labels."""
        point = MetricPoint(
            name="requests",
            value=100,
            metric_type=MetricType.COUNTER,
            labels={"method": "GET", "status": "200"},
        )

        assert point.labels["method"] == "GET"
        assert point.labels["status"] == "200"


class TestSpan:
    """Tests for Span dataclass."""

    def test_duration_calculation(self) -> None:
        """Test span duration calculation."""
        span = Span(
            name="test_span",
            context=SpanContext(trace_id="abc", span_id="123"),
            start_time=1000.0,
            end_time=1002.5,
        )

        assert span.duration_ms == 2500.0

    def test_duration_zero_when_not_ended(self) -> None:
        """Test duration is zero when span not ended."""
        span = Span(
            name="test_span",
            context=SpanContext(trace_id="abc", span_id="123"),
            start_time=1000.0,
        )

        assert span.duration_ms == 0


class TestInMemoryMetricsBackend:
    """Tests for InMemoryMetricsBackend."""

    def test_emit_and_get(self) -> None:
        """Test emitting and retrieving metrics."""
        backend = InMemoryMetricsBackend()

        backend.emit(MetricPoint("test", 1.0, MetricType.COUNTER))
        backend.emit(MetricPoint("test", 2.0, MetricType.COUNTER))

        metrics = backend.get_metrics()
        assert len(metrics) == 2

    def test_clear(self) -> None:
        """Test clearing metrics."""
        backend = InMemoryMetricsBackend()

        backend.emit(MetricPoint("test", 1.0, MetricType.COUNTER))
        backend.clear()

        assert len(backend.get_metrics()) == 0


class TestPrometheusBackend:
    """Tests for PrometheusBackend."""

    def test_emit_counter(self) -> None:
        """Test emitting counter metric."""
        backend = PrometheusBackend()

        backend.emit(MetricPoint("requests", 1.0, MetricType.COUNTER))
        backend.emit(MetricPoint("requests", 1.0, MetricType.COUNTER))

        output = backend.get_prometheus_output()
        assert "requests 2" in output

    def test_emit_gauge(self) -> None:
        """Test emitting gauge metric."""
        backend = PrometheusBackend()

        backend.emit(MetricPoint("temperature", 25.5, MetricType.GAUGE))

        output = backend.get_prometheus_output()
        assert "temperature 25.5" in output

    def test_emit_with_labels(self) -> None:
        """Test emitting metric with labels."""
        backend = PrometheusBackend()

        backend.emit(MetricPoint(
            "requests",
            1.0,
            MetricType.COUNTER,
            labels={"method": "GET"},
        ))

        output = backend.get_prometheus_output()
        assert 'method="GET"' in output


class TestGenesisTracer:
    """Tests for GenesisTracer."""

    def test_span_context_manager(self) -> None:
        """Test span as context manager."""
        tracer = GenesisTracer(service_name="test")

        with tracer.span("test_operation") as span:
            span.attributes["key"] = "value"
            time.sleep(0.01)

        assert span.end_time is not None
        assert span.duration_ms > 0
        assert span.status == "OK"

    def test_nested_spans(self) -> None:
        """Test nested span handling."""
        tracer = GenesisTracer(service_name="test")

        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                assert child.context.parent_span_id == parent.context.span_id

    def test_span_error_handling(self) -> None:
        """Test span error status on exception."""
        tracer = GenesisTracer(service_name="test")

        with pytest.raises(ValueError):
            with tracer.span("failing_op") as span:
                raise ValueError("Test error")

        assert span.status == "ERROR"
        assert "Test error" in span.status_message

    def test_add_event(self) -> None:
        """Test adding events to span."""
        tracer = GenesisTracer(service_name="test")

        with tracer.span("test_op") as span:
            tracer.add_event("checkpoint", {"step": 1})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create metrics collector fixture."""
        return MetricsCollector(backend=InMemoryMetricsBackend())

    def test_record_generation(self, collector: MetricsCollector) -> None:
        """Test recording generation metrics."""
        collector.record_generation(
            table="users",
            rows=10000,
            duration_seconds=5.0,
            generator_type="ctgan",
        )

        metrics = collector.backend.get_metrics()

        # Should have multiple metrics
        assert len(metrics) >= 3

        # Check specific metrics exist
        names = [m.name for m in metrics]
        assert any("rows_generated" in n for n in names)
        assert any("duration" in n for n in names)
        assert any("throughput" in n for n in names)

    def test_record_quality_score(self, collector: MetricsCollector) -> None:
        """Test recording quality score."""
        collector.record_quality_score(
            table="users",
            metric="statistical_similarity",
            score=0.95,
        )

        metrics = collector.backend.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].value == 0.95

    def test_record_sla_check(self, collector: MetricsCollector) -> None:
        """Test recording SLA check result."""
        collector.record_sla_check(
            table="users",
            passed=True,
            metric="uniqueness",
        )

        metrics = collector.backend.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].labels["result"] == "pass"

    def test_record_error(self, collector: MetricsCollector) -> None:
        """Test recording error."""
        collector.record_error(
            error_type="GenerationError",
            table="users",
        )

        metrics = collector.backend.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].labels["type"] == "GenerationError"

    def test_prefix_applied(self) -> None:
        """Test that prefix is applied to metrics."""
        collector = MetricsCollector(
            backend=InMemoryMetricsBackend(),
            prefix="myapp",
        )

        collector.record_error("test")

        metrics = collector.backend.get_metrics()
        assert metrics[0].name.startswith("myapp_")


class TestGenerationObserver:
    """Tests for GenerationObserver."""

    def test_observe_records_metrics(self) -> None:
        """Test that observe records metrics."""
        backend = InMemoryMetricsBackend()
        observer = GenerationObserver(
            tracer=GenesisTracer(),
            metrics=MetricsCollector(backend=backend),
        )

        with observer.observe("generate", table="users") as ctx:
            ctx.set_rows(1000)

        metrics = backend.get_metrics()
        assert len(metrics) > 0

    def test_observe_handles_error(self) -> None:
        """Test that observe handles errors."""
        backend = InMemoryMetricsBackend()
        observer = GenerationObserver(
            tracer=GenesisTracer(),
            metrics=MetricsCollector(backend=backend),
        )

        with pytest.raises(ValueError):
            with observer.observe("generate", table="users"):
                raise ValueError("Test error")

        # Should record error metric
        metrics = backend.get_metrics()
        error_metrics = [m for m in metrics if "error" in m.name.lower()]
        assert len(error_metrics) > 0


class TestObservationContext:
    """Tests for ObservationContext."""

    def test_set_rows(self) -> None:
        """Test setting rows generated."""
        ctx = ObservationContext(table="users", generator_type="ctgan")
        ctx.set_rows(5000)

        assert ctx.rows_generated == 5000


class TestDashboardExporter:
    """Tests for DashboardExporter."""

    def test_to_grafana_json(self) -> None:
        """Test Grafana JSON export."""
        backend = InMemoryMetricsBackend()
        backend.emit(MetricPoint("test_metric", 42.0, MetricType.GAUGE))

        exporter = DashboardExporter(backend)
        output = exporter.to_grafana_json()

        assert "metrics" in output
        assert len(output["metrics"]) > 0

    def test_to_cloudwatch_format(self) -> None:
        """Test CloudWatch format export."""
        backend = InMemoryMetricsBackend()
        backend.emit(MetricPoint(
            "test_metric",
            42.0,
            MetricType.GAUGE,
            labels={"env": "prod"},
        ))

        exporter = DashboardExporter(backend)
        output = exporter.to_cloudwatch_format()

        assert len(output) == 1
        assert output[0]["MetricName"] == "test_metric"
        assert output[0]["Value"] == 42.0


class TestCreateObservabilityStack:
    """Tests for create_observability_stack."""

    def test_creates_all_components(self) -> None:
        """Test that all components are created."""
        stack = create_observability_stack(service_name="test")

        assert "tracer" in stack
        assert "metrics" in stack
        assert "observer" in stack
        assert "backend" in stack

    def test_uses_in_memory_by_default(self) -> None:
        """Test that in-memory backend is used by default."""
        stack = create_observability_stack()

        assert isinstance(stack["backend"], InMemoryMetricsBackend)

    def test_prometheus_backend_when_enabled(self) -> None:
        """Test Prometheus backend when enabled."""
        stack = create_observability_stack(enable_prometheus=True)

        assert isinstance(stack["backend"], PrometheusBackend)
        stack["backend"].stop_server()
