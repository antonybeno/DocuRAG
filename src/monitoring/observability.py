"""
Observability and Monitoring Module

This module sets up OpenTelemetry for distributed tracing (Jaeger) and Prometheus metrics collection.
It provides:
    - Distributed tracing with Jaeger (trace aggregation and visualization)
    - Metrics collection with Prometheus (counters, histograms, gauges)
    - Centralized tracer and meter instances for application-wide use

Configuration:
    - Jaeger endpoint: http://jaeger:4317
    - Prometheus scrape interval: 5s (configured in src/prometheus.yml)
"""

import logging
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)


# ==================== TRACING SETUP ====================

def _setup_jaeger_tracing(jaeger_endpoint: str = "http://jaeger:4317") -> Optional[TracerProvider]:
    """
    Initialize Jaeger tracing provider.

    Args:
        jaeger_endpoint: Jaeger collector endpoint (default: Docker Compose)

    Returns:
        TracerProvider instance if successful, None if connection fails

    Raises:
        None - only logs
    """
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=jaeger_endpoint,
            insecure=True,
        )
        tracer_provider = TracerProvider(
            resource=Resource.create({
                "service.name": "rag-system",
                "service.version": "1.0.0"
            })
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(tracer_provider)

        logger.info("Jaeger tracing initialized: %s", jaeger_endpoint)
        return tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize Jaeger: {e}")
        logger.warning("Falling back to no-op tracer")
        trace.set_tracer_provider(TracerProvider())
        return None


# ==================== METRICS SETUP ====================

def _setup_prometheus_metrics() -> MeterProvider:
    """
    Initialize Prometheus metrics provider.

    Returns:
        MeterProvider instance for metrics collection

    Raises:
        None - only logs
    """
    try:
        prometheus_reader = PrometheusMetricReader()
        meter_provider = MeterProvider(
            metric_readers=[prometheus_reader],
            resource=Resource.create({
                "service.name": "rag-system"
            })
        )

        metrics.set_meter_provider(meter_provider)

        logger.info("Prometheus metrics initialized.")
        return meter_provider

    except Exception:
        logger.exception("Failed to initialize Prometheus.")
        raise


# ==================== INITIALIZATION ====================

# Setup tracing and metrics
_setup_jaeger_tracing()
_setup_prometheus_metrics()

# Get tracer and meter instances
tracer = trace.get_tracer("rag_system")
meter = metrics.get_meter("rag_system")

# ==================== COUNTERS ====================

documents_uploaded = meter.create_counter(
    name="documents_uploaded_total",
    description="Total number of documents uploaded to the system",
    unit="1"
)

chunks_created = meter.create_counter(
    name="document_chunks_created_total",
    description="Total number of chunks created from documents",
    unit="1"
)

rag_queries_total = meter.create_counter(
    name="rag_queries_total",
    description="Total number of RAG queries processed",
    unit="1"
)

retrieved_documents = meter.create_counter(
    name="retrieved_documents_total",
    description="Total number of documents retrieved across all queries",
    unit="1"
)

llm_api_calls = meter.create_counter(
    name="llm_api_calls_total",
    description="Total number of LLM API calls made",
    unit="1"
)

rag_errors = meter.create_counter(
    name="rag_errors_total",
    description="Total number of failed RAG queries",
    unit="1"
)

llm_token_usage = meter.create_counter(
    name="llm_tokens_total",
    description="Total number of tokens consumed by LLM",
    unit="1"
)

# ==================== HISTOGRAMS ====================

upload_latency = meter.create_histogram(
    name="document_upload_latency_seconds",
    description="Distribution of document upload latency",
    unit="s"
)

retrieval_latency = meter.create_histogram(
    name="retrieval_latency_seconds",
    description="Distribution of document retrieval latency (vector search + deduplication)",
    unit="s"
)

llm_latency = meter.create_histogram(
    name="llm_latency_seconds",
    description="Distribution of LLM response latency",
    unit="s"
)

total_latency = meter.create_histogram(
    name="rag_total_latency_seconds",
    description="Distribution of end-to-end RAG pipeline latency",
    unit="s"
)

retrieval_accuracy = meter.create_histogram(
    "retrieval_accuracy",
    description="Retrieval accuracy",
    unit="1"
)

hallucination_rate = meter.create_histogram(
    "hallucination_rate",
    description="Hallucination score",
    unit="1"
)

# ==================== GAUGES ====================

vector_db_size = meter.create_observable_gauge(
    name="vector_db_size_documents",
    description="Current number of documents in vector database",
    unit="1",
    callbacks=[]
)
