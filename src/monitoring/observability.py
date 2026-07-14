from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader


otlp_exporter = OTLPSpanExporter(
    endpoint="http://jaeger:4317",
    insecure=True,
)

trace.set_tracer_provider(
    TracerProvider()
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(
        otlp_exporter
    )
)

prometheus_reader = PrometheusMetricReader()

metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[
            prometheus_reader
        ]
    )
)


meter = metrics.get_meter("rag_system")
documents_uploaded = meter.create_counter(
    "documents_uploaded",
    description="Total documents uploaded"
)


chunks_created = meter.create_counter(
    "document_chunks_created",
    description="Total chunks created"
)


rag_queries_total = meter.create_counter(
    "rag_queries",
    description="Total RAG queries"
)


retrieved_documents = meter.create_counter(
    "retrieved_documents",
    description="Total retrieved documents"
)


llm_api_calls = meter.create_counter(
    "llm_api_calls",
    description="Total LLM API calls"
)


upload_latency = meter.create_histogram(
    "document_upload_latency_seconds",
    description="Document upload latency",
    unit="s"
)


retrieval_latency = meter.create_histogram(
    "retrieval_latency_seconds",
    description="Retrieval latency",
    unit="s"
)


llm_latency = meter.create_histogram(
    "llm_latency_seconds",
    description="LLM latency",
    unit="s"
)


total_latency = meter.create_histogram(
    "rag_total_latency_seconds",
    description="Total RAG latency",
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


vector_db_size = meter.create_up_down_counter(
    "vector_db_size_documents",
    description="Vector DB document count"
)


llm_token_usage = meter.create_counter(
    "llm_tokens_total",
    description="Total LLM tokens"
)

rag_errors = meter.create_counter(
    "rag_errors_total",
    description="Failed RAG queries"
)

tracer = trace.get_tracer("rag_system")
