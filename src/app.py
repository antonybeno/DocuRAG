"""
FastAPI Application - Production RAG System

Main entry point for the Retrieval-Augmented Generation API.

Endpoints:
    POST   /query                     - RAG query endpoint
    POST   /documents/upload          - Single document upload
    POST   /documents/upload-batch    - Batch upload
    GET    /documents/list            - List all documents
    GET    /documents/{file_id}       - Get document info
    DELETE /documents/{file_id}       - Delete document
    GET    /documents/stats/vector-db - Vector store statistics
    GET    /health                    - Health check
    GET    /metrics                   - Prometheus metrics

Infrastructure:
    - OpenTelemetry for distributed tracing (Jaeger)
    - Prometheus for metrics collection
    - FastAPI automatic OpenAPI documentation
    - Structured logging with JSON support

Configuration:
    All settings via environment variables (see config/settings.py)
    - OLLAMA_MODEL: LLM model to use
    - OLLAMA_BASE_URL: Ollama server URL
    - EMBEDDING_MODEL: Embedding model name
    - LOG_LEVEL: Logging verbosity
    - FAISS_INDEX_PATH: Vector store location

Example .env:
    OLLAMA_MODEL=llama3:latest
    OLLAMA_BASE_URL=http://localhost:11434
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    LOG_LEVEL=INFO

Deployment:
    docker-compose up -d  # With monitoring stack
    python -m uvicorn src.app:app --host 0.0.0.0 --port 8001

Monitoring:
    - Traces: http://localhost:16686 (Jaeger)
    - Metrics: http://localhost:3000 (Grafana)
    - Health: http://localhost:8001/health
"""
import logging
import time
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import StatusCode, Status
from prometheus_client import generate_latest, REGISTRY
from pydantic import BaseModel
import uvicorn

from src.config.settings import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    LOG_LEVEL
)
from src.monitoring.observability import (
    tracer,
    rag_queries_total,
    rag_errors,
    retrieved_documents,
    total_latency,
    documents_uploaded,
    retrieval_accuracy,
    hallucination_rate
)
from src.rag.pipeline import RAGPipeline, PipelineError
from src.ingestion.MetadataStore import MetaDataStore
from src.ingestion.document_ingestion_service import DocumentIngestionService
from src.ingestion.file_manager import FileManager
from src.ingestion.document_parser import DocumentParser
from src.ingestion.chunker import Chunker
from src.rag.evaluation import RAGEvaluator
from src.rag.retrieval import HybridRetriever
from src.vectorstore.vector_store import VectorStore

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: list
    latency_ms: float
    retrieval_confidence: float
    hallucination_score: float


class UploadResponse(BaseModel):
    file_id: str
    status: str
    chunks_created: int
    file_size: int
    processing_time_seconds: float
    message: str


class BatchUploadResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    results: List[dict]


def _initialize_app() -> tuple:
    """
    Initialize RAG application components.

    Order matters:
    1. Embedding model (used by vector store and evaluator)
    2. Vector store (used by retriever and pipeline)
    3. Retriever (used by pipeline)
    4. Evaluator (used by pipeline)
    5. LLM (used by pipeline)
    6. Pipeline (uses all above)
    7. Ingestion service (uses vector store and chunker)

    Returns:
        Tuple of initialized components

    Raises:
        RuntimeError: If initialization fails
    """
    try:
        logger.info("Initializing RAG application...")

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        logger.info("Embedding model loaded")

        logger.info(f"Initializing vector store: {FAISS_INDEX_PATH}")
        vector_store = VectorStore(
            embedding_model,
            FAISS_INDEX_PATH
        )
        vector_store.load()
        logger.info("Vector store ready")

        logger.info("Initializing hybrid retriever")
        retriever = HybridRetriever(k=5)
        logger.info("Retriever initialized")

        logger.info("Initializing RAG evaluator")
        evaluator = RAGEvaluator(embedding_model)
        logger.info("Evaluator initialized")

        logger.info(f"Connecting to LLM: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0,
            base_url=OLLAMA_BASE_URL,
            timeout=30
        )
        logger.info("LLM connected")

        logger.info("Initializing RAG pipeline")
        rag_pipeline = RAGPipeline(
            vector_store,
            retriever,
            evaluator,
            llm
        )
        rag_pipeline.initialize()
        logger.info("RAG pipeline ready")

        logger.info("Initializing document ingestion service")
        file_manager = FileManager()
        parser = DocumentParser()
        chunker = Chunker()
        metadata = MetaDataStore()

        doc_ingestion_service = DocumentIngestionService(
            file_manager,
            parser,
            chunker,
            vector_store,
            metadata
        )
        logger.info("Ingestion service ready")
        logger.info("Application initialization complete")

        return (
            rag_pipeline,
            doc_ingestion_service,
            metadata,
            vector_store
        )

    except Exception as e:
        logger.exception("Failed to initialize application")
        raise RuntimeError("Application initialization failed") from e


try:
    rag_pipeline, ingestion_service, metadata_store, vector_store = _initialize_app()
except RuntimeError as e:
    logger.critical(f"Cannot start application: {str(e)}")
    raise

app = FastAPI(title="Production RAG API")

FastAPIInstrumentor.instrument_app(app)


@app.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...), user_id: str = "anonymous") -> UploadResponse:
    start_time = time.perf_counter()
    with tracer.start_as_current_span("document_upload") as span:
        try:
            if not file or not file.filename:
                raise ValueError("Filename required")
            logger.info("Document upload started: %s",file.filename)
            span.set_attribute("filename", file.filename)
            span.set_attribute("user_id", user_id)

            result = ingestion_service.process_document(
                file=file.file,
                filename=file.filename,
                user_id=user_id
            )

            processing_time = (time.perf_counter() - start_time)

            if result["status"] != "success":
                error_msg = result.get("error", "Unknown error")
                documents_uploaded.add(1, {"status": "error"})
                span.record_exception(Exception(error_msg))

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )

            documents_uploaded.add(1, {"status": "success"})
            span.set_attribute("status", "success")
            span.set_attribute("chunks_created", result["chunks_created"])
            span.set_attribute("processing_time_seconds", processing_time)

            logger.info(
                f"Document processed: {result['file_id']} "
                f"({result['chunks_created']} chunks)"
            )

            return UploadResponse(
                file_id=result["file_id"],
                status="success",
                chunks_created=result["chunks_created"],
                file_size=result["file_size"],
                processing_time_seconds=processing_time,
                message=f"{result['chunks_created']} chunks created"
            )

        except ValueError as e:
            documents_uploaded.add(1, {"status": "error"})
            span.record_exception(e)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        except Exception as e:
            documents_uploaded.add(1, {"status": "error"})
            logger.exception("Upload failed")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document processing failed"
            )


@app.post(
    "/documents/upload-batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Documents"]
)
async def upload_batch(
        files: List[UploadFile] = File(...),
        user_id: str = "anonymous"
) -> BatchUploadResponse:
    start_time = time.perf_counter()
    with tracer.start_as_current_span("batch_upload") as span:
        try:
            if not files:
                raise ValueError("At least one file required")
            span.set_attribute("file_count", len(files))
            span.set_attribute("user_id", user_id)
            results = []
            successful = 0

            for file in files:
                try:
                    result = ingestion_service.process_document(
                        file=file.file,
                        filename=file.filename,
                        user_id=user_id
                    )

                    results.append(result)

                    if result["status"] == "success":
                        documents_uploaded.add(1, {"status": "success"})
                        successful += 1
                    else:
                        documents_uploaded.add(1, {"status": "error"})

                except Exception as e:
                    logger.error(f"Failed to process {file.filename}: {str(e)}")
                    documents_uploaded.add(1, {"status": "error"})

                    results.append({
                        "file_id": None,
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e)
                    })

            processing_time = (time.perf_counter() - start_time)

            logger.info("Batch upload completed - successful: %d/%d", successful, len(files))
            span.set_attribute("total_files", len(files))
            span.set_attribute("successful_files", successful)
            span.set_attribute("failed_files", len(files) - successful)
            span.set_attribute("processing_time", processing_time)

            return BatchUploadResponse(
                total_files=len(files),
                successful=successful,
                failed=len(files) - successful,
                results=results
            )

        except Exception as e:
            logger.exception("Batch upload failed")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch upload failed"
            )


@app.get(
    "/documents/list",
    tags=["Documents"]
)
async def list_documents():
    with tracer.start_as_current_span("list_documents") as span:
        try:
            documents = metadata_store.list_documents()
            span.set_attribute("documents_count", len(documents))
            return {
                "total_documents": len(documents),
                "documents": documents
            }

        except Exception as e:
            logger.exception("Failed to list documents")
            span.record_exception(e)

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list documents"
            )


@app.get(
    "/documents/{file_id}",
    tags=["Documents"]
)
async def get_document_info(file_id: str):
    with tracer.start_as_current_span("get_document_info") as span:
        try:
            span.set_attribute("file_id", file_id)
            info = metadata_store.get_document_info(file_id)
            if not info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )

            return info

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to get document info")
            span.record_exception(e)

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document info"
            )


@app.delete(
    "/documents/{file_id}",
    tags=["Documents"]
)
async def delete_document(file_id: str):
    with tracer.start_as_current_span("delete_document") as span:
        try:
            span.set_attribute("file_id", file_id)
            success = ingestion_service.delete_document(file_id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found"
                )

            return {
                "status": "success",
                "message": f"Document {file_id} deleted"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to delete document")
            span.record_exception(e)

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )


@app.get(
    "/documents/stats/vector-db",
    tags=["Statistics"]
)
async def get_vector_db_stats():
    with tracer.start_as_current_span("vector_db_stats") as span:
        try:
            stats = vector_store.get_vector_db_stats(FAISS_INDEX_PATH)
            span.set_attribute("total_documents", stats.get("total_documents", 0))

            return stats

        except Exception as e:
            logger.exception("Failed to get vector DB stats")
            span.record_exception(e)

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve statistics"
            )


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG"]
)
async def query(request: QueryRequest) -> QueryResponse:
    start_time = time.perf_counter()
    with tracer.start_as_current_span("query") as span:
        try:
            span.set_attribute("user_id", request.user_id)
            span.set_attribute("session_id", request.session_id)
            span.set_attribute("query", request.query[:100])
            logger.info("RAG query started: %s ", request.query[:50])

            try:
                query_result = rag_pipeline.query(request.query)
            except PipelineError as e:
                logger.exception("Pipeline error")
                rag_queries_total.add(1, {"status": "error"})
                rag_errors.add(1)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="RAG processing failed"
                )

            retrieved_docs = query_result.sources
            retrieved_documents.add(len(retrieved_docs))
            retrieval_confidence = query_result.retrieval_confidence
            hallucination_score = query_result.hallucination_score

            if hallucination_score > 0.3:
                result = "hallucination_detected"
            elif retrieval_confidence < 0.5:
                result = "low_confidence"
            else:
                result = "success"

            rag_queries_total.add(1, attributes={"status": result})

            sources = []
            for doc in retrieved_docs:
                sources.append(
                    {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "content_preview": doc.page_content[:200]
                    }
                )

            total_time = (time.perf_counter() - start_time)
            total_latency.record(
                total_time,
                attributes={"endpoint": "/query"}
            )
            retrieval_accuracy.record(retrieval_confidence)
            hallucination_rate.record(hallucination_score)
            span.set_attribute("query.status", result)
            span.set_attribute("query.documents_retrieved", len(retrieved_docs))
            span.set_attribute("query.duration", total_time)
            span.set_attribute("query.retrieval_accuracy", retrieval_confidence)
            span.set_attribute("query.hallucination_rate", hallucination_score)
            logger.info(
                f"RAG completed "
                f"status={status} "
                f"time={total_time:.2f}s"
                f"confidence={retrieval_confidence:.1%}"
            )

            return QueryResponse(
                answer=query_result.answer,
                sources=sources,
                latency_ms=total_time * 1000,
                retrieval_confidence=retrieval_confidence,
                hallucination_score=hallucination_score
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Unexpected error in RAG query")
            rag_queries_total.add(1, {"status": "error"})
            rag_errors.add(1)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Query processing failed"
            )


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["System"]
)
async def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics_endpoint():
    return generate_latest(REGISTRY)


if __name__ == "__main__":
    logger.info("Starting RAG API server...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )
