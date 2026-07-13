import time
import logging

from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app
from pydantic import BaseModel
import uvicorn

from monitoring.observability import (
    tracer,
    rag_queries_total,
    retrieved_documents,
    total_latency,
    documents_uploaded
)

from rag.pipeline import RAGPipeline
from src.config.settings import BASE_URL
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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Production RAG API")

FastAPIInstrumentor.instrument_app(app)

llm = ChatOllama(
    model="llama3:latest",
    temperature=0,
    base_url=BASE_URL
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
evaluator = RAGEvaluator(embedding_model)
retriever = HybridRetriever()
file_manager = FileManager()
parser = DocumentParser()
chunk = Chunker()
metadata = MetaDataStore()
metrics_app = make_asgi_app()

vector_store = VectorStore(
    embedding_model,
    "faiss_index"
)

rag_pipeline = RAGPipeline(
    vector_store,
    retriever,
    evaluator,
    llm
)

docu_ingestion = DocumentIngestionService(
    file_manager,
    parser,
    chunk,
    vector_store,
    metadata
)

metrics_app = make_asgi_app()
app.mount(
    "/metrics",
    metrics_app
)


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


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = "anonymous"):
    start_time = time.time()
    with tracer.start_as_current_span(
            "document_upload"
    ) as span:
        try:
            logger.info(f"Document upload started: {file.filename}")
            span.set_attribute("filename", file.filename)
            span.set_attribute("user_id", user_id)

            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="Filename required"
                )

            result = docu_ingestion.process_document(
                file=file.file,
                filename=file.filename,
                user_id=user_id
            )

            processing_time = (time.time() - start_time)

            if result["status"] == "success":
                logger.info(
                    f"Document processed: "
                    f"{result['file_id']} "
                    f"({result['chunks_created']} chunks)"
                )
                documents_uploaded.add(1, attributes={"status": "success"})
                span.set_attribute("status", "success")
                span.set_attribute("chunks_created", result["chunks_created"])

                return JSONResponse(
                    status_code=200,
                    content={
                        "file_id": result["file_id"],
                        "status": "success",
                        "chunks_created": result["chunks_created"],
                        "file_size": result["file_size"],
                        "processing_time_seconds": processing_time,
                        "message": f"{result['chunks_created']} chunks created"
                    }
                )
            else:
                documents_uploaded.add(1, attributes={"status": "error"})

                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": result.get("error","Unknown error")
                    }
                )
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            documents_uploaded.add(1, attributes={"status": "error"})
            span.record_exception(e)
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


@app.get("/documents/list")
async def list_documents():
    documents = metadata.list_documents()
    return {
        "total_documents": len(documents),
        "documents": documents
    }


@app.get("/documents/{file_id}")
async def get_document_info(file_id: str):
    info = metadata.get_document_info(file_id)

    if not info:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return info


@app.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    success = docu_ingestion.delete_document(file_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {
        "status": "success",
        "message": f"Document {file_id} deleted"
    }


@app.get("/documents/stats/vector-db")
async def get_vector_db_stats():
    return vector_store.get_vector_db_stats("faiss_index")


@app.post("/documents/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...), user_id: str = "anonymous"):
    results = []

    with tracer.start_as_current_span(
            "batch_upload"
    ) as span:
        span.set_attribute("file_count", len(files))
        span.set_attribute("user_id", user_id)

        for file in files:
            result = (
                docu_ingestion.process_document(
                    file=file.file,
                    filename=file.filename,
                    user_id=user_id
                )
            )

            results.append(result)

            if result["status"] == "success":
                documents_uploaded.add(1, attributes={"status": "success"})
            else:
                documents_uploaded.add(1, attributes={"status": "error"})

        successful = sum(
            1
            for r in results
            if r["status"] == "success"
        )

        logger.info(
            f"Batch upload completed: "
            f"{successful}/{len(files)}"
        )

        return {
            "total_files": len(files),
            "successful": successful,
            "failed": len(files) - successful,
            "results": results
        }


@app.post("/query")
async def rag_query(request: QueryRequest) -> QueryResponse:
    start_time = time.time()

    with tracer.start_as_current_span(
            "rag_query"
    ) as span:
        try:
            span.set_attribute("user_id", request.user_id)
            span.set_attribute("session_id", request.session_id)
            result = rag_pipeline.query(request.query)
            answer = result["answer"]
            retrieved_docs = result["documents"]
            retrieved_documents.add(len(retrieved_docs))
            metrics = result.get("metrics", {})
            retrieval_confidence = float(metrics.get("retrieval_accuracy", 0.0))
            hallucination_score = float(metrics.get("hallucination_rate", 0.0))
            span.set_attribute("retrieval_accuracy", retrieval_confidence)
            span.set_attribute("hallucination_rate", hallucination_score)

            if hallucination_score > 0.3:
                status = "hallucination_detected"
            elif retrieval_confidence < 0.5:
                status = "low_confidence"
            else:
                status = "success"

            rag_queries_total.add(1, attributes={"status": status})

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

            total_time = (time.time() - start_time)
            total_latency.record(
                total_time,
                attributes={
                    "endpoint": "rag_query"
                }
            )
            logger.info(
                f"RAG completed "
                f"status={status} "
                f"time={total_time:.2f}s"

            )

            return QueryResponse(
                answer=answer,
                sources=sources,
                latency_ms=total_time * 1000,
                retrieval_confidence=retrieval_confidence,
                hallucination_score=hallucination_score
            )
        except Exception as e:
            rag_queries_total.add(1, attributes={"status": "error"})
            logger.error(
                f"RAG query failed: {str(e)}",
                exc_info=True
            )

            span.record_exception(e)
            span.set_attribute("error", True)

            raise HTTPException(
                status_code=500,
                detail=str(e)
            )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001
    )
