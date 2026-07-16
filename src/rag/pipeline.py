"""
RAG Pipeline Module

Implements the complete Retrieval-Augmented Generation (RAG) pipeline:

    User Query
        ↓
    [RETRIEVAL] → Fetch k relevant documents from vector DB
        ↓
    [GENERATION] → Feed docs + query to LLM for answer generation
        ↓
    [EVALUATION] → Measure retrieval quality & hallucination
        ↓
    Response + Metrics

Pipeline Flow:
    1. Retrieval Phase:
       - Hybrid search (dense + sparse)
       - Deduplication
       - Format context
       - Metrics: latency, doc count, scores

    2. Generation Phase:
       - Construct prompt with context
       - Call LLM with timeout
       - Extract response
       - Metrics: latency, token usage

    3. Evaluation Phase:
       - Compute retrieval quality metrics
       - Detect hallucinations
       - Evaluate answer faithfulness
       - Metrics: confidence scores

    4. Response:
       - Return answer + metadata
       - Log for monitoring
       - Record metrics
"""

import time
import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from src.config.settings import LLM_TIMEOUT, LLM_MAX_TOKENS, TOP_K_RETRIEVAL
from src.monitoring.observability import (
    tracer,
    retrieval_latency,
    llm_latency,
    total_latency,
    rag_errors,
    rag_queries_total
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when pipeline operations fail."""
    pass


class QueryResponse:
    """
    Response object for RAG queries.

    Attributes:
        answer: Generated answer text
        sources: Retrieved documents
        retrieval_confidence: 0-1 score of retrieval quality
        hallucination_score: 0-1 score of hallucination risk (0=no hallucination)
        latency_ms: Total pipeline latency in milliseconds
    """

    def __init__(
            self,
            answer: str,
            sources: List[Document],
            retrieval_confidence: float,
            hallucination_score: float
    ) -> None:
        self.answer = answer
        self.sources = sources
        self.retrieval_confidence = max(0.0, min(1.0, retrieval_confidence))
        self.hallucination_score = max(0.0, min(1.0, hallucination_score))

    """Convert response to dictionary for JSON serialization."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [doc.metadata.get("source", "unknown") for doc in self.sources],
            "retrieval_confidence": round(self.retrieval_confidence, 3),
            "hallucination_score": round(self.hallucination_score, 3),
            "document_count": len(self.sources)
        }


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.

    Coordinates retrieval, generation, and evaluation to produce high-quality, grounded answers to user queries.

    Attributes:
        vector_store: VectorStore instance for similarity search
        retriever: HybridRetriever for dense + sparse search
        evaluator: RAGEvaluator for quality metrics
        llm: Language model for answer generation
    """

    def __init__(
            self,
            vector_store,
            retriever,
            evaluator,
            llm
    ):
        """
        Initialize RAG pipeline.

        Args:
            vector_store: VectorStore instance
            retriever: HybridRetriever instance
            evaluator: RAGEvaluator instance
            llm: Language model instance

        Raises:
            PipelineError: If any component is None
        """
        if not all([vector_store, retriever, evaluator, llm]):
            logger.error("All pipeline components must be provided")
            raise PipelineError("All pipeline components must be provided")

        self.vector_store = vector_store
        self.retriever = retriever
        self.evaluator = evaluator
        self.llm = llm
        self._initialized = False
        logger.info("RAGPipeline initialized")

    def initialize(self):
        """
        Initialize BM25 index from existing documents.

        Must be called after pipeline creation and before first query.
        Loads all documents from vector store and builds keyword index.

        Raises:
            Exception: If initialization fails
        """
        try:
            logger.info("Initializing RAG pipeline...")

            with tracer.start_as_current_span("pipeline_initialization") as span:
                docs = self.vector_store.get_all_documents()
                if not docs:
                    logger.warning("No documents available for BM25 initialization")
                    self._initialized = True
                    return

                self.retriever.create_bm25_index(docs)
                self._initialized = True

                span.set_attribute("pipeline.initialized_documents", len(docs))
                logger.info("Pipeline initialized with %d documents", {len(docs)})

        except Exception as e:
            logger.exception("Pipeline initialization failed")
            raise PipelineError("Pipeline initialization failed") from e

    def generate_answer(self, question: str, documents: List[Document]):
        try:
            context = "\n\n".join(
                doc.page_content
                for doc in documents
            )
            prompt = f"""You are a technical assistant. Your task is to answer 
                        questions based ONLY on the provided context. If the answer is not in the context, 
                        explicitly state "Cannot find this information in the provided documents."
                        
                        IMPORTANT RULES:
                        1. Answer only using information from the context
                        2. If unsure, say "I don't know" rather than guess
                        3. Quote relevant passages when appropriate
                        4. Do not hallucinate or make up information
                        5. Be concise and clear
                        
                        Context:
                        
                        {context}
                        
                        Question:
                        
                        {question}
                        """
            response = self.llm.invoke(prompt, timeout=30)
            return response.content

        except TimeoutError as e:
            logger.warning("LLM request timed out after %ss", LLM_TIMEOUT)
            raise PipelineError(f"LLM request timed out after {LLM_TIMEOUT}s") from e
        except Exception as e:
            logger.exception("LLM generation failed")
            raise PipelineError("LLM generation failed") from e

    def query(self, question: str) -> QueryResponse:
        """
        Execute complete RAG pipeline for a query.

        Process:
            1. Retrieve relevant documents
            2. Generate answer using LLM
            3. Evaluate quality and hallucination
            4. Record metrics
            5. Return response

        Args:
            question: User question

        Returns:
            QueryResponse with answer and metrics

        Raises:
            PipelineError: If any pipeline stage fails

        Metrics Recorded:
            - retrieval_latency: Document retrieval time
            - llm_latency: LLM response time
            - total_latency: End-to-end pipeline time
            - rag_queries_total: Query counter
            - rag_errors: Error counter (if failed)
        """
        try:
            logger.info(f"Starting RAG query: {question[:50]}")
            with tracer.start_as_current_span("retrieve_documents") as retrieval_span:
                try:
                    retrieval_start = time.perf_counter()

                    retrieved_docs = self.retriever.retrieve(question, self.vector_store)

                    retrieval_duration = time.perf_counter() - retrieval_start
                    retrieval_latency.record(retrieval_duration)
                    retrieval_span.set_attribute("query.retrieved_documents_count", len(retrieved_docs))
                    retrieval_span.set_attribute("query.retrieved_documents_duration", retrieval_duration)

                    logger.debug(
                        "Retrieved %d documents in %.2fs",
                        len(retrieved_docs),
                        retrieval_duration
                    )

                except Exception as e:
                    logger.error(f"Retrieval failed: {str(e)}")
                    raise PipelineError(f"Document retrieval failed: {str(e)}") from e

            with tracer.start_as_current_span("llm_generate") as llm_spam:
                try:
                    llm_start = time.perf_counter()

                    answer = self.generate_answer(question, retrieved_docs)

                    llm_duration = time.perf_counter() - llm_start
                    llm_latency.record(llm_duration)
                    llm_spam.set_attribute("query.model", "llama3")
                    llm_spam.set_attribute("query.llm_duration", llm_duration)
                    logger.debug("LLM generation duration %.2fs",llm_duration)

                except Exception as e:
                    logger.error(f"LLM answer generation failed: {str(e)}")
                    raise PipelineError(f"LLM generation failed: {str(e)}") from e

            with tracer.start_as_current_span("evaluate_answer") as eval_span:
                try:
                    eval_start = time.perf_counter()

                    evaluation = self.evaluator.evaluate(question, answer, retrieved_docs)

                    eval_duration = time.perf_counter() - eval_start
                    eval_span.set_attribute("query.evaluation_duration", eval_duration)
                    eval_span.set_attribute("context_relevance",
                                            evaluation.get("context_relevance", 0))
                    eval_span.set_attribute("faithfulness",
                                            evaluation.get("faithfulness", 0))

                    logger.debug("Evaluation complete")

                except Exception as e:
                    logger.error(f"Evaluation failed (continuing): {str(e)}")
                    evaluation = {
                        "context_relevance": 0.0,
                        "faithfulness": 0.0,
                        "context_recall": 0.0
                    }

            return QueryResponse(
                answer=answer,
                sources=retrieved_docs,
                retrieval_confidence=evaluation["context_relevance"],
                hallucination_score=1 - evaluation["faithfulness"]
            )

        except Exception as e:
            logger.exception("Unexpected error in RAG pipeline")
            raise PipelineError("Unexpected pipeline error") from e
