"""
RAG Evaluation Module

Implements multiple evaluation metrics for RAG system quality:

Metrics Computed:
    1. Context Relevance (0-1):
       - Measures how relevant retrieved documents are to the query
       - Uses cosine similarity between query and documents
       - Weighted average: recent documents weighted higher
       - Interpretation: 0.8+ = excellent, 0.5-0.8 = good, <0.5 = poor

    2. Faithfulness (0-1):
       - Measures if answer is grounded in retrieved documents
       - Uses cosine similarity between answer and documents
       - Takes maximum similarity (answer should match SOME document)
       - Interpretation: 0.8+ = faithful, <0.5 = hallucination risk

    3. Context Recall (0-1):
       - Measures how much information from documents is in answer
       - Uses average cosine similarity
       - Interpretation: High = answer uses most context

Hallucination Detection:
    Computed as: hallucination_score = 1 - faithfulness
    - 0.0 = No hallucination (answer fully grounded)
    - 1.0 = Complete hallucination (no grounding)
    - Threshold: > 0.3 usually indicates hallucination
"""

import logging
from typing import List, Dict, Optional

import numpy as np
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.config.settings import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation operations fail."""
    pass


class RAGEvaluator:
    """
    RAG system quality evaluator.

    Computes multiple metrics to assess retrieval quality and answer faithfulness.

    Attributes:
        embedding_model: HuggingFace embedding model instance
        relevance_threshold: Score above which context is considered relevant
        faithfulness_threshold: Score below which answer is considered hallucinated

    Note:
        All metrics are normalized to [0, 1] range for consistency.
        Missing context returns 0.0 (worst case, not undefined).
    """
    RELEVANCE_THRESHOLD = 0.5
    FAITHFULNESS_THRESHOLD = 0.3

    def __init__(self, embedding_model) -> None:
        """
        Initialize evaluator.

        Args:
            embedding_model: HuggingFace embedding model

        Raises:
            ValueError: If embedding_model is None
        """
        if embedding_model is None:
            raise ValueError("embedding_model cannot be None")

        self.embedding_model = embedding_model
        logger.info("RAGEvaluator initialized")

    def _embed_query(self, text: str) -> Optional[np.ndarray]:
        """
        Embed query text using the embedding model.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector

        Raises:
            EvaluationError: If embedding fails
        """
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")

            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.exception("Failed to embed query")
            raise EvaluationError("Query embedding failed") from e

    def _embed_documents(self, docs: List[Document]) -> Optional[np.ndarray]:
        """
        Embed multiple documents using the embedding model.

        Args:
            docs: List of Document objects to embed

        Returns:
            2D numpy array of embeddings (n_docs x embedding_dim)

        Raises:
            EvaluationError: If embedding fails
        """
        try:
            if not docs:
                return np.array([], dtype=np.float32).reshape(0, 384)

            texts = [doc.page_content for doc in docs]
            embeddings = self.embedding_model.embed_documents(texts)
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.exception("Failed to embed documents")
            raise EvaluationError("Document embedding failed") from e

    @staticmethod
    def _normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize score to [min_val, max_val] range.

        Args:
            score: Raw score value
            min_val: Minimum allowed value (default: 0.0)
            max_val: Maximum allowed value (default: 1.0)

        Returns:
            Clipped and normalized score
        """
        return float(max(min_val, min(score, max_val)))

    def context_relevance(
            self,
            query_embedding: np.ndarray,
            document_embeddings: np.ndarray
    ) -> float:
        """
        Measure relevance of retrieved documents to query.

        Args:
            query_embedding: Query embedding (1D array)
            document_embeddings: Document embeddings (2D array, n_docs x embedding_dim)

        Returns:
            Relevance score 0-1 (1 = highly relevant, 0 = not relevant)

        Raises:
            EvaluationError: If computation fails
        """

        if len(document_embeddings) == 0:
            logger.warning("context_relevance called with empty documents")
            return 0.0
        try:
            similarities = cosine_similarity(normalize([query_embedding]), normalize(document_embeddings))[0]
            weights = np.linspace(1.0, 0.5, len(similarities))
            score = np.average(similarities, weights=weights)
            return self._normalize_score(score)
        except Exception as e:
            logger.exception("Failed to compute context relevance")
            raise EvaluationError("Context relevance computation failed") from e

    def faithfulness(
            self,
            answer_embedding: np.ndarray,
            document_embeddings: np.ndarray
    ) -> float:
        """
        Measure if answer is grounded in retrieved documents (no hallucination).

        Args:
            answer_embedding: Answer embedding (1D array)
            document_embeddings: Document embeddings (2D array)

        Returns:
            Faithfulness score 0-1 (1 = faithful, 0 = hallucination)

        Raises:
            EvaluationError: If computation fails
        """
        if len(document_embeddings) == 0:
            logger.warning("faithfulness called with empty documents")
            return 0.0
        try:
            similarities = cosine_similarity(normalize([answer_embedding]), normalize(document_embeddings))[0]
            score = np.max(similarities)
            return self._normalize_score(score)
        except Exception as e:
            logger.exception("Failed to compute faithfulness")
            raise EvaluationError("Failed to compute faithfulness") from e

    def context_recall(
            self,
            answer_embedding: np.ndarray,
            document_embeddings: np.ndarray
    ) -> float:
        """
        Measure how much information from documents is present in answer.

        Args:
            answer_embedding: Answer embedding (1D array)
            document_embeddings: Document embeddings (2D array)

        Returns:
            Recall score 0-1 (1 = high recall, 0 = low recall)

        Raises:
            EvaluationError: If computation fails
        """
        if len(document_embeddings) == 0:
            logger.warning("context_recall called with empty documents")
            return 0.0
        try:
            similarities = cosine_similarity(normalize([answer_embedding]), normalize(document_embeddings))[0]
            score = np.mean(similarities)
            return self._normalize_score(score)
        except Exception as e:
            logger.exception("Failed to compute context recall")
            raise EvaluationError("Failed to compute context recall") from e

    @staticmethod
    def document_count(retrieved_docs: List[Document]) -> int:
        """
        Count number of retrieved documents.

        Args:
            retrieved_docs: List of documents

        Returns:
            Number of documents
        """
        return len(retrieved_docs)

    @staticmethod
    def average_document_length(retrieved_docs: List[Document]) -> float:
        """
        Compute average length of retrieved documents.

        Args:
            retrieved_docs: List of documents

        Returns:
            Average content length in characters (0.0 if empty)
        """
        if not retrieved_docs:
            return 0.0

        return float(
            np.mean(
                [
                    len(doc.page_content)
                    for doc in retrieved_docs
                ]
            )
        )

    def evaluate(self, query: str, answer: str, retrieved_docs: List[Document]) -> Dict[str, float]:
        """
        Comprehensive RAG evaluation.

        Computes all metrics for a query-answer-documents triple.

        Process:
            1. Embed query, answer, and documents
            2. Compute relevance
            3. Compute faithfulness
            4. Compute recall
            5. Compute document statistics
            6. Return all metrics

        Args:
            query: User question
            answer: Generated answer
            retrieved_docs: Documents used for generation

        Returns:
            Dictionary with metrics:
            - context_relevance: Query-doc similarity (0-1)
            - faithfulness: Answer grounding (0-1)
            - context_recall: Document coverage (0-1)
            - retrieved_documents: Count
            - average_document_length: Avg length

        Raises:
            EvaluationError: If any sub-evaluation fails
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            if not answer or not answer.strip():
                raise ValueError("Answer cannot be empty")

            query_embedding = self._embed_query(query)
            answer_embedding = self._embed_query(answer)
            document_embeddings = self._embed_documents(retrieved_docs)

            context_relevance = self.context_relevance(query_embedding, document_embeddings)
            faithfulness = self.faithfulness(answer_embedding, document_embeddings)
            context_recall = self.context_recall(answer_embedding, document_embeddings)

            logger.debug("Evaluation Completed")

            return {
                "context_relevance": context_relevance,
                "faithfulness": faithfulness,
                "context_recall": context_recall,
                "retrieved_documents": self.document_count(retrieved_docs),
                "average_document_length": self.average_document_length(retrieved_docs),
            }
        except Exception as e:
            logger.exception("RAG evaluation failed")
            raise EvaluationError("Evaluation failed") from e
