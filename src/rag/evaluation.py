from typing import List, Dict

import logging
import numpy as np
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def _embed_query(self, text: str):
        return self.embedding_model.embed_query(text)

    def _embed_documents(self, docs: List[Document]):
        return self.embedding_model.embed_documents(
            [
                doc.page_content
                for doc in docs
            ]
        )

    @staticmethod
    def _normalize_score(score: float) -> float:
        return float(max(0.0, min(score, 1.0)))

    def context_relevance(self, query_embedding, document_embeddings) -> float:
        if not document_embeddings:
            return 0.0
        try:
            similarities = cosine_similarity(normalize([query_embedding]), normalize(document_embeddings))[0]
            weights = np.linspace(1.0, 0.5, len(similarities))
            score = np.average(similarities, weights=weights)
            return self._normalize_score(score)
        except Exception:
            logger.exception("Failed to compute context relevance")
            return 0.0

    def faithfulness(self, answer_embedding, document_embeddings) -> float:
        if not document_embeddings:
            return 0.0
        try:
            similarities = cosine_similarity(normalize([answer_embedding]), normalize(document_embeddings))[0]
            score = np.max(similarities)
            return self._normalize_score(score)
        except Exception:
            logger.exception("Failed to compute faithfulness")
            return 0.0

    def context_recall(self, answer_embedding, document_embeddings) -> float:
        if not document_embeddings:
            return 0.0
        try:
            similarities = cosine_similarity(normalize([answer_embedding]), normalize(document_embeddings))[0]
            score = np.mean(similarities)
            return self._normalize_score(score)
        except Exception:
            logger.exception("Failed to compute context recall")
            return 0.0

    @staticmethod
    def document_count(retrieved_docs: List[Document]) -> int:
        return len(retrieved_docs)

    @staticmethod
    def average_document_length(retrieved_docs: List[Document]) -> float:
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
        try:
            query_embedding = self._embed_query(query)
            answer_embedding = self._embed_query(answer)
            document_embeddings = self._embed_documents(retrieved_docs)

            context_relevance = self.context_relevance(query_embedding, document_embeddings)
            faithfulness = self.faithfulness(answer_embedding, document_embeddings)
            context_recall = self.context_recall(answer_embedding, document_embeddings)

            return {
                "context_relevance": context_relevance,
                "faithfulness": faithfulness,
                "context_recall": context_recall,
                "retrieved_documents": self.document_count(retrieved_docs),
                "average_document_length": self.average_document_length(retrieved_docs),
            }
        except Exception:
            logger.exception("RAG evaluation failed")
            return {
                "context_relevance": 0.0,
                "faithfulness": 0.0,
                "context_recall": 0.0,
                "retrieved_documents": self.document_count(retrieved_docs),
                "average_document_length": self.average_document_length(retrieved_docs),
            }