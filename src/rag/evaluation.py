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
            [doc.page_content for doc in docs]
        )

    def retrieval_similarity(
        self,
        query: str,
        retrieved_docs: List[Document],
    ) -> float:
        if not retrieved_docs:
            return 0.0

        try:
            query_embedding = self._embed_query(query)

            doc_embeddings = self._embed_documents(
                retrieved_docs
            )

            similarities = cosine_similarity(
                [query_embedding],
                doc_embeddings
            )[0]

            weights = np.linspace(
                1.0,
                0.5,
                len(similarities)
            )

            score = float(
                np.average(
                    similarities,
                    weights=weights
                )
            )

            logger.info(
                "Retrieval similarity: %.3f",
                score
            )

            return score
        except Exception as e:
            logger.exception(
                f"Failed to compute retrieval similarity: {e}"
            )
            return 0.0

    def answer_grounding(
        self,
        answer: str,
        retrieved_docs: List[Document],
    ) -> float:
        if not retrieved_docs:
            return 0.0
        try:
            answer_embedding = self._embed_query(answer)

            doc_embeddings = self._embed_documents(
                retrieved_docs
            )

            similarities = cosine_similarity(
                [answer_embedding],
                doc_embeddings
            )[0]

            score = float(np.max(similarities))

            logger.info(
                "Answer grounding: %.3f",
                score
            )

            return score
        except Exception as e:
            logger.exception(
                f"Failed to compute answer grounding: {e}"
            )
            return 0.0

    def context_coverage(
        self,
        answer: str,
        retrieved_docs: List[Document],
    ) -> float:
        if not retrieved_docs:
            return 0.0

        try:
            answer_embedding = self._embed_query(answer)

            doc_embeddings = self._embed_documents(
                retrieved_docs
            )

            similarities = cosine_similarity(
                [answer_embedding],
                doc_embeddings
            )[0]

            return float(np.mean(similarities))
        except Exception as e:
            logger.exception(
                f"Failed to compute context coverage: {e}"
            )
            return 0.0

    @staticmethod
    def document_count(
        retrieved_docs: List[Document]
    ) -> int:
        return len(retrieved_docs)

    @staticmethod
    def average_document_length(
        retrieved_docs: List[Document]
    ) -> float:
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

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
    ) -> Dict[str, float]:
        retrieval = self.retrieval_similarity(
            query,
            retrieved_docs
        )

        grounding = self.answer_grounding(
            answer,
            retrieved_docs
        )

        coverage = self.context_coverage(
            answer,
            retrieved_docs
        )

        return {
            "retrieval_similarity": retrieval,
            "answer_grounding": grounding,
            "context_coverage": coverage,
            "retrieved_documents": self.document_count(retrieved_docs),
            "average_document_length": self.average_document_length(retrieved_docs),
        }
