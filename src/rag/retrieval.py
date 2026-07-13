import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, k: int = 6):
        self.k = k
        self.bm25 = None

    def create_bm25_index(
            self,
            documents: List[Document]
    ):

        if not documents:
            logger.warning(
                "No documents for BM25"
            )
            return

        self.bm25 = BM25Retriever.from_documents(
            documents
        )

        self.bm25.k = self.k

        logger.info(
            f"BM25 created with {len(documents)} documents"
        )

    def retrieve(
            self,
            question: str,
            vector_store
    ) -> List[Document]:

        vector_docs = vector_store.similarity_search(
            question,
            self.k
        )

        if self.bm25:
            keyword_docs = self.bm25.invoke(question)
        else:
            keyword_docs = []

        merged = vector_docs + keyword_docs

        unique_docs = list(
            {
                doc.page_content: doc
                for doc in merged
            }.values()
        )

        logger.info(
            f"Hybrid returned {len(unique_docs)} documents"
        )

        return unique_docs
