import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever

from src.vectorstore.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, k: int = 6):
        self.k = k
        self.bm25 = None

    def create_bm25_index(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents for BM25")
            return

        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = self.k

        logger.info(f"BM25 created with {len(documents)} documents")

    @staticmethod
    def _document_key(document: Document) -> tuple:
        metadata = document.metadata
        return (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("chunk_index"),
            document.page_content
        )

    @staticmethod
    def _remove_duplicates(documents: List[Document]) -> List[Document]:
        unique_documents = {}
        for document in documents:
            key = HybridRetriever._document_key(document)
            unique_documents[key] = document

        return list(unique_documents.values())

    def retrieve(self, question: str, vector_store: VectorStore) -> List[Document]:
        vector_docs = vector_store.similarity_search(question, self.k)
        if self.bm25:
            keyword_docs = self.bm25.invoke(question)
        else:
            logger.warning("BM25 index is not initialized. Using dense retrieval only.")
            keyword_docs = []

        merged_docs = vector_docs + keyword_docs

        unique_docs = self._remove_duplicates(
            merged_docs
        )

        logger.info(f"Hybrid retrieval completed."
                    f"Dense= {len(vector_docs)},"
                    f"BM25= {len(keyword_docs)},"
                    f"Returned docs= {len(unique_docs)}")

        return unique_docs
