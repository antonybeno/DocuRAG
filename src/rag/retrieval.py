"""
Hybrid Retrieval Module

Implements hybrid retrieval combining:
1. Dense Retrieval (Vector Similarity): Uses FAISS for semantic similarity
2. Sparse Retrieval (BM25): Keyword-based matching

Architecture:
    Query → [Vector Search] → Results (k=5)
         ↓
         [BM25 Search] → Results (k=5)
         ↓
    Merge & Deduplicate → Unique Documents → LLM
"""

import logging
from typing import List, Tuple, Set

from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever

from src.config.settings import TOP_K_RETRIEVAL, BM25_K

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval.

    Attributes:
        k: Number of documents to retrieve from each method
        bm25: BM25Retriever instance (initialized by create_bm25_index)

    Note:
        BM25 index must be initialized with create_bm25_index() before
        calling retrieve() with hybrid mode.
    """

    def __init__(self, k: int = TOP_K_RETRIEVAL) -> None:
        """
        Initialize hybrid retriever.

        Args:
            k: Number of documents to retrieve from each retrieval method.
               Default is TOP_K_RETRIEVAL (5).

        Raises:
            ValueError: If k <= 0
        """
        if k <= 0:
            raise ValueError("k must be positive, got %d", k)

        self.k = k
        self.bm25 = None
        self.total_bm25_documents = 0

        logger.debug("HybridRetriever initialized with k=%d", k)

    def create_bm25_index(self, documents: List[Document]) -> None:
        """
        Build BM25 keyword search index from documents.

        This must be called before retrieval with BM25 is possible.
        Should be called:
        - After initial document ingestion
        - When new documents are added
        - When rebuilding the system

        Args:
            documents: List of Document objects to index

        Raises:
            Exception: If BM25 index creation fails
        """
        try:
            if not documents:
                logger.error("create_bm25_index called with empty document list, BM25 index not created")
                self.bm25 = None
                self.total_bm25_documents = 0
                return

            self.bm25 = BM25Retriever.from_documents(documents)
            self.bm25.k = BM25_K
            self.total_bm25_documents = len(documents)

            logger.info(
                "BM25 index created successfully: %d documents, k=%d",
                len(documents),
                self.bm25.k
            )

        except Exception:
            logger.exception("Failed to create BM25 index")
            raise

    @staticmethod
    def _get_document_key(document: Document) -> Tuple[str, int, int]:
        """
        Generate unique key for document deduplication.

        Uses page content + metadata to identify duplicates.
        Rationale: Same content from different chunks should be counted once.

        Args:
            document: Document to generate key for

        Returns:
            Tuple of (source, chunk_index, content_hash_prefix)
        """
        source = document.metadata.get("source", "unknown")
        chunk_index = document.metadata.get("chunk_index", -1)

        # Use first 10 chars of content for hash (faster than full hash)
        content_prefix = hash(document.page_content) % 10000

        return source, chunk_index, content_prefix

    @staticmethod
    def _remove_duplicates(documents: List[Document]) -> List[Document]:
        """
        Remove duplicate documents from list.

        Preserves order: first occurrence is kept.
        Uses _get_document_key for deduplication logic.

        Args:
            documents: List potentially containing duplicates

        Returns:
            List with duplicates removed, order preserved
        """
        seen_keys = set()
        unique_documents = []

        for document in documents:
            key = HybridRetriever._get_document_key(document)

            if key not in seen_keys:
                seen_keys.add(key)
                unique_documents.append(document)

        logger.debug(
            "Removed %d duplicates from %d documents",
            (len(documents) - len(unique_documents)),
            {len(documents)}
        )

        return unique_documents

    def retrieve(self, question: str, vector_store) -> List[Document]:
        """
        Retrieve documents using hybrid retrieval strategy.

        Process:
            1. Dense Search: Query FAISS vector database
               - Returns k most similar documents by embedding
               - Captures semantic meaning

            2. Sparse Search: Query BM25 index
               - Returns k most relevant documents by keyword
               - Captures exact terminology
               - Falls back to dense-only if not initialized

            3. Merge: Combine results (may have duplicates)

            4. Deduplicate: Remove exact/near duplicates
               - Preserves order (dense results first)
               - Keeps most relevant occurrence

        Args:
            question: User query/question
            vector_store: VectorStore instance with similarity_search method

        Returns:
            List of unique Document objects, sorted by relevance.

        Raises:
            ValueError: If question is empty
        """
        try:
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")

            logger.debug("Starting hybrid retrieval for: %s ", question[:50])

            # ==================== DENSE RETRIEVAL ====================
            try:
                vector_docs = vector_store.similarity_search(question, k=self.k)

                if not vector_docs:
                    logger.warning("Vector search returned no results")
                    vector_docs = []
                else:
                    logger.debug("Vector search returned %d documents", len(vector_docs))

            except Exception:
                logger.exception("Vector search failed")
                vector_docs = []

            # ==================== SPARSE RETRIEVAL ====================
            if self.bm25 is None:
                logger.warning(
                    "BM25 index not initialized. Using dense retrieval only. "
                    "Call create_bm25_index() to enable hybrid retrieval."
                )
                keyword_docs = []
            else:
                try:
                    keyword_docs = self.bm25.invoke(question)

                    if not keyword_docs:
                        logger.debug("BM25 search returned no results")
                        keyword_docs = []
                    else:
                        logger.debug("BM25 search returned %d documents", len(keyword_docs))

                except Exception:
                    logger.exception("BM25 search failed")
                    keyword_docs = []

            # ==================== MERGE ====================
            merged_docs = vector_docs + keyword_docs

            if not merged_docs:
                logger.warning("No documents retrieved for query: %s ", {question[:50]})
                return []

            # ==================== DEDUPLICATE ====================
            unique_docs = self._remove_duplicates(merged_docs)

            # ==================== LOG METRICS ====================
            logger.info(
                "Hybrid retrieval completed: "
                "vector=%d, "
                "bm25=%d, "
                "merged=%d, "
                "unique=%d, ",
                {len(vector_docs)},
                {len(keyword_docs)},
                {len(merged_docs)},
                {len(unique_docs)}
            )

            return unique_docs

        except ValueError as e:
            logger.error("Invalid input for retrieval: %s.", str(e))
            return []

        except Exception:
            logger.exception(f"Unexpected error during retrieval")
            return []
