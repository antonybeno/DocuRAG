"""
Vector Store Module

FAISS-based vector database for semantic search.

Overview:
    Manages embeddings using FAISS. Provides similarity search and document storage/retrieval.

Architecture:
    Query Vector
        ↓
    [Normalize] → [FAISS Index] → [Get Top-K]
        ↓
    Retrieve Metadata
        ↓
    Document Objects
"""
import os
import logging
from typing import List, Dict, Optional, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config.settings import FAISS_INDEX_PATH

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


class VectorStore:
    """
    FAISS vector store wrapper for semantic search.

    Manages embeddings and provides similarity search functionality.
    Loads/saves index to disk for persistence.

    Attributes:
        embeddings: HuggingFace embedding model instance
        path: Path to FAISS index directory
    """
    INDEX_FILENAME = "index.faiss"
    INDEX_PICKLE = "index.pkl"

    def __init__(self, embeddings, path: str = FAISS_INDEX_PATH) -> None:
        """
        Initialize vector store.

        Args:
            embeddings: HuggingFace embedding model instance
            path: Path to store FAISS index (created if not exists)

        Raises:
            ValueError: If embeddings is None or path is invalid
        """
        if embeddings is None:
            raise ValueError("embeddings cannot be None")

        if not path or not path.strip():
            raise ValueError("path cannot be empty")

        self.embeddings = embeddings
        self.path = path
        self.db: Optional[FAISS] = None
        self._loaded = False

        os.makedirs(path, exist_ok=True)
        logger.info("VectorStore initialized: path=%s ", self.path)

    def _ensure_loaded(self) -> bool:
        """
        Ensure FAISS index is loaded before operations.

        Returns:
            True if loaded successfully, False if no index exists
        """
        if self._loaded and self.db is not None:
            return True

        return self.load()

    def load(self) -> bool:
        """
        Load FAISS index from disk.

        Attempts to load existing index. Returns False if not found.
        Index must exist from previous save() call.

        Returns:
            True if loaded successfully, False if index not found

        Raises:
            VectorStoreError: If load fails unexpectedly
        """
        try:
            index_file = os.path.join(self.path, self.INDEX_FILENAME)

            if not os.path.exists(index_file):
                logger.info("No FAISS index found at %s ", self.path)
                self.db = None
                self._loaded = False
                return False

            logger.info("Loading FAISS index from %s ", self.path)

            self.db = FAISS.load_local(
                self.path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            self._loaded = True

            doc_count = len(self.db.index_to_docstore_id)
            logger.info("FAISS index loaded successfully: %d documents", doc_count)

            return True

        except Exception as e:
            logger.exception("Failed to load FAISS index")
            self.db = None
            self._loaded = False
            raise VectorStoreError("Failed to load FAISS index}") from e

    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to vector store.

        Process:
            1. Validate input
            2. Load existing index or create new
            3. Add document embeddings
            4. Save index to disk
            5. Update metadata

        Args:
            documents: List of Document objects with page_content and metadata

        Returns:
            True if successful, False if no documents

        Raises:
            VectorStoreError: If add operation fails
        """
        try:
            if not documents:
                logger.warning("add_documents called with empty list")
                return False

            logger.info("Adding %d documents to vector store", len(documents))

            if self.db is None:
                index_exists = self.load()

                if not index_exists:
                    logger.info("Creating new FAISS index with %d documents", len(documents))
                    os.makedirs(self.path, exist_ok=True)
                    self.db = FAISS.from_documents(
                        documents,
                        self.embeddings
                    )
                    self._loaded = True
                    logger.debug("New index created")

            else:
                logger.debug("Adding %d to existing index", len(documents))
                self.db.add_documents(documents)

            self.db.save_local(self.path)

            doc_count = len(self.db.index_to_docstore_id)
            logger.info("Documents added and index saved: %d total documents", doc_count)

            return True

        except Exception as e:
            logger.exception("Failed to add documents to vector store")
            raise VectorStoreError("Add documents failed") from e

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents.

        Finds k documents most similar to query using cosine distance.

        Process:
            1. Ensure index is loaded
            2. Embed query
            3. Search FAISS index
            4. Retrieve document metadata
            5. Return Document objects

        Args:
            query: Search query string
            k: Number of results to return (default: 5)

        Returns:
            List of Document objects sorted by similarity (best first)
            Empty list if no results or index not loaded

        Raises:
            ValueError: If query is empty or k <= 0
            VectorStoreError: If search fails
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            if k <= 0:
                raise ValueError("k must be positive, got %d", k)

            logger.debug("Similarity search: query=%s, k=%d", query[:50], k)

            if not self._ensure_loaded():
                logger.warning("FAISS index not loaded, no documents to search")
                return []

            results = self.db.similarity_search(query, k=k)
            logger.debug("Found %d similar documents", len(results))

            return results

        except ValueError as e:
            logger.exception("Invalid search parameters")
            raise VectorStoreError("Invalid search parameters") from e

        except Exception as e:
            logger.exception("Similarity search failed")
            raise VectorStoreError("Search failed") from e

    def get_vector_db_stats(self, index_path: str) -> Dict[str, Any]:
        """
        Get statistics about vector database.

        Returns information about index size, embedding dimension, etc.

        Args:
            index_path: Path to FAISS index

        Returns:
            Dict with statistics:
            - total_documents: Number of documents indexed
            - embedding_dimension: Vector dimension (usually 384)
            - status: "active", "empty", or "error"
            - index_type: Type of FAISS index (if available)

        Raises:
            VectorStoreError: If stats retrieval fails
        """
        try:
            index_file = os.path.join(index_path, self.INDEX_FILENAME)

            if not os.path.exists(index_file):
                logger.debug("No index found at %s", index_path)
                return {
                    'total_documents': 0,
                    'status': 'empty'
                }

            if self.db is None:
                self.load()

            if self.db is None:
                return {
                    'total_documents': 0,
                    'status': 'error'
                }

            doc_count = len(self.db.index_to_docstore_id)
            embedding_dim = self.db.index.d

            stats = {
                'total_documents': doc_count,
                'embedding_dimension': embedding_dim,
                'status': 'active'
            }

            logger.debug("Vector DB stats: %s ", stats)
            return stats

        except Exception as e:
            logger.error("Failed to get vector DB stats")
            return {
                'status': 'error',
                'error': str(e)
            }

    def get_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from vector store.

        Loads entire index into memory and extracts documents.

        Returns:
            List of all Document objects
            Empty list if index not loaded or empty

        Raises:
            VectorStoreError: If retrieval fails
        """
        try:
            if not self._ensure_loaded():
                logger.warning("FAISS index not loaded, returning empty list")
                return []

            if self.db is None:
                logger.warning("FAISS database is not available")
                return []

            logger.info("Retrieving all documents from FAISS")

            documents: List[Document] = []

            for doc_id in self.db.index_to_docstore_id.values():
                try:
                    doc = self.db.docstore.search(doc_id)

                    if isinstance(doc, Document):
                        documents.append(doc)
                    else:
                        logger.warning("Unexpected type in docstore %s", type(doc).__name__)

                except Exception:
                    logger.warning("Failed to retrieve document %s ", doc_id)
                    continue

            logger.info("Retrieved %d documents from FAISS", len(documents))
            return documents

        except Exception as e:
            logger.exception("Failed to retrieve all documents")
            raise VectorStoreError("Failed to get all documents") from e
