"""
Document Chunking Module

Handles splitting of parsed documents into smaller chunks suitable for embedding
and retrieval. Uses recursive character splitting with overlap to maintain context.

Strategy:
1. Split by double newlines (paragraph boundaries)
2. Split by single newlines (sentence boundaries)
3. Split by spaces (word boundaries)
4. Split by characters (last resort)

This hierarchical approach preserves semantic units (paragraphs/sentences) while ensuring size constraints are met.
"""
import hashlib
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.MetadataStore import MetaDataStore

logger = logging.getLogger(__name__)


class Chunker:
    """
    Document chunker using recursive character splitting.

    Attributes:
        text_splitter: RecursiveCharacterTextSplitter instance
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters (default from config)
            chunk_overlap: Overlap between chunks (default from config)

        Raises:
            ValueError: If chunk_overlap >= chunk_size
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap: %d must be < chunk_size: %d.", chunk_overlap, chunk_size)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )

        logger.debug("Chunker initialized: size=%d, overlap=%d", chunk_size, chunk_overlap)

    def chunk_document(self, texts: List[str], file_id: str, metadata_store: MetaDataStore) -> List[Document]:
        """
        Split document texts into chunks with metadata.

        Args:
            texts: List of text strings
            file_id: Unique identifier of the source document
            metadata_store: Metadata store for document info

        Returns:
            List of Document objects with metadata

        Raises:
            Exception: If chunking fails
        """
        try:
            if not file_id:
                logger.error("Chunking failed: file_id cannot be empty")
                return []

            metadata = metadata_store.get_document_info(file_id)
            if not metadata:
                logger.error("Chunking failed: No metadata found for file_id: %s", file_id)
                return []

            documents = []
            chunk_index = 0

            for text_idx, text in enumerate(texts):
                if not text or not text.strip():
                    logger.debug(f"Skipping empty text at index {text_idx}")
                    continue

                chunks = self.text_splitter.split_text(text)

                for chunk_idx, chunk in enumerate(chunks):
                    chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "file_id": file_id,
                            "chunk_index": chunk_index,
                            "text_index": text_idx,
                            "source": metadata.get("original_name", "unknown"),
                            "upload_time": metadata.get("upload_timestamp"),
                            "chunk_size": len(chunk),
                            "chunk_hash": chunk_hash,
                            "file_type": metadata.get("file_type")
                        }
                    )

                    documents.append(doc)
                    chunk_index += 1

            if not documents:
                logger.error("Chunking failed: No chunks created from texts for file_id: %s", file_id)
                return []

            logger.info(
                "Chunked document successfully. file_id: %s chunks: %d",
                file_id,
                len(documents)
            )

            return documents

        except Exception:
            logger.exception("Chunking failed for file_id: %s", file_id)
            return []
