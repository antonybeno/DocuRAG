

"""
Metadata Store Module

In-memory metadata management for documents.

Overview:
    Stores and manages document metadata (file info, processing status, etc.).
    Uses in-memory dictionary (NOT persistent).

Status Values:
    - uploaded: File saved to disk
    - parsing: Parsing document
    - parse_failed: Parsing error
    - chunking: Splitting text
    - chunk_failed: Chunking error
    - storing: Embedding and storing
    - storage_failed: Storage error
    - processed: Complete and ready
    - failed: Processing failed
"""

import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class MetadataStoreError(Exception):
    """Raised when metadata operations fail."""
    pass


class MetaDataStore:
    """
    In-memory document metadata store.

    Tracks document information and processing status.

    Attributes:
        metadata: Dictionary of document metadata keyed by file_id
    """

    def __init__(self) -> None:
        """
        Initialize metadata store.

        Creates empty in-memory dictionary.
        """
        self.metadata: Dict[str, Dict[str, Any]] = {}
        logger.info("MetaDataStore initialized")

    def add_document(self, metadata: Dict[str, Any]) -> None:
        """
        Register new document in metadata store.

        Called after file upload but before processing.

        Args:
            metadata: Document metadata dict with keys:
                     file_id, original_name, file_path, file_size,
                     file_type, file_hash, upload_timestamp, user_id, status

        Raises:
            MetadataStoreError: If file_id already exists or metadata invalid
        """
        try:
            if not metadata:
                raise ValueError("Metadata cannot be empty")

            file_id = metadata.get("file_id")

            if not file_id:
                raise ValueError("file_id required in metadata")

            if file_id in self.metadata:
                raise MetadataStoreError("Document already exists: %s", file_id)

            self.metadata[file_id] = metadata.copy()

            logger.info("Document registered for file_id %s with name %s)",
                        file_id,
                        metadata.get("original_name"))

        except MetadataStoreError:
            raise
        except Exception as e:
            logger.exception("Failed to add document metadata")
            raise MetadataStoreError("Add failed") from e

    def update_document(self, file_id: str, updates: Dict[str, Any]) -> None:
        """
        Update document metadata.

        Called at each processing stage to update status.

        Args:
            file_id: Document identifier
            updates: Dict with fields to update:
                    status, chunk_count, processing_timestamp, error, etc.

        Raises:
            MetadataStoreError: If document not found or update fails
        """
        try:
            if not file_id or not file_id.strip():
                raise ValueError("file_id cannot be empty")

            if not updates:
                logger.debug("update_document called with empty updates")
                return

            if file_id not in self.metadata:
                raise MetadataStoreError("Document not found: %s", file_id)

            self.metadata[file_id].update(updates)

            logger.debug("Document updated for file_id: %s with %d fields", file_id, len(updates))

        except MetadataStoreError:
            raise
        except Exception as e:
            logger.error("Failed to update document metadata")
            raise MetadataStoreError("Update failed") from e

    def get_document_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document metadata.

        Args:
            file_id: Document identifier

        Returns:
            Document metadata dict, or None if not found
        """
        try:
            if not file_id or not file_id.strip():
                logger.warning("get_document_info called with empty file_id")
                return None

            metadata = self.metadata.get(file_id)

            if metadata is None:
                logger.debug("Document not found: %s", file_id)
                return None

            return metadata.copy()

        except Exception:
            logger.exception("Failed to get document info")
            return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents.

        Returns:
            List of all document metadata dicts
            Empty list if no documents
        """
        try:
            return list(self.metadata.values())

        except Exception:
            logger.exception("Failed to list documents")
            return []

    def remove_document_from_metadata(self, file_id: str) -> bool:
        """
        Remove document from metadata store.

        Called when document is deleted.

        Args:
            file_id: Document identifier

        Returns:
            True if removed, False if not found

        Raises:
            MetadataStoreError: If removal fails unexpectedly
        """
        try:
            if not file_id or not file_id.strip():
                logger.warning("remove_document called with empty file_id")
                return False

            if file_id not in self.metadata:
                logger.debug("Document not found for removal: %s", file_id)
                return False

            # Remove metadata
            del self.metadata[file_id]

            logger.info("Document removed from metadata: %s", file_id)
            return True

        except KeyError:
            logger.debug("Document already removed: %s", file_id)
            return False
        except Exception as e:
            logger.exception("Failed to remove document")
            raise MetadataStoreError("Remove failed") from e
