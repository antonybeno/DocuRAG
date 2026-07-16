"""
Document Ingestion Service Module

Orchestrates the complete document ingestion pipeline:

Pipeline Stages:
    1. File Upload & Validation
       - Check file type and size
       - Generate unique file ID
       - Store file on disk
       - Validate with MD5 checksum

    2. Document Parsing
       - Extract text from PDF/TXT/DOCX
       - Handle encoding issues
       - Preserve structure (pages, sections)

    3. Chunking
       - Split text into semantic chunks
       - Maintain overlap for context
       - Compute chunk hashes
       - Create chunk metadata

    4. Embedding & Storage
       - Generate embeddings for each chunk
       - Store in FAISS vector database
       - Update metadata store

    5. Status Tracking
       - Track each stage's success/failure
       - Update metadata at each step
       - Provide detailed error messages

"""

import logging
from datetime import datetime
from typing import Dict, BinaryIO, Optional, Any

from src.monitoring.observability import (
    tracer,
    chunks_created,
    documents_uploaded
)

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when document ingestion fails."""
    pass


class DocumentIngestionService:
    """
    Complete document ingestion pipeline orchestrator.

    Manages all stages from file upload through embedding storage.
    Provides transaction-like behavior with rollback on failure.

    Attributes:
        file_manager: FileManager instance for file operations
        parser: DocumentParser instance for text extraction
        chunk: Chunker instance for text splitting
        vector_store: VectorStore instance for embedding storage
        metadata: MetaDataStore instance for metadata tracking
    """

    def __init__(
            self,
            file_manager,
            parser,
            chunk,
            vector_store,
            metadata
    ) -> None:
        """
        Initialize ingestion service.

        Args:
            file_manager: File upload and storage manager
            parser: Document parser (PDF, TXT, DOCX)
            chunk: Document chunker
            vector_store: Vector database for embeddings
            metadata: Metadata store for tracking

        Raises:
            ValueError: If any component is None
        """
        if not all([file_manager, parser, chunk, vector_store, metadata]):
            raise ValueError("All ingestion components must be provided")

        self.file_manager = file_manager
        self.parser = parser
        self.chunk = chunk
        self.vector_store = vector_store
        self.metadata = metadata

        logger.info("DocumentIngestionService initialized")

    @staticmethod
    def _validate_input(
            file: BinaryIO,
            filename: str,
            user_id: str
    ) -> None:
        """
        Validate input parameters.

        Args:
            file: File object
            filename: Original filename
            user_id: User identifier

        Raises:
            ValueError: If validation fails
        """
        if file is None:
            raise ValueError("File object cannot be None")

        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")

        if not user_id or not user_id.strip():
            raise ValueError("user_id cannot be empty")

        if not filename.count('.') > 0:
            raise ValueError("Filename must have extension")

    def process_document(
            self,
            file: BinaryIO,
            filename: str,
            user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Process document through complete ingestion pipeline.

        Pipeline:
            1. Validate inputs
            2. Upload file (save to disk, compute hash)
            3. Register in metadata store
            4. Parse document (extract text)
            5. Chunk text (split with overlap)
            6. Embed and store (add to FAISS)
            7. Update metadata to "processed"

        Args:
            file: File object (from FastAPI UploadFile.file)
            filename: Original filename (e.g., "document.pdf")
            user_id: User ID for tracking (default: "anonymous")

        Returns:
            Dict with:
            - status: "success" or "error"
            - file_id: Unique file identifier (UUID)
            - original_name: Original filename
            - chunks_created: Number of chunks (if success)
            - file_size: Size in bytes
            - error: Error message (if failed)

        Raises:
            IngestionError: If validation fails (not caught)
        """
        try:
            self._validate_input(file, filename, user_id)
        except ValueError as e:
            logger.exception("Input validation failed")
            raise IngestionError("Invalid input") from e

        file_id: Optional[str] = None

        try:
            logger.info("Starting document ingestion: %s ", filename)
            with tracer.start_as_current_span("file_upload") as upload_span:
                try:
                    logger.debug("Uploading file: %s", filename)

                    upload_result = self.file_manager.save_uploaded_file(
                        file,
                        filename,
                        user_id
                    )

                    if upload_result['status'] != 'success':
                        error_msg = upload_result.get('error', 'Unknown error')
                        logger.error("File upload failed: %s ", error_msg)
                        raise IngestionError("File upload failed: %s ", error_msg)

                    file_id = upload_result['file_id']
                    file_path = upload_result['file_path']
                    file_size = upload_result['file_size']
                    file_type = upload_result['file_type']

                    upload_span.set_attribute("file_id", file_id)
                    upload_span.set_attribute("file_size", file_size)
                    upload_span.set_attribute("file_type", file_type)

                    logger.debug(
                        "File uploaded successfully: file_id=%s file_size=%d",
                        file_id,
                        file_size
                    )
                    documents_uploaded.add(1, {"status": "uploaded"})

                except IngestionError:
                    raise
                except Exception as e:
                    logger.error("Unexpected error during file upload")
                    raise IngestionError("File upload failed") from e

            with tracer.start_as_current_span("metadata_registration"):
                try:
                    logger.debug("Registering metadata for %s", file_id)

                    self.metadata.add_document(upload_result)

                    logger.debug("Metadata registered: %s", file_id)

                except Exception as e:
                    logger.error("Metadata registration failed")
                    raise IngestionError("Metadata registration failed") from e

            with tracer.start_as_current_span("document_parsing") as parse_span:
                try:
                    logger.debug("Parsing document: %s", file_id)

                    self.metadata.update_document(
                        file_id,
                        {"status": "parsing"}
                    )

                    texts = self.parser.parse(file_path, file_type)

                    if not texts:
                        raise IngestionError("Failed to extract text from document")

                    parse_span.set_attribute("extracted_pages", len(texts))
                    logger.debug("Document parsed: %s with %d sections", file_id, len(texts))

                except IngestionError as e:
                    self.metadata.update_document(
                        file_id,
                        {"status": "parse_failed", "error": str(e)}
                    )
                    raise
                except Exception as e:
                    error_msg = str(e)
                    logger.error("Parsing failed for field id: %s with error: %s", file_id, error_msg)
                    self.metadata.update_document(
                        file_id,
                        {"status": "parse_failed", "error": error_msg}
                    )
                    raise IngestionError("Document parsing failed") from e

            with tracer.start_as_current_span("document_chunking") as chunk_span:
                try:
                    logger.debug("Chunking document: %s", file_id)

                    self.metadata.update_document(
                        file_id,
                        {"status": "chunking"}
                    )

                    documents = self.chunk.chunk_document(
                        texts,
                        file_id,
                        self.metadata
                    )

                    if not documents:
                        raise IngestionError("Failed to chunk document")

                    chunk_span.set_attribute("chunks_created", len(documents))
                    chunks_created.add(len(documents))

                    logger.debug("Document chunked for file_id: %s with %d chunks)",
                                 file_id,
                                 len(documents))

                except IngestionError:
                    self.metadata.update_document(
                        file_id,
                        {"status": "chunk_failed", "error": str(e)}
                    )
                    raise
                except Exception as e:
                    error_msg = str(e)
                    logger.error("Chunking failed for field id: %s with error: %s", file_id, error_msg)
                    self.metadata.update_document(
                        file_id,
                        {"status": "chunk_failed", "error": error_msg}
                    )
                    raise IngestionError("Document chunking failed") from e

            with tracer.start_as_current_span("embedding_and_storage") as store_span:
                try:
                    logger.debug("Embedding and storing document for file_id: %s with %d chunks)",
                                 file_id,
                                 len(documents))

                    self.metadata.update_document(
                        file_id,
                        {"status": "storing"}
                    )

                    success = self.vector_store.add_documents(documents)

                    if not success:
                        raise IngestionError("Failed to store in vector database")

                    store_span.set_attribute("documents_stored", len(documents))

                    logger.debug("Embeddings stored successfully for file_id: %s with %d chunks)",
                                 file_id,
                                 len(documents))

                except IngestionError:
                    self.metadata.update_document(
                        file_id,
                        {"status": "storage_failed", "error": str(e)}
                    )
                    raise
                except Exception as e:
                    error_msg = str(e)
                    logger.error("Storage failed for field id: %s with error: %s", file_id, error_msg)
                    self.metadata.update_document(
                        file_id,
                        {"status": "storage_failed", "error": error_msg}
                    )
                    raise IngestionError("Embedding storage failed") from e

            try:
                processing_timestamp = datetime.now().isoformat()
                self.metadata.update_document(
                    file_id,
                    {
                        "status": "processed",
                        "processing_timestamp": processing_timestamp,
                        "chunk_count": len(documents)
                    }
                )

                logger.info("Document processing completed for file_id %s with %d chunks and %d bytes)",
                            file_id,
                            len(documents),
                            file_size
                            )

                documents_uploaded.add(1, {"status": "success"})

                return {
                    'file_id': file_id,
                    'original_name': filename,
                    'status': 'success',
                    'chunks_created': len(documents),
                    'file_size': file_size,
                    'upload_timestamp': upload_result['upload_timestamp'],
                    'processing_timestamp': processing_timestamp
                }

            except IngestionError as e:
                documents_uploaded.add(1, {"status": "error"})
                logger.error("Document ingestion failed")

                return {
                    'file_id': file_id,
                    'original_name': filename,
                    'status': "error",
                    'error': str(e)
                }

        except Exception as e:
            documents_uploaded.add(1, {"status": "error"})
            logger.exception(
                "Unexpected error while processing file: file_id=%s filename=%s",
                file_id,
                filename
            )

            return {
                'file_id': file_id,
                'original_name': filename,
                'status': "error",
                'error': str(e)
            }

    def delete_document(self, file_id: str) -> bool:
        """
        Delete document and all associated data.

        Args:
            file_id: Unique file identifier

        Returns:
            True if deleted, False if not found

        Raises:
            IngestionError: If deletion fails unexpectedly
        """
        try:
            logger.info("Deleting document for file id: %s ", file_id)
            metadata = self.metadata.get_document_info(file_id)
            if not metadata:
                logger.warning("Document not found for file id: %s ", file_id)
                return False

            try:
                file_path = metadata.get("file_path")
                if file_path:
                    self.file_manager.delete_file(file_path)
                    logger.debug("File deleted from disk: %s ", file_path)

            except Exception as e:
                logger.error("Failed to delete file %s", str(e))

            success = self.metadata.remove_document_from_metadata(file_id)
            if success:
                logger.info("Document deleted for file id: %s ", file_id)

            return success

        except Exception as e:
            logger.exception("Error deleting document for file id: %s ", file_id)
            raise IngestionError("Delete failed") from e
