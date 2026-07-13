import logging
from datetime import datetime
from typing import Dict, BinaryIO

logger = logging.getLogger(__name__)


class DocumentIngestionService:

    def __init__(
            self,
            file_manager,
            parser,
            chunk,
            vector_store,
            metadata
    ):
        self.file_manager = file_manager
        self.parser = parser
        self.chunk = chunk
        self.vector_store = vector_store
        self.metadata = metadata

    def process_document(
            self,
            file: BinaryIO,
            filename: str,
            user_id: str = "anonymous"
    ) -> Dict:
        logger.info(f"Starting document processing: {filename}")

        upload_result = self.file_manager.save_uploaded_file(file, filename, user_id)
        if upload_result['status'] != 'success':
            return upload_result

        self.metadata.add_document(upload_result)

        file_id = upload_result['file_id']
        file_path = upload_result['file_path']
        file_type = upload_result['file_type']

        try:
            logger.info(f"[{file_id}] Parsing document...")
            texts = self.parser.parse(file_path, file_type)
            if not texts:
                raise ValueError("Failed to extract text from document")

            logger.info(f"[{file_id}] Chunking document...")
            documents = self.chunk.chunk_document(texts, file_id, self.metadata)
            self.metadata.update_document(
                file_id,
                {
                    "chunk_count": len(documents),
                    "status": "chunked"
                }
            )
            if not documents:
                raise ValueError("Failed to chunk document")

            logger.info(f"[{file_id}] Embedding and storing in FAISS...")
            success = self.vector_store.add_documents(documents)

            if not success:
                raise ValueError("Failed to store in FAISS")

            self.metadata.update_document(
                file_id,
                {
                    "status": "processed",
                    "processing_timestamp": datetime.now().isoformat()
                }
            )

            logger.info(
                f"[{file_id}] Document processing complete: "
                f"{len(documents)} chunks, {upload_result['file_size']} bytes"
            )
            metadata = self.metadata.get_document_info(file_id)

            return {
                'file_id': file_id,
                'original_name': filename,
                'status': 'success',
                'chunks_created': len(documents),
                'file_size': upload_result['file_size'],
                'upload_timestamp': upload_result['upload_timestamp'],
                'processing_timestamp': metadata["processing_timestamp"]
            }

        except Exception as e:
            logger.exception(
                f"[{file_id}] Document processing failed"
            )
            self.metadata.update_document(
                file_id,
                {
                    "status": "failed",
                    "error": str(e)
                }
            )

            return {
                'file_id': file_id,
                'original_name': filename,
                'status': 'error',
                'error': str(e)
            }

    def delete_document(self, file_id: str):
        metadata = self.metadata.get_document(file_id)

        if not metadata:
            return False

        self.file_manager.delete_file(metadata["file_path"])
        self.metadata.remove_document_from_metadata(file_id)

        return True
