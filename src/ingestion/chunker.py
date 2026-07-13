import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import logging

from src.ingestion.MetadataStore import MetaDataStore

logger = logging.getLogger("docurag")


class Chunker:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_document(self, texts: List[str], file_id: str, metadata: MetaDataStore) -> List[Document]:
        try:
            documents = []
            chunk_index = 0

            metadata = metadata.get_document_info(file_id)

            if not metadata:
                logger.error(f"No metadata found for file_id={file_id}")
                return []

            for text in texts:
                chunks = self.text_splitter.split_text(text)

                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "file_id": file_id,
                            "chunk_index": chunk_idx,
                            "source": metadata["original_name"],
                            "upload_time": metadata["upload_timestamp"],
                            "chunk_size": len(chunk),
                            "chunk_hash": hashlib.md5(
                                chunk.encode()
                            ).hexdigest()
                        }
                    )

                    documents.append(doc)
                    chunk_index += 1

            logger.info(f"Chunked into {len(documents)} documents (file_id: {file_id})")

            return documents

        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            return []
