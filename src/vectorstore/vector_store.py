import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embeddings, path):
        self.embeddings = embeddings
        self.path = path
        self.db = None

    def load(self) -> bool:
        try:
            index_file = os.path.join(
                self.path,
                "index.faiss"
            )

            if not os.path.exists(index_file):
                logger.info("No FAISS index found")
                return False

            self.db = FAISS.load_local(
                self.path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            logger.info(f"Loaded FAISS index from {self.path}")
            return True
        except Exception:
            logger.exception("Failed loading FAISS")
            return False

    def add_documents(self, documents: List[Document]) -> bool:
        try:
            if not documents:
                logger.warning("No documents to add to FAISS")
                return False

            if self.db is None:
                self.load()

            if self.db is not None:
                self.db.add_documents(documents)
            else:
                os.makedirs(
                    self.path,
                    exist_ok=True
                )

                self.db = FAISS.from_documents(
                    documents,
                    self.embeddings
                )

                logger.info("Created new FAISS index")

            self.db.save_local(self.path)
            logger.info(f"Saved FAISS index with {len(documents)} new documents")
            return True
        except Exception:
            logger.exception("FAISS operation failed")
            return False

    def get_vector_db_stats(self, faiss_index_path: str) -> Dict:
        try:
            if not os.path.exists(faiss_index_path):
                return {
                    'total_documents': 0,
                    'status': 'empty'
                }

            if self.db is not None:
                db = self.db
            else:
                db = self.load()

            return {
                'total_documents': len(db.index_to_docstore_id),
                'embedding_dimension': db.index.d,
                'status': 'active'
            }

        except Exception as e:
            logger.exception("Failed to get vector DB stats")
            return {'status': 'error', 'error': str(e)}

    def get_all_documents(self) -> List[Document]:
        try:
            if self.db is None:
                self.load()

            if self.db is None:
                logger.warning("FAISS database is not loaded. No documents available.")
                return []

            docs = []

            for doc_id in self.db.index_to_docstore_id.values():
                doc = self.db.docstore.search(doc_id)
                if isinstance(
                        doc,
                        Document
                ):
                    docs.append(doc)

            logger.info("Retrieved %d documents from FAISS", len(docs))
            return docs
        except Exception:
            logger.exception("Failed retrieving documents from FAISS")
            return []

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if self.db is None:
            self.load()

        if self.db is None:
            return []

        return self.db.similarity_search(query, k=k)
