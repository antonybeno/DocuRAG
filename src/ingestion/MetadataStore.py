from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetaDataStore:
    def __init__(self):
        self.metadata = {}

    def add_document(self, metadata: Dict) -> None:
        file_id = metadata["file_id"]
        self.metadata[file_id] = metadata

        logger.info(f"Metadata stored: {file_id}")

    def update_document(
            self,
            file_id: str,
            values: Dict
    ):
        self.metadata[file_id].update(values)

    def list_documents(self) -> List[Dict]:
        return list(self.metadata.values())

    def get_document_info(self, file_id: str) -> Optional[Dict]:
        return self.metadata.get(file_id)

    def remove_document_from_metadata(
            self,
            file_id: str
    ) -> bool:
        try:
            if file_id not in self.metadata:
                logger.warning(f"Metadata not found: {file_id}")
                return False

            del self.metadata[file_id]

            logger.info(f"Metadata removed: {file_id}")

            return True
        except Exception as e:
            logger.exception(f"Failed removing metadata: {e}")
            return False
