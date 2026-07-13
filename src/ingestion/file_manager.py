import hashlib
import uuid
from datetime import datetime
from typing import Dict, BinaryIO
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FileManager:
    def __init__(self):
        self.upload_dir = Path("uploaded_documents")
        self.upload_dir.mkdir(exist_ok=True)

    def save_uploaded_file(
            self,
            file: BinaryIO,
            filename: str,
            user_id: str = "anonymous"
    ) -> Dict:
        try:
            file_id = str(uuid.uuid4())

            allowed_types = {
                'pdf': 'application/pdf',
                'txt': 'text/plain',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            }

            file_ext = filename.split('.')[-1].lower()
            if file_ext not in allowed_types:
                raise ValueError(f"Unsupported file type: {file_ext}")

            file_path = self.upload_dir / f"{file_id}_{filename}"
            file_content = file.read()

            file_size = len(file_content)
            if file_size > 50 * 1024 * 1024:
                raise ValueError(f"File too large: {file_size / 1024 / 1024:.2f}MB (max 50MB)")

            with open(file_path, 'wb') as f:
                f.write(file_content)

            file_hash = hashlib.md5(file_content).hexdigest()

            metadata = {
                'file_id': file_id,
                'original_name': filename,
                'file_path': file_path,
                'file_size': file_size,
                'file_type': file_ext,
                'file_hash': file_hash,
                'upload_timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'status': 'uploaded',
                'chunk_count': 0
            }

            logger.info(f"File uploaded: {file_id} - {filename} ({file_size} bytes)")

            return {
                'file_id': file_id,
                'original_name': filename,
                'file_path': file_path,
                'file_size': file_size,
                'file_type': file_ext,
                'upload_timestamp': metadata['upload_timestamp'],
                'status': 'success',
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return {
                'file_id': None,
                'original_name': filename,
                'status': 'error',
                'error': str(e)
            }
