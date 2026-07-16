"""
File Manager Module

Manages document file uploads, storage, and deletion.
"""

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, BinaryIO, Any

logger = logging.getLogger(__name__)


class FileManagerError(Exception):
    pass


class FileManager:
    """
    File upload and storage manager.

    Handles document file operations with validation and safety checks.

    Attributes:
        upload_dir: Directory for storing uploaded files
        allowed_types: Set of allowed file extensions
        max_size_bytes: Maximum file size in bytes
    """

    # Configuration
    UPLOAD_DIR = Path("uploaded_documents")

    ALLOWED_TYPES = {"pdf", "txt", "docx"}

    MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

    def __init__(self) -> None:
        """
        Initialize file manager.

        Creates upload directory if it doesn't exist.
        """
        try:
            self.upload_dir = self.UPLOAD_DIR
            self.upload_dir.mkdir(exist_ok=True, parents=True)

            logger.info("FileManager initialized: %s", self.upload_dir)

        except Exception as e:
            logger.exception("Failed to initialize FileManager")
            raise FileManagerError("Could not create upload directory") from e

    @staticmethod
    def _validate_file_type(filename: str) -> str:
        """
        Validate and extract file type from filename.

        Args:
            filename: Original filename

        Returns:
            Lowercase file extension

        Raises:
            ValueError: If file type not allowed
        """
        if "." not in filename:
            raise ValueError("Filename must have extension")

        parts = filename.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError("Invalid filename format")

        file_ext = parts[1].lower().strip()

        if not file_ext:
            raise ValueError("File extension cannot be empty")

        if file_ext not in FileManager.ALLOWED_TYPES:
            raise ValueError(
                "File type not allowed: %s. "
                "Allowed types: %s", file_ext, (', '.join(FileManager.ALLOWED_TYPES))
            )

        return file_ext

    @staticmethod
    def _validate_file_size(file_size: int) -> None:
        """
        Validate file size.

        Args:
            file_size: Size in bytes

        Raises:
            ValueError: If size exceeds limit
        """
        if file_size <= 0:
            raise ValueError("File size must be positive")

        if file_size > FileManager.MAX_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            max_mb = FileManager.MAX_SIZE_BYTES / (1024 * 1024)
            raise ValueError(
                f"File too large: {size_mb:.1f}MB (max {max_mb:.0f}MB)"
            )

    @staticmethod
    def _compute_hash(file_content: bytes) -> str:
        """
        Compute MD5 hash of file content.

        Used for:
        - Deduplication detection
        - Integrity verification
        - Caching

        Args:
            file_content: File bytes

        Returns:
            Hex digest of MD5 hash
        """
        return hashlib.md5(file_content).hexdigest()

    def save_uploaded_file(
            self,
            file: BinaryIO,
            filename: str,
            user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Save uploaded file to disk.

        Args:
            file: File object (from FastAPI UploadFile.file)
            filename: Original filename
            user_id: User identifier (for tracking)

        Returns:
            Dict with:
            - file_id: Unique identifier (UUID)
            - original_name: Original filename
            - file_path: Path to saved file
            - file_size: Size in bytes
            - file_type: File extension
            - file_hash: MD5 hash (for deduplication)
            - upload_timestamp: ISO timestamp
            - status: "success"

        Raises:
            ValueError: If validation fails
            FileManagerError: If save fails
        """
        try:
            if not filename or not filename.strip():
                raise ValueError("Filename cannot be empty")

            try:
                file_type = self._validate_file_type(filename)
            except ValueError:
                logger.exception("Invalid file type")
                raise

            try:
                file_content = file.read()
            except Exception as e:
                raise FileManagerError(f"Failed to read file: {str(e)}") from e

            file_size = len(file_content)
            try:
                self._validate_file_size(file_size)
            except ValueError:
                logger.exception("Invalid file size")
                raise

            file_id = str(uuid.uuid4())
            file_hash = self._compute_hash(file_content)
            file_path = self.upload_dir / f"{file_id}_{filename}"

            try:
                with open(file_path, 'wb') as f:
                    f.write(file_content)

                logger.info("File saved with file_id: %s, filename: %s, file_size %s bytes)",
                            file_id,
                            filename,
                            file_size
                            )

            except IOError as e:
                logger.error("Failed to write file to disk")
                raise FileManagerError("Failed to save file") from e
            except Exception as e:
                logger.error("Unexpected error saving file")
                raise FileManagerError("Save failed") from e

            metadata = {
                'file_id': file_id,
                'original_name': filename,
                'file_path': str(file_path),
                'file_size': file_size,
                'file_type': file_type,
                'file_hash': file_hash,
                'upload_timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'status': 'uploaded'
            }

            return {
                'file_id': file_id,
                'original_name': filename,
                'file_path': str(file_path),
                'file_size': file_size,
                'file_type': file_type,
                'file_hash': file_hash,
                'upload_timestamp': metadata['upload_timestamp'],
                'status': 'success',
                'metadata': metadata
            }

        except (ValueError, FileManagerError):
            raise
        except Exception as e:
            logger.exception("Unexpected error in save_uploaded_file")
            raise FileManagerError("File save failed") from e

    @staticmethod
    def delete_file(file_path: str) -> bool:
        """
        Delete file from disk.

        Args:
            file_path: Path to file (from save_uploaded_file response)

        Returns:
            True if deleted, False if not found

        Raises:
            FileManagerError: If deletion fails unexpectedly
        """
        try:
            if not file_path or not file_path.strip():
                logger.warning("delete_file called with empty path")
                return False

            path = Path(file_path)

            if not path.exists():
                logger.debug("File not found: %s", file_path)
                return False

            try:
                path.unlink()
                logger.info("File deleted: %s", file_path)
                return True

            except FileNotFoundError:
                logger.debug("File already deleted: %s", file_path)
                return False
            except PermissionError as e:
                raise FileManagerError("Permission denied deleting file %s: {str(e)}", file_path) from e
            except Exception as e:
                raise FileManagerError("Failed to delete file") from e

        except FileManagerError:
            raise
        except Exception as e:
            logger.exception("Unexpected error deleting file")
            raise FileManagerError("Deletion failed") from e
