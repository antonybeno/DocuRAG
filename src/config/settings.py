"""
Configuration Management Module

Centralized configuration management using environment variables with fallback defaults.
All settings can be overridden via .env file.

Environment Variables:
    DB_PATH: Path to SQLite database (default: data/db)
    UPLOAD_PATH: Path to uploaded documents (default: data/uploads)
    OLLAMA_MODEL: LLM model identifier (default: llama3:latest)
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    EMBEDDING_MODEL: HuggingFace embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
    CHUNK_SIZE: Document chunk size in characters (default: 250)
    CHUNK_OVERLAP: Overlap between chunks in characters (default: 50)
    JAEGER_ENDPOINT: Jaeger collector endpoint (default: http://jaeger:4317)
    LOG_LEVEL: Logging level (default: INFO)

Example .env:
    OLLAMA_MODEL=mistral:latest
    CHUNK_SIZE=500
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def _get_env_int(key: str, default: int, min_value: Optional[int] = None) -> int:
    """
    Get integer environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set
        min_value: Minimum acceptable value (optional)

    Returns:
        Integer value from environment or default

    Raises:
        ConfigurationError: If value is invalid
    """
    try:
        value = int(os.getenv(key, str(default)))

        if min_value is not None and value < min_value:
            raise ConfigurationError(
                f"{key} must be >= {min_value}, got {value}"
            )

        return value
    except ValueError:
        raise ConfigurationError(
            f"{key} must be an integer, got {os.getenv(key)}"
        )


# ==================== DATABASE CONFIGURATION ====================

DB_PATH: str = os.getenv("DB_PATH", "data/db")

# Create database directory if it doesn't exist
Path(DB_PATH).mkdir(parents=True, exist_ok=True)

# ==================== FILE UPLOAD CONFIGURATION ====================

UPLOAD_PATH: str = os.getenv("UPLOAD_PATH", "data/uploads")

# Create upload directory if it doesn't exist
Path(UPLOAD_PATH).mkdir(parents=True, exist_ok=True)

"""Maximum file upload size in megabytes."""
MAX_UPLOAD_SIZE_MB: int = 50

"""Supported document file types."""
ALLOWED_FILE_TYPES: tuple = ("pdf", "txt", "docx")

# ==================== LLM CONFIGURATION ====================

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:latest")

"""Ollama server base URL (Docker: http://ollama:11434)."""
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

"""LLM request timeout in seconds."""
LLM_TIMEOUT: int = 30

"""Maximum tokens in LLM response."""
LLM_MAX_TOKENS: int = 1024

# ==================== EMBEDDING CONFIGURATION ====================

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

"""Dimension of embeddings from selected model."""
EMBEDDING_DIMENSION: int = 384

# ==================== CHUNKING CONFIGURATION ====================

"""
Document chunk size in characters.
"""
CHUNK_SIZE: int = _get_env_int("CHUNK_SIZE", 250, min_value=100)

"""
Overlap between consecutive chunks in characters.
"""
CHUNK_OVERLAP: int = _get_env_int("CHUNK_OVERLAP", 50, min_value=0)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ConfigurationError(
        f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_SIZE ({CHUNK_SIZE})"
    )

# ==================== RETRIEVAL CONFIGURATION ====================

"""Number of documents to retrieve per query."""
TOP_K_RETRIEVAL: int = 5

"""Number of documents to retrieve via BM25 (keyword search)."""
BM25_K: int = 5

# ==================== VECTOR DATABASE CONFIGURATION ====================

"""Path to FAISS vector database index."""
FAISS_INDEX_PATH: str = "faiss_index"

# ==================== OBSERVABILITY CONFIGURATION ====================

JAEGER_ENDPOINT: str = os.getenv("JAEGER_ENDPOINT", "http://jaeger:4317")

"""Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

if LOG_LEVEL not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    raise ConfigurationError(f"Invalid LOG_LEVEL: {LOG_LEVEL}")

# ==================== LOGGING SETUP ====================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger.info(f"Configuration loaded from environment")
logger.debug(f"CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")
logger.debug(f"OLLAMA_MODEL={OLLAMA_MODEL}, OLLAMA_BASE_URL={OLLAMA_BASE_URL}")