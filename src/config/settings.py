import os
from dotenv import load_dotenv

load_dotenv()


DB_PATH = os.getenv(
    "DB_PATH",
    "data/db"
)

UPLOAD_PATH = os.getenv(
    "UPLOAD_PATH",
    "data/uploads"
)


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3:latest"
)

BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        250
    )
)


CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        50
    )
)
