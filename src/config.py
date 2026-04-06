import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
LLM_MODEL       = os.getenv("LLM_MODEL", "groq:llama-3.1-8b-instant")

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_DATA_PATH   = os.getenv("PDF_DATA_PATH", "data/pdf")
METADATA_PATH   = os.getenv("METADATA_PATH", "metadata/processed_docs.json")

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_PATH       = os.getenv("QDRANT_PATH", "qdrant_data")         # local embedded storage
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cpq_documents")
QDRANT_URL        = os.getenv("QDRANT_URL", "")                     # set for remote Qdrant server
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")                 # required for Qdrant Cloud

# ── Redis (optional — enables persistent, multi-worker chat memory) ───────────
REDIS_URL = os.getenv("REDIS_URL", "")  # e.g. redis://localhost:6379/0

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVER_K     = int(os.getenv("RETRIEVER_K", "10"))
DENSE_WEIGHT    = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT   = float(os.getenv("SPARSE_WEIGHT", "0.4"))
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "8"))

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Security & Middleware ─────────────────────────────────────────────────────
CORS_ORIGINS    = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
RATE_LIMIT      = os.getenv("RATE_LIMIT", "20/minute")
API_KEY         = os.getenv("RAG_API_KEY", "")  # empty = no auth required (dev mode)

# ── Chat Memory ───────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS  = int(os.getenv("SESSION_TTL_SECONDS", "86400"))   # 24 hours
MAX_SESSIONS         = int(os.getenv("MAX_SESSIONS", "1000"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "50"))

# ── Logging / Observability ──────────────────────────────────────────────────
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")    # "text" for dev, "json" for prod
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO")
