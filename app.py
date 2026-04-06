"""
app.py — Production FastAPI server for the CPQ RAG chatbot.

Prerequisites
─────────────
Run `python ingest.py` first to build the Qdrant vector store.
The server loads the pre-built index on startup — it never parses PDFs.

Start the server
────────────────
  # Development (auto-reload)
  uvicorn app:app --reload

  # Production
  gunicorn app:app -k uvicorn.workers.UvicornWorker --workers 1 --bind 0.0.0.0:8000
"""

import sys
import time
import logging

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from langchain.chat_models import init_chat_model

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.logging_config import setup_logging
from src.middleware import RequestTracingMiddleware
from src.embeddings.embedding_model import load_embeddings
from src.vectorstore.qdrant_store import QdrantStore
from src.reranking.llm_reranker import LLMReranker
from src.rag.conversational_rag import ConversationalRAG
from src.memory.chat_memory import ChatMemoryManager
from src.config import (
    LLM_MODEL, CORS_ORIGINS, RATE_LIMIT, API_KEY,
    RETRIEVER_K, DENSE_WEIGHT, SPARSE_WEIGHT,
)

# ── Logging (call ONCE before anything else) ──────────────────────────────────
setup_logging()
log = logging.getLogger("app")

# ── Rate limiter ──────────────────────────────────────────────────────────────

def _rate_limit_key(request: Request) -> str:
    """Composite rate-limit key: IP + session_id (from JSON body, if available)."""
    ip = get_remote_address(request)
    # Try to extract session_id from a cached body parse; fall back to IP-only
    session_id = getattr(request.state, "_rate_limit_session_id", None)
    if session_id:
        return f"{ip}:{session_id}"
    return ip

limiter = Limiter(key_func=get_remote_address)

# ── API-key authentication (for /chat) ────────────────────────────────────────
_bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """
    If API_KEY is configured, require a matching Bearer token.
    When API_KEY is empty (dev mode), authentication is skipped.
    """
    if not API_KEY:
        return  # dev mode — no auth required
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ── Singleton pipeline state ───────────────────────────────────────────────────
_conversational_chain = None
_memory_manager: ChatMemoryManager | None = None
_vector_db: QdrantStore | None = None


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conversational_chain, _memory_manager, _vector_db

    log.info("=" * 60)
    log.info("CPQ RAG Server — Starting up")
    log.info("=" * 60)

    # ── Embeddings ─────────────────────────────────────────────────────────────
    log.info("Loading embedding model...")
    embeddings = load_embeddings()

    # ── Vector store ───────────────────────────────────────────────────────────
    _vector_db = QdrantStore(embeddings)

    try:
        log.info("Loading Qdrant vector store...")
        _vector_db.load_store()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    retriever = _vector_db.get_hybrid_retriever(
        k=RETRIEVER_K, dense_weight=DENSE_WEIGHT, sparse_weight=SPARSE_WEIGHT,
    )

    # ── LLM ────────────────────────────────────────────────────────────────────
    log.info("Loading LLM  (%s)...", LLM_MODEL)
    llm = init_chat_model(LLM_MODEL)

    # ── Reranker ───────────────────────────────────────────────────────────────
    log.info("Initialising reranker...")
    reranker = LLMReranker(llm)

    # ── RAG chain ──────────────────────────────────────────────────────────────
    log.info("Building conversational RAG chain...")
    rag_builder = ConversationalRAG(retriever=retriever, reranker=reranker, llm=llm)
    rag_chain   = rag_builder.build_chain()

    # ── Memory ─────────────────────────────────────────────────────────────────
    _memory_manager       = ChatMemoryManager()
    _conversational_chain = _memory_manager.wrap_chain(rag_chain)

    log.info("=" * 60)
    log.info("Server ready — CPQ RAG is live")
    log.info("=" * 60)

    yield

    log.info("Shutting down CPQ RAG server.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CPQ RAG Chatbot",
    description="Oracle CPQ conversational AI powered by a RAG pipeline.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware (order matters: last added = first executed) ────────────────────
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
    )

app.add_middleware(RequestTracingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mask_session_id(session_id: str) -> str:
    """Return a masked version of a session ID safe for logging."""
    if len(session_id) <= 4:
        return "***"
    return session_id[:4] + "..."


# ── Request / Response schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Alphanumeric session identifier (hyphens and underscores allowed).",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message (max 2000 characters).",
    )

    @field_validator("message")
    @classmethod
    def message_must_be_printable(cls, v: str) -> str:
        if "\x00" in v:
            raise ValueError("Null bytes are not allowed.")
        return v.strip()


class ChatResponse(BaseModel):
    session_id: str
    answer: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health():
    """
    Deep health check — verifies that all critical components are functional.
    Returns 200 only when the pipeline is fully operational.
    """
    components = {}

    # Check pipeline
    if _conversational_chain is None:
        return {"status": "unavailable", "reason": "Pipeline not initialised"}

    components["pipeline"] = "ok"

    # Check Qdrant
    try:
        if _vector_db and _vector_db.client:
            info = _vector_db.client.get_collection(_vector_db.collection_name)
            components["qdrant"] = f"ok ({info.points_count} vectors)"
        else:
            components["qdrant"] = "not loaded"
    except Exception as exc:
        components["qdrant"] = f"error: {exc}"

    # Check memory manager
    if _memory_manager is not None:
        backend = "redis" if _memory_manager._use_redis else "in-memory"
        components["memory"] = f"ok ({backend})"
    else:
        components["memory"] = "not loaded"

    all_ok = all("ok" in str(v) for v in components.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "components": components,
    }


@app.get("/readiness", tags=["ops"])
def readiness():
    """Readiness probe — returns 200 only when the server can accept traffic."""
    if _conversational_chain is None:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"ready": True}


@app.post("/chat", response_model=ChatResponse, tags=["chat"], dependencies=[Depends(verify_api_key)])
@limiter.limit(RATE_LIMIT, key_func=_rate_limit_key)
def chat(request: Request, body: ChatRequest):
    """
    Send a message to the CPQ assistant.

    - `session_id` — unique identifier for a conversation.
    - `message`    — the user's question or message.

    Returns the assistant's answer and the session_id.
    Response header `X-Request-ID` can be used for debugging.
    """
    # Stash session_id for the composite rate-limit key function
    request.state._rate_limit_session_id = body.session_id

    request_id = getattr(request.state, "request_id", "?")
    masked_sid = _mask_session_id(body.session_id)

    if _conversational_chain is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet.")

    log.info(
        "Chat request: session=%s, msg_len=%d",
        masked_sid, len(body.message),
        extra={"request_id": request_id, "session_id": masked_sid},
    )

    start = time.perf_counter()

    try:
        answer = _conversational_chain.invoke(
            {"input": body.message},
            config={"configurable": {"session_id": body.session_id}},
        )
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000)
        log.exception(
            "Chat request FAILED after %dms: %s",
            duration_ms, exc,
            extra={
                "request_id": request_id,
                "session_id": masked_sid,
                "duration_ms": duration_ms,
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")

    duration_ms = round((time.perf_counter() - start) * 1000)
    log.info(
        "Chat response: session=%s, answer_len=%d, duration=%dms",
        masked_sid, len(answer), duration_ms,
        extra={
            "request_id": request_id,
            "session_id": masked_sid,
            "duration_ms": duration_ms,
        },
    )

    return ChatResponse(session_id=body.session_id, answer=answer)


# ── Local dev entry-point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
