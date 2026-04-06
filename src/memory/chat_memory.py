"""
chat_memory.py — Session-based chat history with Redis or in-memory backend.

If REDIS_URL is set and Redis is reachable, sessions are stored in Redis with
automatic TTL expiry. This enables multi-worker deployments and survives
server restarts.

Otherwise, falls back to the in-memory store with TTL eviction, LRU cap, and
per-session message limit (same as Phase 1).
"""

import time
import logging
from collections import OrderedDict

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.config import (
    REDIS_URL,
    SESSION_TTL_SECONDS,
    MAX_SESSIONS,
    MAX_HISTORY_MESSAGES,
)

log = logging.getLogger("chat_memory")


class ChatMemoryManager:

    def __init__(self) -> None:
        self._use_redis = False
        self._redis_url = REDIS_URL

        # Try connecting to Redis if URL is provided
        if self._redis_url:
            try:
                import redis as redis_lib
                client = redis_lib.from_url(self._redis_url)
                client.ping()
                client.close()
                self._use_redis = True
                log.info("Chat memory: Redis connected at %s", self._redis_url)
            except Exception as exc:
                log.warning(
                    "Redis unavailable (%s) — falling back to in-memory store.", exc,
                )

        # In-memory fallback state
        if not self._use_redis:
            log.info(
                "Chat memory: in-memory store (TTL=%ds, max=%d sessions, max=%d msgs)",
                SESSION_TTL_SECONDS, MAX_SESSIONS, MAX_HISTORY_MESSAGES,
            )
            self._store: OrderedDict[str, ChatMessageHistory] = OrderedDict()
            self._timestamps: dict[str, float] = {}

    # ── Redis backend ─────────────────────────────────────────────────────────

    def _get_redis_history(self, session_id: str) -> ChatMessageHistory:
        from langchain_community.chat_message_histories import RedisChatMessageHistory

        history = RedisChatMessageHistory(
            session_id=session_id,
            url=self._redis_url,
            ttl=SESSION_TTL_SECONDS,
        )
        # Trim if over max messages
        if len(history.messages) > MAX_HISTORY_MESSAGES:
            keep = history.messages[-MAX_HISTORY_MESSAGES:]
            history.clear()
            for msg in keep:
                history.add_message(msg)
        return history

    # ── In-memory backend (with TTL / LRU / trim) ────────────────────────────

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, ts in self._timestamps.items()
            if now - ts > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._store[sid]
            del self._timestamps[sid]
        if expired:
            log.info("Evicted %d expired session(s).", len(expired))

    def _evict_lru(self) -> None:
        while len(self._store) > MAX_SESSIONS:
            oldest_sid, _ = self._store.popitem(last=False)
            self._timestamps.pop(oldest_sid, None)
            log.info("Evicted LRU session: %s", oldest_sid)

    def _trim_history(self, history: ChatMessageHistory) -> None:
        if len(history.messages) > MAX_HISTORY_MESSAGES:
            history.messages = history.messages[-MAX_HISTORY_MESSAGES:]

    def _get_memory_history(self, session_id: str) -> ChatMessageHistory:
        self._evict_expired()

        if session_id not in self._store:
            self._evict_lru()
            self._store[session_id] = ChatMessageHistory()

        self._store.move_to_end(session_id)
        self._timestamps[session_id] = time.time()

        history = self._store[session_id]
        self._trim_history(history)
        return history

    # ── Public API ────────────────────────────────────────────────────────────

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if self._use_redis:
            return self._get_redis_history(session_id)
        return self._get_memory_history(session_id)

    @property
    def active_session_count(self) -> int:
        if self._use_redis:
            return -1  # not cheaply available from Redis
        return len(self._store)

    def wrap_chain(self, chain: Runnable) -> RunnableWithMessageHistory:
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
