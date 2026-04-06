from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class LLMReranker:
    """
    CrossEncoder-based reranker — deterministic, local, no LLM API calls.
    Uses ms-marco-MiniLM-L-6-v2 which is purpose-built for passage reranking.
    """

    def __init__(self, llm: object = None) -> None:
        # llm arg kept for API compatibility with app.py — not used here
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:

        if not docs:
            return docs

        # Score each (query, passage) pair
        pairs  = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        # Sort by score descending (highest relevance first)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        return [doc for _, doc in ranked]