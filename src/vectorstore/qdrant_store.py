"""
qdrant_store.py — Qdrant-backed vector store for the CPQ RAG system.

Modes
-----
- **Embedded** (default): data persisted to a local directory, no server needed.
  Set QDRANT_PATH in .env (default: "qdrant_data").
- **Server**: connect to a running Qdrant instance.
  Set QDRANT_URL in .env (e.g. "http://localhost:6333").
"""

import os
import json
import logging

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from src.config import (
    QDRANT_PATH,
    QDRANT_COLLECTION,
    QDRANT_URL,
    QDRANT_API_KEY,
    METADATA_PATH,
)

log = logging.getLogger("qdrant_store")


class QdrantStore:
    """
    Qdrant vector store with hybrid (dense + BM25 sparse) retrieval.
    Uses embedded mode by default (local directory, no server needed).
    """

    def __init__(
        self,
        embeddings,
        db_path: str = QDRANT_PATH,
        metadata_path: str = METADATA_PATH,
        collection_name: str = QDRANT_COLLECTION,
    ):
        self.embeddings = embeddings
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.collection_name = collection_name
        self.client: QdrantClient | None = None
        self.vectorstore: QdrantVectorStore | None = None

        meta_dir = os.path.dirname(self.metadata_path)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)

    # ── Client lifecycle ──────────────────────────────────────────────────────

    def _ensure_client(self):
        """Create the Qdrant client (embedded or remote) if not already open."""
        if self.client is not None:
            return

        if QDRANT_URL:
            log.info("Connecting to remote Qdrant at %s", QDRANT_URL)
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,
                port=443,
                https=True,
                timeout=60,
                prefer_grpc=False,
            )
        else:
            log.info("Opening embedded Qdrant store at '%s'", self.db_path)
            self.client = QdrantClient(path=self.db_path)

    def _collection_exists(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    # ── Metadata helpers ──────────────────────────────────────────────────────

    def load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {"processed_files": []}

    def save_metadata(self, metadata):
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # ── Store operations ──────────────────────────────────────────────────────

    def load_store(self):
        """Load an existing Qdrant collection (call at server startup)."""
        self._ensure_client()

        if not self._collection_exists():
            raise FileNotFoundError(
                f"Qdrant collection '{self.collection_name}' not found at "
                f"'{self.db_path}'. Run `python ingest.py` first."
            )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        self._ensure_payload_index()

        log.info(
            "Loaded Qdrant collection '%s' (%d vectors)",
            self.collection_name,
            self.client.get_collection(self.collection_name).points_count,
        )
        return self.vectorstore

    BATCH_SIZE = 100  # embed + insert in small batches to avoid memory issues

    def _add_documents_batched(self, docs):
        """Add documents in batches with progress logging."""
        total = len(docs)
        for i in range(0, total, self.BATCH_SIZE):
            batch = docs[i : i + self.BATCH_SIZE]
            self.vectorstore.add_documents(batch)
            done = min(i + self.BATCH_SIZE, total)
            log.info("  Embedded %d / %d chunks (%.0f%%)", done, total, done / total * 100)

    def create_or_update_store(self, docs):
        """
        Add documents to the Qdrant collection.
        Skips chunks whose source file is already tracked in processed_docs.json.
        """
        original_processed = set(self.load_metadata()["processed_files"])

        new_docs = []
        new_sources = set()

        for doc in docs:
            raw_source = (
                doc.metadata.get("source")
                or doc.metadata.get("file_path")
                or doc.metadata.get("file_name")
                or ""
            )
            source = os.path.basename(raw_source)
            if source not in original_processed:
                new_docs.append(doc)
                new_sources.add(source)

        self._ensure_client()

        if self._collection_exists():
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            if new_docs:
                log.info("Adding chunks from %d new file(s) to Qdrant...", len(new_sources))
                self._add_documents_batched(new_docs)
            else:
                log.info("No new chunks to add.")
        else:
            if not new_docs:
                log.warning("No documents and no existing collection — nothing to do.")
                return None

            log.info("Creating new Qdrant collection '%s'...", self.collection_name)
            sample_embedding = self.embeddings.embed_query("dimension check")
            vector_size = len(sample_embedding)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            log.info("Collection created (vector_size=%d). Starting embedding...", vector_size)

            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            self._add_documents_batched(new_docs)

        # Persist updated metadata
        all_processed = list(original_processed | new_sources)
        self.save_metadata({"processed_files": all_processed})

        log.info(
            "Qdrant collection '%s' now has %d vectors.",
            self.collection_name,
            self.client.get_collection(self.collection_name).points_count,
        )
        return self.vectorstore

    def reset_collection(self):
        """Drop and recreate the collection (used with --force)."""
        self._ensure_client()
        if self._collection_exists():
            log.info("Dropping collection '%s'...", self.collection_name)
            self.client.delete_collection(self.collection_name)

    def _ensure_payload_index(self):
        """Create a text payload index on source_doc for MatchText filtering."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.source_doc",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                ),
            )
            log.info("Text index on 'metadata.source_doc' ensured.")
        except Exception:
            pass  # index already exists

    # ── Native Qdrant filter ─────────────────────────────────────────────────

    @staticmethod
    def _pdf_filter() -> models.Filter:
        """
        Qdrant-native filter: only match documents that have a source_doc field.
        Applied DURING vector search — not after — so we always get exactly k results.
        """
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source_doc",
                    match=models.MatchText(text=".pdf"),
                )
            ]
        )

    # ── Retrievers ────────────────────────────────────────────────────────────

    def _get_all_pdf_documents(self) -> list[Document]:
        """
        Scroll through all documents in the collection that come from PDFs.
        Uses native Qdrant filtering. Needed to build the BM25 sparse index.
        """
        all_docs = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._pdf_filter(),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                all_docs.append(Document(page_content=content, metadata=metadata))

            offset = next_offset
            if offset is None:
                break

        return all_docs

    def get_retriever(self, k=10):
        """Dense-only Qdrant retriever with native PDF filter."""
        return self.vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": self._pdf_filter()}
        )

    def get_hybrid_retriever(self, k=10, dense_weight=0.6, sparse_weight=0.4):
        """
        Hybrid retriever: Qdrant (dense) + BM25 (sparse).

        - Dense: uses native Qdrant filter (only PDF sources, applied during search)
        - Sparse: BM25 built from PDF-only docs (filtered via Qdrant scroll)
        - Results merged via weighted Reciprocal Rank Fusion.
        """
        # Dense retriever (Qdrant) — native filter guarantees all k results are PDFs
        dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": self._pdf_filter()}
        )

        # Sparse retriever (BM25) — built from PDF-only docs via native scroll filter
        pdf_docs = self._get_all_pdf_documents()

        if not pdf_docs:
            log.warning("No PDF documents found in Qdrant — BM25 index will be empty.")
            return dense_retriever

        sparse_retriever = BM25Retriever.from_documents(pdf_docs)
        sparse_retriever.k = k

        # Ensemble (Hybrid)
        hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[dense_weight, sparse_weight],
        )

        log.info(
            "Hybrid retriever ready: %d PDF docs, k=%d, dense=%.1f/sparse=%.1f",
            len(pdf_docs), k, dense_weight, sparse_weight,
        )
        return hybrid_retriever
