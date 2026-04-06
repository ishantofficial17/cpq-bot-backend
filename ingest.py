"""
ingest.py — Offline ingestion pipeline for the CPQ RAG system.
"""

import os
import sys
import time
import argparse
import logging

from src.config import PDF_DATA_PATH, LLAMA_API_KEY
from src.loaders.pdf_loader import LlamaParseLoader
from src.chunking.semantic_chunker import ThresholdSematicChunker
from src.embeddings.embedding_model import load_embeddings
from src.vectorstore.qdrant_store import QdrantStore

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CPQ RAG ingestion pipeline")
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Specific PDF filenames in data/pdf/ to ingest (default: all new)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if already in the vector store",
    )
    return parser.parse_args()


def get_pdfs_to_ingest(vector_db: QdrantStore, target_files=None, force=False):
    """Return list of PDF filenames (including subfolders) that need to be ingested."""

    all_pdfs = []

    # ✅ FIX: recursive scan of subfolders
    for root, dirs, files in os.walk(PDF_DATA_PATH):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PDF_DATA_PATH)
                all_pdfs.append(rel_path)

    all_pdfs = sorted(all_pdfs)

    # Debug log (optional but useful)
    log.info("Detected %d PDF(s) (including subfolders)", len(all_pdfs))
    for f in all_pdfs:
        log.info("  • %s", f)

    if not all_pdfs:
        log.error("No PDF files found in '%s'. Exiting.", PDF_DATA_PATH)
        sys.exit(1)

    if target_files:
        missing = [
            f for f in target_files
            if not os.path.exists(os.path.join(PDF_DATA_PATH, f))
        ]
        if missing:
            log.error("These files were not found in '%s': %s", PDF_DATA_PATH, missing)
            sys.exit(1)
        pdfs_to_process = target_files
    else:
        pdfs_to_process = all_pdfs

    if force:
        return pdfs_to_process

    metadata        = vector_db.load_metadata()
    already_indexed = set(metadata.get("processed_files", []))
    new_pdfs        = [f for f in pdfs_to_process if f not in already_indexed]

    skipped = len(pdfs_to_process) - len(new_pdfs)
    if skipped:
        log.info("Skipping %d already-indexed file(s). Use --force to re-ingest.", skipped)

    return new_pdfs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    total_start = time.time()

    log.info("=" * 60)
    log.info("CPQ RAG — Offline Ingestion Pipeline")
    log.info("=" * 60)

    # ── Step 1: Embeddings & vector store ──────────────────────────────────────
    log.info("Loading embedding model...")
    embeddings = load_embeddings()
    vector_db  = QdrantStore(embeddings)

    # ── Step 2: Determine PDFs ─────────────────────────────────────────────────
    pdfs_to_ingest = get_pdfs_to_ingest(vector_db, args.files, args.force)

    if not pdfs_to_ingest:
        log.info("Nothing to ingest — all PDFs are already indexed. ✓")
        log.info("Start the server with:  uvicorn app:app --reload")
        return

    log.info("PDFs to ingest  : %d", len(pdfs_to_ingest))

    # ── Step 3: Parse ──────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Step 1/3  Parsing PDFs with LlamaParse...")
    parse_start = time.time()

    loader = LlamaParseLoader(target_files=pdfs_to_ingest)
    docs   = loader.load_documents()

    log.info("Parsed %d document chunks in %.1fs", len(docs), time.time() - parse_start)

    if not docs:
        log.error("No documents returned from parser.")
        sys.exit(1)

    # ── Step 4: Chunk ──────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Step 2/3  Semantic chunking...")
    chunk_start = time.time()

    chunker = ThresholdSematicChunker()
    chunks  = chunker.split_documents(docs)

    log.info("Produced %d chunks in %.1fs", len(chunks), time.time() - chunk_start)

    # ── Step 5: Embed & Save ───────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("Step 3/3  Embedding and saving to Qdrant...")

    if args.force:
        log.info("--force mode: resetting collection and metadata...")
        vector_db.reset_collection()
        vector_db.save_metadata({"processed_files": []})

    vector_db.create_or_update_store(chunks)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    log.info("=" * 60)
    log.info("✅ Ingestion complete!")
    log.info("   Files ingested : %d", len(pdfs_to_ingest))
    log.info("   Chunks stored  : %d", len(chunks))
    log.info("   Total time     : %.1fs", total_elapsed)
    log.info("=" * 60)
    log.info("Start the server with:  uvicorn app:app --reload")


if __name__ == "__main__":
    main()