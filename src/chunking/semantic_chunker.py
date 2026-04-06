"""
semantic_chunker.py — Section-aware text chunker for CPQ PDF documents.

Strategy
--------
1. Each document (page) is first split into sections using section_parser,
   which detects markdown headings produced by LlamaParse.
2. Each section body is further split with RecursiveCharacterTextSplitter,
   which respects paragraph/sentence boundaries and never cuts mid-word.
3. Every resulting chunk gets two extra metadata fields:
     • section  — the section heading it belongs to (e.g. "Understanding Oracle CPQ Overview")
     • source_doc — the basename of the original PDF file

This metadata is what allows the prompt to cite 📄 Source: {doc} — {section}.
"""

import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.section_parser import parse_sections
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class ThresholdSematicChunker:
    """
    Drop-in replacement for the old cosine-similarity chunker.
    Name kept identical so ingest.py requires no changes.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_documents(self, docs: list) -> list:
        """
        Takes a list of LangChain Document objects (one per PDF page / block)
        and returns a flat list of smaller Document chunks, each annotated
        with `section` and `source_doc` metadata.
        """
        result = []

        for doc in docs:
            raw_source = (
                doc.metadata.get("source")
                or doc.metadata.get("file_path")
                or doc.metadata.get("file_name")
                or ""
            )
            source_doc = os.path.basename(raw_source)

            # Split page text into section blocks
            sections = parse_sections(doc.page_content)

            for section_heading, section_body in sections:
                # Further split each section body into overlapping chunks
                sub_chunks = self._splitter.split_text(section_body)

                for chunk_text in sub_chunks:
                    if not chunk_text.strip():
                        continue

                    chunk_metadata = {
                        **doc.metadata,
                        "section":    section_heading,
                        "source_doc": source_doc,
                    }

                    result.append(
                        Document(
                            page_content=chunk_text,
                            metadata=chunk_metadata,
                        )
                    )

        return result