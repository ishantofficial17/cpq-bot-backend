import os

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document

from src.config import LLAMA_API_KEY, PDF_DATA_PATH


class LlamaParseLoader:

    def __init__(self, target_files=None):
        """
        target_files: optional list of filenames or paths
        """
        self.target_files = target_files

        self.parser = LlamaParse(
            api_key=LLAMA_API_KEY,
            result_type="markdown"
        )

    def load_documents(self):

        # ✅ Case: explicitly empty list
        if self.target_files is not None and len(self.target_files) == 0:
            return []

        # 🔥 CASE 1: specific files provided
        if self.target_files:
            input_files = []

            for f in self.target_files:

                # ✅ FIX: avoid duplicate "data/pdf"
                if f.startswith(PDF_DATA_PATH):
                    full_path = f
                else:
                    full_path = os.path.join(PDF_DATA_PATH, f)

                # ✅ Normalize path (fix slashes issues)
                full_path = os.path.normpath(full_path)

                if os.path.exists(full_path):
                    input_files.append(full_path)
                else:
                    print(f"⚠️ File not found: {full_path}")

            # ❌ If nothing valid → stop early
            if not input_files:
                raise ValueError("❌ No valid PDF files found for ingestion")

            print(f"📄 Loading {len(input_files)} PDF files")

            reader = SimpleDirectoryReader(
                input_files=input_files,
                file_extractor={".pdf": self.parser}
            )

        # 🔥 CASE 2: no specific files → load entire directory (with subfolders)
        else:
            print(f"📂 Loading all PDFs from: {PDF_DATA_PATH}")

            reader = SimpleDirectoryReader(
                input_dir=PDF_DATA_PATH,
                recursive=True,  # ✅ VERY IMPORTANT for subfolders
                file_extractor={".pdf": self.parser}
            )

        # 🔄 Load raw docs
        docs = reader.load_data()

        documents = []

        # 🔁 Convert to LangChain Document format
        for doc in docs:
            documents.append(
                Document(
                    page_content=doc.text,
                    metadata=doc.metadata
                )
            )

        return documents