from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings