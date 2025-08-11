from langchain_chroma import Chroma  # modern import
from .config import CHROMA_PATH

_COLLECTION = "default"

def persist_from_documents(chunks, embeddings):
    # Auto-persist via persist_directory (no .persist() in Chroma ≥0.4)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=_COLLECTION,
    )

def load_persisted(embeddings):
    return Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=_COLLECTION,
    )
