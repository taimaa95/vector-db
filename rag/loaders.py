from langchain_community.document_loaders import PyPDFDirectoryLoader
from .config import DATA_DIR

def load_documents():
    # Mirrors your original use of PyPDFDirectoryLoader("data")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    return loader.load()
