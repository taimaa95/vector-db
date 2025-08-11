import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables (expects .env with OPENAI_API_KEY)
load_dotenv()

# Paths for data and ChromaDB storage
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"


def ingest_documents(data_path: str = DATA_PATH, chroma_path: str = CHROMA_PATH) -> Chroma:
    """
    Ingest PDFs from data_path, chunk them, and persist a Chroma vector store.
    """
    # Remove existing vector store
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Load all PDF documents
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No documents found in '{data_path}'")

    # Split into overlapping text chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} docs → {len(chunks)} chunks.")

    # Build and persist ChromaDB
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=chroma_path
    )
    db.persist()  # Save the vector store from RAM to disk
    print(f"Saved {len(chunks)} chunks to Chroma at '{chroma_path}'")
    return db 

if __name__ == "__main__":
    ingest_documents()