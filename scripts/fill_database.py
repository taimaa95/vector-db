import traceback, os
from rag.loaders import load_documents
from rag.splitters import split_documents
from rag.embeddings import make_embeddings
from rag.vectorstore import persist_from_documents

def main():
    print("[ingest] __name__:", __name__, flush=True)
    print("[ingest] cwd:", os.getcwd(), flush=True)
    docs = load_documents()
    print(f"[ingest] loaded {len(docs)} docs", flush=True)

    chunks = split_documents(docs)
    print(f"[ingest] split into {len(chunks)} chunks", flush=True)

    embeddings = make_embeddings()
    print("[ingest] embeddings ready", flush=True)

    persist_from_documents(chunks, embeddings)
    print("[ingest] index written (auto-persist)", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise