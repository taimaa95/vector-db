# scripts/chat.py
from rag.embeddings import make_embeddings
from rag.vectorstore import load_persisted
from rag.chatloop import run_chat_loop

def main():
    print("[chat] start", flush=True)
    embeddings = make_embeddings()
    print("[chat] embeddings ready", flush=True)

    db = load_persisted(embeddings)
    print("[chat] vector store opened", flush=True)

    retriever = db.as_retriever(search_kwargs={"k": 4})
    print("[chat] retriever ready", flush=True)

    run_chat_loop(retriever)

if __name__ == "__main__":
    main()
