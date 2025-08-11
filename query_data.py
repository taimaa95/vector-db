import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma   # NEW import
from langchain.prompts import PromptTemplate

chroma_path = "chroma_db"

def build_retriever():
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    # tighter retrieval; adjust as you like
    retriever = db.as_retriever(search_kwargs={"k": 4})
    return retriever

STRICT_TEMPLATE = """You are a careful RAG assistant.
Use ONLY the context to answer. If the answer isn't fully contained in the context, say "I don't know."

Question: {question}
---
Context:
{context}
---
Answer:"""

def format_context(docs):
    return "\n\n".join(d.page_content.strip() for d in docs)

def main():
    retriever = build_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")  # or your preferred model

    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        docs = retriever.get_relevant_documents(q)

        # Simple similarity gate if Chroma returns scores via metadata
        # If your store doesn't include scores, skip this block or add score_threshold with the new retriever API.
        filtered = []
        for d in docs:
            score = d.metadata.get("distance") or d.metadata.get("score")
            # keep if score is good (Chroma may use distance: smaller is better; adjust to your setup)
            keep = True
            if score is not None:
                # Example heuristic: if this is cosine distance, < 0.4 is reasonably close; tune per your data
                keep = score < 0.4
            if keep:
                filtered.append(d)

        if not filtered:
            print("Bot: I don't know.")
            continue

        context = format_context(filtered)
        prompt = PromptTemplate.from_template(STRICT_TEMPLATE).format(question=q, context=context)

        # use invoke (not predict)
        resp = llm.invoke(prompt)
        print(f"Bot: {resp.content}")

if __name__ == "__main__":
    main()
