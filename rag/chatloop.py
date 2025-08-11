# rag/chatloop.py
import sys
from langchain_openai import ChatOpenAI
from .prompts import build_prompt

def make_llm():
    return ChatOpenAI()  # same defaults as before

def run_chat_loop(retriever):
    llm = make_llm()
    print("Chatbot ready. Type 'exit' to quit.", flush=True)
    while True:
        # write + flush prompt so it shows up even if stdout is buffered
        sys.stdout.write("\nYou: ")
        sys.stdout.flush()
        try:
            q = input().strip()
        except EOFError:
            print("\n[chat] EOF, exiting.", flush=True)
            break

        if q.lower() in {"exit", "quit"}:
            print("[chat] bye.", flush=True)
            break

        docs = retriever.invoke(q)        # modern retriever call
        prompt = build_prompt(q, docs)
        msg = llm.invoke(prompt)          # modern LLM call (AIMessage)
        answer = getattr(msg, "content", str(msg))
        print(f"\nBot: {answer}\n", flush=True)
