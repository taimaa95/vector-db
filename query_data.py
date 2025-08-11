import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
You are a helpful assistant specializing in smoking cessation.

Context (from our knowledge base):
{context}

Recent conversation:
{history}

---
Question: {question}

Answer:
"""

def chat_loop(chroma_path: str = CHROMA_PATH):
    db  = Chroma(persist_directory=chroma_path, embedding_function=OpenAIEmbeddings())
    llm = ChatOpenAI()
    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=PROMPT_TEMPLATE
    )

    # initialize memory with a single system message
    conversation_history = [
        {"role": "system", "content": "You are a smoking-cessation coach."}
    ]

    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ("exit","quit"):
            print("Goodbye! 👋")
            break

        # 1) save user turn
        conversation_history.append({"role":"user","content":user_input})

        # 2) retrieve top-3 docs
        docs = db.similarity_search(user_input, k=3)
        context = "\n\n".join(d.page_content for d in docs)

        # 3) build history snippet (last 10, excluding system)
        window = conversation_history[-10:]
        history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in window)

        # 4) fill in the prompt
        full_prompt = prompt.format(
            context=context or "—no relevant context—",
            history=history_text,
            question=user_input
        )

        # 5) get an answer
        answer = llm.predict(full_prompt)
        print(f"\nBot: {answer}")

        # 6) save assistant turn & truncate
        conversation_history.append({"role":"assistant","content":answer})
        if len(conversation_history) > 11:
            conversation_history = [conversation_history[0]] + conversation_history[-10:]

if __name__ == "__main__":
    chat_loop()
