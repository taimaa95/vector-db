# rag/prompts.py

import os

PROMPT_TEMPLATE = """
You are a helpful assistant specializing in smoking cessation. Do not make things up, if you dont know the answer say "idk bro".

Context (from our knowledge base):
{context}

Recent conversation:
{history}

---
Question: {question}

Answer:
"""

def _fmt(d, i):
    # Nice-to-have: show source + page for your own debugging (optional)
    src = os.path.basename(d.metadata.get("source", "") or "")
    page = d.metadata.get("page")
    tag = f"[{i+1}]"
    meta = f" {src}" if src else ""
    if page is not None:
        meta += f" p={page}"
    header = f"{tag}{meta}".rstrip()
    return f"{header}\n{d.page_content.strip()}"

def build_prompt(question: str, context_docs, history: str = "") -> str:
    context = "\n\n".join(_fmt(d, i) for i, d in enumerate(context_docs)) or "(empty)"
    return PROMPT_TEMPLATE.format(
        context=context,
        history=history.strip(),
        question=question.strip()
    )
