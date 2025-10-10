import os
from dotenv import load_dotenv
from typing import List, Dict
from src.app.utils.helpers.prompts import RAG_BASE_PROMPT

load_dotenv()
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))

def build_prompt(query: str, hits: List[Dict]) -> str:
    texts = [h.get("payload", {}).get("text", "") for h in hits]
    
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "...[truncated]"

    return RAG_BASE_PROMPT.format(context=context, query=query)
