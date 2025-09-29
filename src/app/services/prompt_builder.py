from typing import List, Dict
from src.app.models.settings import settings
from src.app.utils.helpers.prompts import RAG_BASE_PROMPT

def build_prompt(query: str, hits: List[Dict], max_chars: int | None = None) -> str:
    max_chars = max_chars or settings.MAX_CONTEXT_CHARS
    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits):
        text = h["payload"].get("text", "")
        if used + len(text) > max_chars:
            remaining = max_chars - used
            if remaining <= 0:
                break
            text = text[:remaining]
        parts.append(f"[{i}] {text}")
        used += len(text)
    context = "\n\n".join(parts) if parts else ""
    return RAG_BASE_PROMPT.format(context=context, query=query)
