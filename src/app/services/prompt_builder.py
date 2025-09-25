from typing import List, Dict
from src.app.config.settings import settings

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
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer the question.\n"
        "If the answer cannot be found in the context, reply exactly: 'I don't know.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer:"
    )
    return prompt
