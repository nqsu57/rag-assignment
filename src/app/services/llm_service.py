from typing import List
from src.app.models.settings import settings
import time
import requests

def _mock_generate_answer(context_texts: List[str], query: str) -> str:
    if not context_texts:
        return "I don't know."
    combined = " ".join(context_texts)
    summary = combined[:400]
    return f"Based on the context: {summary}... (concise answer based on provided context)."

def generate_answer(prompt: str, hits: List[dict]) -> str:
    """Sync function: generate answer. Caller can run in thread if needed."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "mock":
        texts = [h["payload"].get("text", "") for h in hits]
        return _mock_generate_answer(texts, prompt)
    if provider == "hf":
        url = f"https://api-inference.huggingface.co/models/{settings.HF_MODEL}"
        headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
        resp = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    if provider == "openai":
        import openai
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key not configured")
        openai.api_key = settings.OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        return resp.choices[0].message.content
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")
