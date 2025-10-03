import time
import os
from dotenv import load_dotenv
from typing import List, Dict
from huggingface_hub import InferenceClient

load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))


def _mock_generate_answer(context_texts: List[str], query: str) -> str:
    if not context_texts:
        return "I don't know."
    combined = " ".join(context_texts)
    summary = combined[:400]
    return f"Based on the context: {summary}... (concise answer based on provided context)."

def generate_answer(prompt: str, hits: List[Dict]) -> str:
    client = InferenceClient(api_key=HF_API_TOKEN)
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_NEW_TOKENS,
        )

        return response.choices[0].message["content"].strip()
    except Exception as e:
        raise RuntimeError(f"LLM service error: {e}")