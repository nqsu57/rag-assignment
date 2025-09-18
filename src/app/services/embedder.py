import asyncio
import hashlib
from typing import List
from src.app.config.settings import settings

# Lightweight deterministic mock embedding (useful for tests).
def _mock_vector_for_text(text: str, dim: int) -> List[float]:
    # deterministic pseudo-random floats in [-1,1]
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals: List[float] = []
    for i in range(dim):
        b = h[i % len(h)]
        vals.append(((b / 255.0) * 2.0) - 1.0)
    return vals

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts.
    Default: mock deterministic embedding. Replace with real model easily."""
    # If you integrate sentence-transformers, do heavy work in thread:
    # model = _load_sentence_transformer()
    # return await asyncio.to_thread(model.encode, texts, convert_to_numpy=True).tolist()
    dim = settings.VECTOR_DIM
    return [ _mock_vector_for_text(t, dim) for t in texts ]
