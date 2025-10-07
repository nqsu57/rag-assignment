import os
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL")
print(EMBED_MODEL)
model = SentenceTransformer(EMBED_MODEL)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_numpy=True).tolist()