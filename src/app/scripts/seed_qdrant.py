"""
Quick script to seed Qdrant with sample docs for testing /call-bot.
Run: python scripts/seed_qdrant.py
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.app.config.settings import settings
from src.app.services.embedder import _mock_vector_for_text

def seed_sample():
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    collection = settings.QDRANT_COLLECTION
    dim = settings.VECTOR_DIM
    # create or replace (dev only)
    try:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE)
        )
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE)
        )

    texts = [
        "Hugging Face is a platform for machine learning models and datasets.",
        "Qdrant is a vector database focused on scalable similarity search.",
        "FastAPI is a modern Python framework for building APIs quickly."
    ]
    points = []
    for i, t in enumerate(texts):
        vec = _mock_vector_for_text(t, dim)
        points.append(rest.PointStruct(id=i+1, vector=vec, payload={"text": t, "document_id": f"doc{i+1}", "source": "seed"}))
    client.upsert(collection_name=collection, points=points)
    print("Seeded Qdrant with sample texts.")

if __name__ == "__main__":
    seed_sample()
