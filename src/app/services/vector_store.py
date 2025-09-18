from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import List, Dict, Any
import uuid
from src.app.config.settings import settings

class QdrantStore:
    def __init__(self, host: str = settings.QDRANT_HOST, port: int = settings.QDRANT_PORT):
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        try:
            existing = self.client.get_collections().collections
            names = [c.name for c in existing]
            if collection_name not in names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
                )
        except Exception as e:
            raise RuntimeError(f"Qdrant ensure_collection failed: {e}") from e

    def upsert_points(self, collection_name: str, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> List[str]:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads length mismatch")
        points = []
        ids: List[str] = []
        for i, vec in enumerate(vectors):
            pid = str(uuid.uuid4())
            ids.append(pid)
            points.append(rest.PointStruct(id=pid, vector=vec, payload=payloads[i]))
        self.client.upsert(collection_name=collection_name, points=points)
        return ids

    def search(self, collection_name: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self.client.search(collection_name=collection_name, query_vector=query_vector, limit=top_k, with_payload=True)
        results: List[Dict[str, Any]] = []
        for h in hits:
            results.append({"id": str(h.id), "score": getattr(h, "score", None), "payload": h.payload or {}})
        return results