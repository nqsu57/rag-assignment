from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from typing import List, Dict, Any
import uuid
from src.app.models.settings import settings

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

    def upsert_points(self, collection_name: str, 
                      vectors: List[List[float]], 
                      payloads: List[Dict[str, Any]], 
                      ids: List[str] | None = None, 
                      skip_existing: bool = True) -> List[str]:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads length mismatch")
        if ids and len(ids) != len(vectors):
            raise ValueError("ids length mismatch")
        
        point_ids = ids if ids else [str(uuid.uuid4()) for _ in vectors]

        points = [
            rest.PointStruct(id=pid, vector=vec, payload=payloads[i])
            for i, (pid, vec) in enumerate(zip(point_ids, vectors))
        ]

        if skip_existing and ids:
            records = self.client.retrieve(
                collection_name=collection_name,
                ids=point_ids,
                with_payload=False
            )
            existing_ids = {str(r.id) for r in records} 
            points = [p for p in points if str(p.id) not in existing_ids]
            point_ids = [str(p.id) for p in points]

        if points:
            self.client.upsert(collection_name=collection_name, points=points)

        return point_ids

    def search(self, collection_name: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self.client.search(collection_name=collection_name, query_vector=query_vector, limit=top_k, with_payload=True)
        return [
            {"id": str(h.id), "score": getattr(h, "score", None), "payload": h.payload or {}}
            for h in hits
        ]