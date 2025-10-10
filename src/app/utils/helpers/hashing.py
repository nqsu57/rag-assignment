import hashlib
from pathlib import Path
from qdrant_client.http import models

def compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_doc_hash_filter(doc_hash: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.doc_hash",
                match=models.MatchValue(value=doc_hash),
            )
        ]
    )