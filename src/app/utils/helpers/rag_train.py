from pathlib import Path
from typing import List
import tempfile, shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.app.models.settings import settings
from src.app.utils.logger import get_logger
from src.app.services.vector_store import QdrantStore
from src.app.services.llama_chunker import load_documents_from_dir, chunk_documents
from src.app.utils.helpers.hashing import compute_hash

logger = get_logger(__name__)
ALLOWED_FILE = {".txt", ".pdf", ".docx"}

def save_and_hash_files(files: List[UploadFile], tmp_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_FILE:
            raise HTTPException(400, f"Unsupported type: {ext}")
        dest = tmp_dir / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        hashes[f.filename] = compute_hash(dest)
    return hashes

def chunk_all_documents(tmp_dir: Path) -> list[dict]:
    docs = load_documents_from_dir(tmp_dir)
    if not docs:
        raise HTTPException(400, "No documents loaded")
    return chunk_documents(
        docs,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        use_sentence_splitter=False
    )

def upsert_chunks_to_qdrant(chunks: list[dict], vectors: list[list[float]],
                            file_hashes: dict[str, str]) -> int:
    qdrant = QdrantStore()
    qdrant.ensure_collection(settings.QDRANT_COLLECTION, settings.EMBEDDING_DIM)

    ids, payloads = [], []
    for idx, c in enumerate(chunks):
        fname = c["metadata"].get("file_name", "")
        doc_hash = file_hashes.get(fname, "")
        ids.append(f"{doc_hash}_{idx}")
        payloads.append({
            "text": c["text"],
            "metadata": {**c["metadata"], "doc_hash": doc_hash},
            "chunk_id": c["id"]
        })
    return len(qdrant.upsert_points(settings.QDRANT_COLLECTION, vectors, payloads, ids=ids))

def cleanup_tmp(files: List[UploadFile], tmp_dir: Path) -> None:
    for f in files:
        f.file.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)