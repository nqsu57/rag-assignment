import tempfile
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from src.app.services.embedder import embed_texts
from src.app.utils.helpers.rag_train import save_and_hash_files, chunk_all_documents, upsert_chunks_to_qdrant, cleanup_tmp

train_rag_router = APIRouter()
ALLOWED_FILE = {".txt", ".pdf", ".docx"}

@train_rag_router.post("/train-rag")
async def train_rag(files: List[UploadFile] = File(...)) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        file_hashes = save_and_hash_files(files, tmp_dir)
        chunks = chunk_all_documents(tmp_dir)
        vectors = await embed_texts([c["text"] for c in chunks])
        stored = upsert_chunks_to_qdrant(chunks, vectors, file_hashes)
        return {
            "status": "success",
            "files_received": list(file_hashes.keys()),
            "chunks_stored": stored,
        }
    finally:
        cleanup_tmp(files, tmp_dir)