import tempfile
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from src.app.services.vector_store import QdrantStore
from src.app.services.embedder import embed_texts
from src.app.ingest.llama_chunker import load_documents_from_dir, chunk_documents 
from src.app.config.settings import settings

train_rag_router = APIRouter()
ALLOWED_FILE = {".txt", ".pdf", ".docx"}

@train_rag_router.post("/train-rag")
async def train_rag(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tmp_dir = Path(tempfile.mkdtemp())
    saved_files = []
    
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_FILE:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_FILE)}"
            )

        tmp_path = tmp_dir / f.filename
        with open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(f.file, out_file)
        saved_files.append(f.filename)

    docs = load_documents_from_dir(tmp_dir)
    if not docs:
        raise HTTPException(status_code=400, detail="No documents loaded")

    chunks = chunk_documents(docs,
                             chunk_size=settings.CHUNK_SIZE,
                             chunk_overlap=settings.CHUNK_OVERLAP,
                             use_sentence_splitter=False)
    texts = [c["text"] for c in chunks]

    vectors = await embed_texts(texts)

    qdrant = QdrantStore()
    qdrant.ensure_collection(settings.QDRANT_COLLECTION, settings.EMBEDDING_DIM)

    payloads = [
        {"text": c["text"], "metadata": c["metadata"], "chunk_id": c["id"]}
        for c in chunks
    ]
    ids = qdrant.upsert_points(settings.QDRANT_COLLECTION, vectors, payloads)

    return {
        "status": "success",
        "files_received": saved_files,
        "chunks_stored": len(ids)
    }