from fastapi import APIRouter
from pathlib import Path
from src.app.ingest.llama_chunker import load_documents_from_dir, chunk_documents, save_chunks

router_chunks = APIRouter()

@router_chunks.post("/chunks")
async def generate_chunks():
    docs = load_documents_from_dir(Path("data"))
    chunks = chunk_documents(docs)  
    out_path = save_chunks(chunks, Path("out_chunks"))
    return {"message": f"Saved {len(chunks)} chunks", "path": str(out_path)}