import asyncio
import os
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from src.app.utils.logger import get_logger
from src.app.services.embedder import embed_texts
from src.app.services.vector_store import QdrantStore
from src.app.services.prompt_builder import build_prompt
from src.app.services.llm_service import generate_answer
from src.app.models.call_bot_schema import CallBotRequest, CallBotResponse

logger = get_logger("call_bot")
call_bot_router = APIRouter()
load_dotenv()

TOP_K = int(os.getenv("TOP_K"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM")

def get_qdrant_store() -> QdrantStore:
    return QdrantStore()

@call_bot_router.post("/call-bot", response_model=CallBotResponse)
async def call_bot(req: CallBotRequest, store: QdrantStore = Depends(get_qdrant_store)) -> CallBotResponse:
    
    # Normalize the query
    normalize_query = req.query.strip().lower()
    top_k = req.top_k or TOP_K

    #embed query
    try:
        embeddings = await embed_texts([normalize_query])
        query_vec = embeddings[0]
    except Exception:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail="Embedding service error")

    #search qdrant
    try:
        store.ensure_collection(QDRANT_COLLECTION, EMBEDDING_DIM)
        hits = store.search(QDRANT_COLLECTION, query_vec, top_k=top_k)
    except Exception:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=503, detail="Vector search error")
    
    hits = [h for h in hits if h.get("score", 0.0) >= SIM_THRESHOLD]

    if not hits:
        return CallBotResponse(answer="No relevant context found.", retrieved=[])

    #build prompt
    prompt = build_prompt(normalize_query, hits)

    #call LLMs
    try:
        answer = await asyncio.to_thread(generate_answer, prompt)
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail="LLM service error")
        
    #format retrieved metadata
    retrieved = [
        {
            "id": h.get("id"),
            "score": h.get("score"),
            "source": h.get("payload", {}).get("metadata", {}).get("file_name"),
            "document_id": h.get("payload", {}).get("metadata", {}).get("doc_hash"),
        }
        for h in hits
    ]

    return CallBotResponse(answer=answer, retrieved=retrieved)
