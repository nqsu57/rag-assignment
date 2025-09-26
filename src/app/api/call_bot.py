from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import asyncio
from typing import List, Dict
from src.app.utils.logger import get_logger
from src.app.models.settings import settings
from src.app.services.embedder import embed_texts
from src.app.services.vector_store import QdrantStore
from src.app.services.prompt_builder import build_prompt
from src.app.services.llm_service import generate_answer
from src.app.models.call_bot_schema import CallBotRequest, CallBotResponse

logger = get_logger("call_bot")
call_bot_router = APIRouter()

def get_qdrant_store() -> QdrantStore:
    return QdrantStore()

@call_bot_router.post("/call-bot", response_model=CallBotResponse)
async def call_bot(
    req: CallBotRequest,
    store: QdrantStore = Depends(get_qdrant_store),
) -> CallBotResponse:
    # Normalize the query
    clean_query = req.query.strip()
    lower_query = clean_query.lower()

    top_k = req.top_k or settings.TOP_K
    SIM_THRESHOLD = settings.SIM_THRESHOLD
  
    #embed query
    try:
        embeddings = await embed_texts([lower_query])
        query_vec = embeddings[0]
    except Exception:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail="Embedding service error")

    #search qdrant
    try:
        store.ensure_collection(settings.QDRANT_COLLECTION, settings.EMBEDDING_DIM)
        hits = store.search(settings.QDRANT_COLLECTION, query_vec, top_k=top_k)
    except Exception:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=503, detail="Vector search error")
    
    hits = [h for h in hits if h.get("score", 0.0) >= SIM_THRESHOLD]
    if not hits:
        return CallBotResponse(answer="No relevant context found.", retrieved=[])

    #build prompt
    prompt = build_prompt(clean_query, hits, max_chars=settings.MAX_CONTEXT_CHARS)

    #call LLMs
    try:
        answer = await asyncio.to_thread(generate_answer, prompt, hits)
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=502, detail="LLM service error")

    #format retrieved metadata
    retrieved = [
        {
            "id": h.get("id"),
            "score": h.get("score"),
            "source": h.get("payload", {}).get("source"),
            "document_id": h.get("payload", {}).get("document_id"),
        }
        for h in hits
    ]

    return CallBotResponse(answer=answer, retrieved=retrieved)
