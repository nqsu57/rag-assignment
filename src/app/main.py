import asyncio
import logging
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from qdrant_client import QdrantClient
from src.app.services.qdrant_collection import create_collection
from src.app.api.call_bot import call_bot_router
from src.app.api.rag import train_rag_router

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT) 

    max_tries = 5
    for attempt in range(1, max_tries + 1):
        try:
            create_collection(client)
            logger.info("Qdrant collection ready: %s", QDRANT_COLLECTION)
            break
        except Exception as exc:
            logger.warning("Qdrant not ready (%d/%d): %s", attempt, max_tries, exc)
            if attempt == max_tries:
                logger.exception("Could not connect to Qdrant")
                raise
            await asyncio.sleep(1 * attempt)

    app.state.qdrant_client = client
    yield
    logger.info("Shutting down FastAPI app – closing Qdrant client")
    client_close = getattr(client, "close", None)
    if callable(client_close):
        maybe = client_close()
        if asyncio.iscoroutine(maybe):
            await maybe


app = FastAPI(title="Minimal RAG with FastAPI & Qdrant", version="1.0.0", lifespan=lifespan)

app.include_router(call_bot_router, prefix="/api")
app.include_router(train_rag_router, prefix="/api")


def get_qdrant_client(request: Request) -> QdrantClient:
    return request.app.state.qdrant_client


@app.get("/health")
async def health_check(client: QdrantClient = Depends(get_qdrant_client)):
    return {
        "status": "ok",
        "collection": QDRANT_COLLECTION,
        "qdrant_connected": client is not None,
    }
