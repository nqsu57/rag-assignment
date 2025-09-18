from contextlib import asynccontextmanager
import asyncio
import logging
from fastapi import FastAPI, Request, Depends
from qdrant_client import QdrantClient
from src.app.config.settings import settings
from src.app.services.qdrant_collection import create_collection
from src.app.api.routes_chunk import router_chunks
from src.app.api.call_bot import call_bot_router


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connecting to Qdrant
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT) 

    #retry/backoff nhỏ đề phòng Qdrant khởi chậm
    max_tries = 5
    for attempt in range(1, max_tries + 1):
        try:
            create_collection(client)
            logger.info("Qdrant collection ready: %s", settings.QDRANT_COLLECTION)
            break
        except Exception as exc:
            logger.warning("Qdrant not ready (%d/%d): %s", attempt, max_tries, exc)
            if attempt == max_tries:
                logger.exception("Could not connect to Qdrant")
                raise
            await asyncio.sleep(1 * attempt)  # simple backoff

    # gắn client để dùng trong route
    app.state.qdrant_client = client

    # Nếu có bước init LlamaIndex (ví dụ build index), thực hiện tại đây.
    # app.state.index = build_llama_index(...)
    yield
    logger.info("Shutting down FastAPI app – closing Qdrant client")
    client_close = getattr(client, "close", None)
    if callable(client_close):
        maybe = client_close()
        if asyncio.iscoroutine(maybe):
            await maybe


app = FastAPI(title="Minimal RAG with FastAPI & Qdrant", version="1.0.0", lifespan=lifespan)

app.include_router(router_chunks, prefix="/api")
app.include_router(call_bot_router, prefix="/api")



def get_qdrant_client(request: Request) -> QdrantClient:
    """Dependency để lấy QdrantClient đã khởi tạo."""
    return request.app.state.qdrant_client


@app.get("/health")
async def health_check(client: QdrantClient = Depends(get_qdrant_client)):
    """
    Endpoint kiểm tra Qdrant có sẵn sàng.
    """
    return {
        "status": "ok",
        "collection": settings.QDRANT_COLLECTION,
        "qdrant_connected": client is not None,
    }
