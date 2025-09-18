from qdrant_client.http import models as rest
from qdrant_client import QdrantClient
from src.app.config.settings import settings
from src.app.utils.logger import get_logger
from qdrant_client.http import models as rest

logger = get_logger(__name__)

def create_collection(client: QdrantClient) -> None:
    try:
        existing = [c.name for c in client.get_collections().collections]
        if settings.QDRANT_COLLECTION in existing:
            logger.info("Collection %s already exists; skip create", settings.QDRANT_COLLECTION)
            return
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(size=settings.EMBEDDING_DIM, distance=rest.Distance.COSINE),
        )
        logger.info("Created collection %s", settings.QDRANT_COLLECTION)
    except Exception as exc:
        logger.exception("Failed to create collection: %s", exc)
        raise


def recreate_collection(client: QdrantClient) -> None:
    try:
        if client.collection_exists(settings.QDRANT_COLLECTION):
            client.delete_collection(settings.QDRANT_COLLECTION)
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=rest.Distance.COSINE,
            ),
        )
        logger.info("Recreated collection %s", settings.QDRANT_COLLECTION)

    except Exception as exc:
        logger.exception("Failed to recreate collection: %s", exc)
        raise

"""Lấy thông tin cấu hình collection"""
def get_collection_info(client: QdrantClient):
    try:
        return client.get_collection(settings.QDRANT_COLLECTION)
    except Exception as exc:
        logger.exception("Failed to get collection info: %s", exc)
        raise