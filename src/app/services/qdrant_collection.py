import os
from dotenv import load_dotenv
from qdrant_client.http import models as rest
from qdrant_client import QdrantClient
from src.app.utils.logger import get_logger
from qdrant_client.http import models as rest

load_dotenv()
logger = get_logger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_COLLECTION = os.get("QDRANT_COLLECTION")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))


def create_collection(client: QdrantClient) -> None:
    try:
        existing = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION in existing:
            logger.info("Collection %s already exists; skip create", QDRANT_COLLECTION)
            return
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(size=EMBEDDING_DIM, distance=rest.Distance.COSINE),
        )
        logger.info("Created collection %s", QDRANT_COLLECTION)
    except Exception as exc:
        logger.exception("Failed to create collection: %s", exc)
        raise

def recreate_collection(client: QdrantClient) -> None:
    try:
        if client.collection_exists(QDRANT_COLLECTION):
            client.delete_collection(QDRANT_COLLECTION)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(
                size=EMBEDDING_DIM,
                distance=rest.Distance.COSINE,
            ),
        )
        logger.info("Recreated collection %s", QDRANT_COLLECTION)

    except Exception as exc:
        logger.exception("Failed to recreate collection: %s", exc)
        raise

def get_collection_info(client: QdrantClient):
    try:
        return client.get_collection(QDRANT_COLLECTION)
    except Exception as exc:
        logger.exception("Failed to get collection info: %s", exc)
        raise