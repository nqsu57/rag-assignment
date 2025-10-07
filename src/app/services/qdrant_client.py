import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))


def build_qdrant_client() -> QdrantClient:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("Connected to Qdrant %s:%s", QDRANT_HOST, QDRANT_PORT)
        return client
    except Exception as exc:
        logger.exception("Unable to connect to Qdrant: %s", exc)
        raise

    
if __name__ == "__main__":
    client = build_qdrant_client()
    print("Collections:", client.get_collections())