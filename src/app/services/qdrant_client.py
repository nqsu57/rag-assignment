from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.app.config.settings import settings
from src.app.utils.logger import get_logger

logger = get_logger(__name__)

def build_qdrant_client() -> QdrantClient:
    """Trả về QdrantClient kết nối; raise nếu không kết nối được."""
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        logger.info("Connected to Qdrant %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT)
        return client
    except Exception as exc:
        logger.exception("Unable to connect to Qdrant: %s", exc)
        raise

    
if __name__ == "__main__":
    client = build_qdrant_client()
    print("Collections:", client.get_collections())