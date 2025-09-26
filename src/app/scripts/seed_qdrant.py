import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from src.app.models.settings import settings
from src.app.services.qdrant_collection import create_collection, recreate_collection
from src.app.utils.logger import get_logger


logger = get_logger(__name__)

SAMPLE_TEXTS: List[str] = [
    "Hugging Face is a platform for machine learning models and datasets. "
    "It provides tools like Transformers, Datasets, and the Hugging Face Hub "
    "to share, train, and deploy natural language processing models efficiently.",

    "Qdrant is a high-performance vector database designed for scalable similarity search "
    "and neural network embeddings. It supports features such as filtering, "
    "payload storage, and distributed deployments for production-grade applications.",

    "FastAPI is a modern, high-performance, web framework for building APIs with Python 3. "
    "It is based on standard Python type hints, enabling automatic documentation, "
    "data validation, and asynchronous request handling for maximum speed.",

    "Sentence-Transformers is a Python library that makes it easy to compute dense vector "
    "representations (embeddings) for sentences, paragraphs, and images. "
    "These embeddings can be used for tasks like semantic search, clustering, "
    "and information retrieval.",
]


def seed_sample_texts(texts: List[str]) -> None:
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    logger.info("Connected to Qdrant at %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT)

    model = SentenceTransformer(settings.EMBED_MODEL)
    model_dim = model.get_sentence_embedding_dimension()
    logger.info("Using embedding model '%s' (dim=%s)", settings.EMBED_MODEL, model_dim)

    if settings.RECREATE_COLLECTION:
        logger.info("RECREATE_COLLECTION=True -> recreating collection '%s'", settings.QDRANT_COLLECTION)
        recreate_collection(client)
    else:
        logger.info("Ensuring collection '%s' exists", settings.QDRANT_COLLECTION)
        create_collection(client)

    all_chunks = texts

    #Encode thành vectors
    vectors = model.encode(all_chunks, convert_to_numpy=True)
    assert vectors.shape[1] == model_dim, "Vector dimension mismatch with model."

    # creat list PointStruct để upsert
    points = [
        rest.PointStruct(
            id=uuid.uuid4().hex,
            vector=vec.tolist(),
            payload={
                "text": chunk,
                "document_id": "seed_doc",
                "source": "seed",
                "chunk_id": idx
            }
        )
        for idx, (chunk, vec) in enumerate(zip(all_chunks, vectors))
    ]

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    logger.info("Seeded %d chunks into collection '%s'", len(points), settings.QDRANT_COLLECTION)


if __name__ == "__main__":
    seed_sample_texts(SAMPLE_TEXTS)
