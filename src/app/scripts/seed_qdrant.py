import uuid
import os
from dotenv import load_dotenv
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from src.app.services.qdrant_collection import create_collection, recreate_collection
from src.app.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
EMBED_MODEL = os.getenv("EMBED_MODEL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")



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
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)

    model = SentenceTransformer(EMBED_MODEL)
    model_dim = model.get_sentence_embedding_dimension()
    logger.info("Using embedding model '%s' (dim=%s)", EMBED_MODEL, model_dim)

    RECREATE_COLLECTION = os.getenv('RECREATE_COLLECTION', 'True').lower() == 'true'
    print(RECREATE_COLLECTION)

    if RECREATE_COLLECTION:
        logger.info("RECREATE_COLLECTION=True -> recreating collection '%s'", QDRANT_COLLECTION)
        recreate_collection(client)
    else:
        logger.info("Ensuring collection '%s' exists", QDRANT_COLLECTION)
        create_collection(client)

    all_chunks = texts

    #Encode into vectors
    vectors = model.encode(all_chunks, convert_to_numpy=True)
    assert vectors.shape[1] == model_dim, "Vector dimension mismatch with model."

    #create a list of PointStruct for upsert
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

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info("Seeded %d chunks into collection '%s'", len(points), QDRANT_COLLECTION)


if __name__ == "__main__":
    seed_sample_texts(SAMPLE_TEXTS)
