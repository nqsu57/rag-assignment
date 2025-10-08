import os
import logging
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from qdrant_client.http.exceptions import ApiException
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.app.models.delete_vector_schema import DeleteVectorRequest, DeleteVectorResponse
from src.app.utils.helpers.hashing import build_doc_hash_filter

load_dotenv()
logger = logging.getLogger(__name__)
delete_vector_router = APIRouter()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
print(COLLECTION_NAME)


@delete_vector_router.post("/delete-vector", response_model=DeleteVectorResponse)
def delete_vector(request: Request, payload: DeleteVectorRequest):
    client: QdrantClient = request.app.state.qdrant_client
    try:
        filter_condition = build_doc_hash_filter(payload.doc_hash)

        count_result = client.count(collection_name=COLLECTION_NAME,
                                    count_filter=filter_condition,
                                    exact=True)
        count = count_result.count

        if count == 0:
            return DeleteVectorResponse(is_successful=True,
                                        deleted_count=0,
                                        details={"message": f"No points found for doc_hash={payload.doc_hash}"})

        client.delete(collection_name=COLLECTION_NAME,
                      points_selector=models.FilterSelector(filter=filter_condition), wait=True)

        logger.info(
            f"Deleted {count} vectors from collection='{COLLECTION_NAME}'")

        return DeleteVectorResponse(is_successful=True,
                                    deleted_count=count,
                                    details={"message": f"Deleted {count} points from collection '{COLLECTION_NAME}'"})

    except ApiException as e:
        logger.exception("Qdrant API error during delete-vector")
        raise HTTPException(status_code=502, detail="Qdrant API error") from e
    except Exception as e:
        logger.exception("Unexpected error during delete-vector")
        raise HTTPException(status_code=500, detail=str(e))