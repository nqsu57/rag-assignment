# src/app/schemas/rag.py
from typing import Optional
from pydantic import BaseModel

class DeleteVectorRequest(BaseModel):
    doc_hash: str
    collection_name: Optional[str] = None

class DeleteVectorResponse(BaseModel):
    is_successful: bool
    deleted_count: int
    details: Optional[dict] = None