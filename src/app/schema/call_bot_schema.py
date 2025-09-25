from pydantic import BaseModel, Field
from typing import List, Optional


class CallBotRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=50)

class RetrievedItem(BaseModel):
    id: str
    score: float | None
    source: str | None
    document_id: str | None

class CallBotResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedItem]