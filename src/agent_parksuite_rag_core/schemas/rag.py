from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrieveResponseItem(BaseModel):
    id: int
    source: str
    title: str
    content: str


class RetrieveResponse(BaseModel):
    items: list[RetrieveResponseItem]
