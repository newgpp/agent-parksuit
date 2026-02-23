from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeSourceUpsertRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=128)
    doc_type: str = Field(min_length=1, max_length=32)
    source_type: str = Field(default="biz_derived", min_length=1, max_length=32)
    title: str = Field(default="", max_length=255)
    city_code: str | None = Field(default=None, max_length=32)
    lot_codes: list[str] = Field(default_factory=list)
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    version: str | None = Field(default=None, max_length=64)
    source_uri: str | None = Field(default=None, max_length=512)
    is_active: bool = True


class KnowledgeSourceResponse(BaseModel):
    id: int
    source_id: str
    doc_type: str
    source_type: str
    title: str
    city_code: str | None
    lot_codes: list[str]
    effective_from: datetime | None
    effective_to: datetime | None
    version: str | None
    source_uri: str | None
    is_active: bool


class ChunkIngestItem(BaseModel):
    scenario_id: str | None = Field(default=None, max_length=64)
    chunk_index: int = Field(default=0, ge=0)
    chunk_text: str = Field(min_length=1)
    embedding: list[float] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkIngestRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=128)
    replace_existing: bool = False
    chunks: list[ChunkIngestItem] = Field(min_length=1)


class ChunkIngestResponse(BaseModel):
    source_pk: int
    inserted_count: int


class RetrieveRequest(BaseModel):
    query: str = ""
    query_embedding: list[float] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    doc_type: str | None = None
    source_type: str | None = None
    city_code: str | None = None
    lot_code: str | None = None
    at_time: datetime | None = None
    source_ids: list[str] | None = None
    include_inactive: bool = False


class RetrieveResponseItem(BaseModel):
    chunk_id: int
    source_pk: int
    source_id: str
    doc_type: str
    source_type: str
    title: str
    content: str
    scenario_id: str | None
    metadata: dict[str, Any]
    score: float | None = None


class RetrieveResponse(BaseModel):
    items: list[RetrieveResponseItem]
