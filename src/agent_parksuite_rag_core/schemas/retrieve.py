from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(default="", description="用户查询文本")
    query_embedding: list[float] | None = Field(default=None, description="查询向量（可选，提供后按向量相似度排序）")
    top_k: int = Field(default=5, ge=1, le=50, description="返回结果数量上限")
    doc_type: str | None = Field(default=None, description="按文档类型过滤")
    source_type: str | None = Field(default=None, description="按来源类型过滤")
    city_code: str | None = Field(default=None, description="按城市编码过滤")
    lot_code: str | None = Field(default=None, description="按停车场编码过滤")
    at_time: datetime | None = Field(default=None, description="按生效时间点过滤")
    source_ids: list[str] | None = Field(default=None, description="限定来源ID集合")
    include_inactive: bool = Field(default=False, description="是否包含未生效数据")


class RetrieveResponseItem(BaseModel):
    chunk_id: int = Field(description="分块ID")
    source_pk: int = Field(description="知识来源主键ID")
    source_id: str = Field(description="知识来源ID")
    doc_type: str = Field(description="文档类型")
    source_type: str = Field(description="来源类型")
    title: str = Field(description="知识标题")
    content: str = Field(description="分块文本")
    scenario_id: str | None = Field(description="关联场景ID")
    metadata: dict[str, Any] = Field(description="分块扩展元数据")
    score: float | None = Field(default=None, description="相似度距离分数（越小越相似）")


class RetrieveResponse(BaseModel):
    items: list[RetrieveResponseItem] = Field(description="检索结果列表")

