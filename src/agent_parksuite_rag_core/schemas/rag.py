from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeSourceUpsertRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=128, description="知识来源唯一标识")
    doc_type: str = Field(min_length=1, max_length=32, description="文档类型，如 rule_explain/faq/policy_doc")
    source_type: str = Field(
        default="biz_derived",
        min_length=1,
        max_length=32,
        description="来源类型，如 biz_derived/manual",
    )
    title: str = Field(default="", max_length=255, description="知识标题")
    city_code: str | None = Field(default=None, max_length=32, description="城市编码")
    lot_codes: list[str] = Field(default_factory=list, description="停车场编码列表")
    effective_from: datetime | None = Field(default=None, description="生效开始时间")
    effective_to: datetime | None = Field(default=None, description="生效结束时间（为空表示长期生效）")
    version: str | None = Field(default=None, max_length=64, description="知识版本号")
    source_uri: str | None = Field(default=None, max_length=512, description="来源链接或文件路径")
    is_active: bool = Field(default=True, description="是否生效")


class KnowledgeSourceResponse(BaseModel):
    id: int = Field(description="知识来源主键ID")
    source_id: str = Field(description="知识来源唯一标识")
    doc_type: str = Field(description="文档类型")
    source_type: str = Field(description="来源类型")
    title: str = Field(description="知识标题")
    city_code: str | None = Field(description="城市编码")
    lot_codes: list[str] = Field(description="停车场编码列表")
    effective_from: datetime | None = Field(description="生效开始时间")
    effective_to: datetime | None = Field(description="生效结束时间")
    version: str | None = Field(description="知识版本号")
    source_uri: str | None = Field(description="来源链接或文件路径")
    is_active: bool = Field(description="是否生效")


class ChunkIngestItem(BaseModel):
    scenario_id: str | None = Field(default=None, max_length=64, description="关联场景ID（可选）")
    chunk_index: int = Field(default=0, ge=0, description="分块序号，从0开始")
    chunk_text: str = Field(min_length=1, description="分块文本内容")
    embedding: list[float] = Field(min_length=1, description="分块向量")
    metadata: dict[str, Any] = Field(default_factory=dict, description="分块扩展元数据")


class ChunkIngestRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=128, description="要写入的知识来源ID")
    replace_existing: bool = Field(default=False, description="是否先删除该来源下已有分块再写入")
    chunks: list[ChunkIngestItem] = Field(min_length=1, description="待写入的分块列表")


class ChunkIngestResponse(BaseModel):
    source_pk: int = Field(description="知识来源主键ID")
    inserted_count: int = Field(description="本次成功写入的分块数量")


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
