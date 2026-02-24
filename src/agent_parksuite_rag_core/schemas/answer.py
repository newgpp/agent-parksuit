from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, description="用户提问文本")
    query_embedding: list[float] | None = Field(default=None, description="查询向量（可选，提供后按向量相似度排序）")
    top_k: int = Field(default=5, ge=1, le=20, description="用于回答的召回数量上限")
    doc_type: str | None = Field(default=None, description="按文档类型过滤")
    source_type: str | None = Field(default=None, description="按来源类型过滤")
    city_code: str | None = Field(default=None, description="按城市编码过滤")
    lot_code: str | None = Field(default=None, description="按停车场编码过滤")
    at_time: datetime | None = Field(default=None, description="按生效时间点过滤")
    source_ids: list[str] | None = Field(default=None, description="限定来源ID集合")
    include_inactive: bool = Field(default=False, description="是否包含未生效数据")


class AnswerCitation(BaseModel):
    chunk_id: int = Field(description="分块ID")
    source_id: str = Field(description="知识来源ID")
    doc_type: str = Field(description="文档类型")
    title: str = Field(description="知识标题")
    snippet: str = Field(description="证据片段")
    score: float | None = Field(default=None, description="检索分数（越小越相似）")


class AnswerResponse(BaseModel):
    conclusion: str = Field(description="最终结论")
    key_points: list[str] = Field(default_factory=list, description="要点列表")
    citations: list[AnswerCitation] = Field(default_factory=list, description="引用证据")
    retrieved_count: int = Field(description="参与回答的检索条数")
    model: str = Field(description="回答使用的模型标识")


class HybridAnswerRequest(BaseModel):
    query: str = Field(min_length=1, description="用户提问文本")
    intent_hint: str | None = Field(default=None, description="可选意图提示：rule_explain/arrears_check/fee_verify")
    query_embedding: list[float] | None = Field(default=None, description="查询向量（可选）")
    top_k: int = Field(default=5, ge=1, le=20, description="召回数量上限")
    doc_type: str | None = Field(default=None, description="按文档类型过滤")
    source_type: str | None = Field(default=None, description="按来源类型过滤")
    city_code: str | None = Field(default=None, description="按城市编码过滤")
    lot_code: str | None = Field(default=None, description="按停车场编码过滤")
    at_time: datetime | None = Field(default=None, description="按生效时间点过滤")
    source_ids: list[str] | None = Field(default=None, description="限定来源ID集合")
    include_inactive: bool = Field(default=False, description="是否包含未生效数据")
    plate_no: str | None = Field(default=None, description="车牌号（arrears_check）")
    order_no: str | None = Field(default=None, description="订单号（fee_verify）")
    rule_code: str | None = Field(default=None, description="规则编码（fee_verify，可选）")
    entry_time: datetime | None = Field(default=None, description="入场时间（fee_verify，可选）")
    exit_time: datetime | None = Field(default=None, description="离场时间（fee_verify，可选）")


class HybridAnswerResponse(BaseModel):
    intent: str = Field(description="命中的意图类型")
    conclusion: str = Field(description="最终结论")
    key_points: list[str] = Field(default_factory=list, description="要点列表")
    business_facts: dict[str, Any] = Field(default_factory=dict, description="业务工具事实结果")
    citations: list[AnswerCitation] = Field(default_factory=list, description="引用证据")
    retrieved_count: int = Field(description="参与回答的检索条数")
    model: str = Field(description="回答使用的模型标识")
    graph_trace: list[str] = Field(default_factory=list, description="图执行节点轨迹")

