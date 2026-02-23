from __future__ import annotations

from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.db.base import Base


def _utcnow_utc() -> datetime:
    return datetime.now(UTC)


class KnowledgeSource(Base):
    __tablename__ = "knowledge_sources"
    __table_args__ = (
        Index("ix_knowledge_sources_lot_codes_gin", "lot_codes", postgresql_using="gin"),
        {"comment": "知识来源主表（文档级）"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键ID")
    source_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, comment="来源业务ID")
    doc_type: Mapped[str] = mapped_column(String(32), index=True, comment="文档类型")
    title: Mapped[str] = mapped_column(String(255), default="", comment="标题")
    city_code: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True, comment="城市编码")
    lot_codes: Mapped[list[str]] = mapped_column(JSONB, default=list, comment="停车场编码列表")
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True, comment="生效开始时间"
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True, comment="生效结束时间"
    )
    version: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="知识版本号")
    source_uri: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="来源链接或文件路径")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True, comment="是否生效")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow_utc, comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow_utc, onupdate=_utcnow_utc, comment="更新时间"
    )

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = ({"comment": "知识分块表（向量检索）"},)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键ID")
    source_pk: Mapped[int] = mapped_column(index=True, comment="所属知识来源ID（逻辑绑定）")
    scenario_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True, comment="场景ID")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0, comment="分块序号")
    chunk_text: Mapped[str] = mapped_column(Text, comment="分块内容")
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), comment="向量")
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, comment="扩展元数据")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow_utc, comment="创建时间")
