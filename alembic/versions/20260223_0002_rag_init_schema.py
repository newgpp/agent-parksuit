"""init rag schema

Revision ID: 20260223_0002_rag
Revises: 20260223_0001_biz
Create Date: 2026-02-23 20:31:00

"""
from __future__ import annotations

from urllib.parse import urlparse

from alembic import context, op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260223_0002_rag"
down_revision = "20260223_0001_biz"
branch_labels = None
depends_on = None


TARGET_DB = "parksuite_rag"


def _current_db_name() -> str:
    url = context.config.get_main_option("sqlalchemy.url")
    return urlparse(url).path.lstrip("/").split("?", 1)[0]


def _is_target_db() -> bool:
    return _current_db_name() == TARGET_DB


def upgrade() -> None:
    if not _is_target_db():
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "knowledge_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("source_id", sa.String(length=128), nullable=False, comment="来源业务ID"),
        sa.Column("doc_type", sa.String(length=32), nullable=False, comment="文档类型"),
        sa.Column("source_type", sa.String(length=32), nullable=False, server_default="biz_derived", comment="来源类型"),
        sa.Column("title", sa.String(length=255), nullable=False, server_default="", comment="标题"),
        sa.Column("city_code", sa.String(length=32), nullable=True, comment="城市编码"),
        sa.Column("lot_codes", postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment="停车场编码列表"),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True, comment="生效开始时间"),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True, comment="生效结束时间"),
        sa.Column("version", sa.String(length=64), nullable=True, comment="知识版本号"),
        sa.Column("source_uri", sa.String(length=512), nullable=True, comment="来源链接或文件路径"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true"), comment="是否生效"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        comment="知识来源主表（文档级）",
    )
    op.create_index(op.f("ix_knowledge_sources_source_id"), "knowledge_sources", ["source_id"], unique=True)
    op.create_index(op.f("ix_knowledge_sources_doc_type"), "knowledge_sources", ["doc_type"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_source_type"), "knowledge_sources", ["source_type"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_city_code"), "knowledge_sources", ["city_code"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_effective_from"), "knowledge_sources", ["effective_from"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_effective_to"), "knowledge_sources", ["effective_to"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_is_active"), "knowledge_sources", ["is_active"], unique=False)
    op.create_index(
        "ix_knowledge_sources_lot_codes_gin",
        "knowledge_sources",
        ["lot_codes"],
        unique=False,
        postgresql_using="gin",
    )

    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("source_pk", sa.Integer(), nullable=False, comment="所属知识来源ID（逻辑绑定）"),
        sa.Column("scenario_id", sa.String(length=64), nullable=True, comment="场景ID"),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0", comment="分块序号"),
        sa.Column("chunk_text", sa.Text(), nullable=False, comment="分块内容"),
        sa.Column("embedding", Vector(dim=1536), nullable=False, comment="向量"),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment="扩展元数据"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.PrimaryKeyConstraint("id"),
        comment="知识分块表（向量检索）",
    )
    op.create_index(op.f("ix_knowledge_chunks_source_pk"), "knowledge_chunks", ["source_pk"], unique=False)
    op.create_index(op.f("ix_knowledge_chunks_scenario_id"), "knowledge_chunks", ["scenario_id"], unique=False)
    op.execute(
        "CREATE INDEX ix_knowledge_chunks_embedding_ivfflat "
        "ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    if not _is_target_db():
        return

    op.drop_index("ix_knowledge_chunks_embedding_ivfflat", table_name="knowledge_chunks")
    op.drop_index(op.f("ix_knowledge_chunks_scenario_id"), table_name="knowledge_chunks")
    op.drop_index(op.f("ix_knowledge_chunks_source_pk"), table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")

    op.drop_index("ix_knowledge_sources_lot_codes_gin", table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_is_active"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_effective_to"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_effective_from"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_city_code"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_source_type"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_doc_type"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_source_id"), table_name="knowledge_sources")
    op.drop_table("knowledge_sources")
