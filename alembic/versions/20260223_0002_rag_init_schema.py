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
        "knowledge_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dim=1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_knowledge_chunks_source"), "knowledge_chunks", ["source"], unique=False)


def downgrade() -> None:
    if not _is_target_db():
        return

    op.drop_index(op.f("ix_knowledge_chunks_source"), table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
