from __future__ import annotations

import re
from datetime import UTC, datetime

from sqlalchemy import and_, delete, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_parksuite_rag_core.db.models import KnowledgeChunk, KnowledgeSource
from agent_parksuite_rag_core.schemas.rag import (
    ChunkIngestRequest,
    KnowledgeSourceUpsertRequest,
    RetrieveRequest,
    RetrieveResponseItem,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _tokenize_for_match(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query.lower())
    return [token for token in tokens if len(token) >= 2]


def _lexical_match_score(query: str, title: str, content: str) -> int:
    tokens = _tokenize_for_match(query)
    if not tokens:
        return 0

    haystack = f"{title} {content}".lower()
    score = 0
    for token in tokens:
        if token in haystack:
            score += len(token)
    return score


class KnowledgeRepository:
    def __init__(self, session: AsyncSession, embedding_dim: int) -> None:
        self.session = session
        self.embedding_dim = embedding_dim

    async def upsert_source(self, payload: KnowledgeSourceUpsertRequest) -> KnowledgeSource:
        existing = (
            await self.session.execute(select(KnowledgeSource).where(KnowledgeSource.source_id == payload.source_id))
        ).scalar_one_or_none()

        if existing is None:
            row = KnowledgeSource(
                source_id=payload.source_id,
                doc_type=payload.doc_type,
                source_type=payload.source_type,
                title=payload.title,
                city_code=payload.city_code,
                lot_codes=payload.lot_codes,
                effective_from=payload.effective_from,
                effective_to=payload.effective_to,
                version=payload.version,
                source_uri=payload.source_uri,
                is_active=payload.is_active,
            )
            self.session.add(row)
        else:
            existing.doc_type = payload.doc_type
            existing.source_type = payload.source_type
            existing.title = payload.title
            existing.city_code = payload.city_code
            existing.lot_codes = payload.lot_codes
            existing.effective_from = payload.effective_from
            existing.effective_to = payload.effective_to
            existing.version = payload.version
            existing.source_uri = payload.source_uri
            existing.is_active = payload.is_active
            existing.updated_at = _utcnow()
            row = existing

        await self.session.commit()
        await self.session.refresh(row)
        return row

    async def ingest_chunks(self, payload: ChunkIngestRequest) -> tuple[int, int]:
        source = (
            await self.session.execute(select(KnowledgeSource).where(KnowledgeSource.source_id == payload.source_id))
        ).scalar_one_or_none()
        if source is None:
            raise LookupError(f"source_id not found: {payload.source_id}")

        for idx, item in enumerate(payload.chunks):
            if len(item.embedding) != self.embedding_dim:
                raise ValueError(
                    f"chunk[{idx}] embedding dim mismatch: expected {self.embedding_dim}, got {len(item.embedding)}"
                )

        if payload.replace_existing:
            await self.session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source_pk == source.id))

        rows = [
            KnowledgeChunk(
                source_pk=source.id,
                scenario_id=item.scenario_id,
                chunk_index=item.chunk_index,
                chunk_text=item.chunk_text,
                embedding=item.embedding,
                chunk_metadata=item.metadata,
            )
            for item in payload.chunks
        ]
        self.session.add_all(rows)
        source.updated_at = _utcnow()
        await self.session.commit()
        return source.id, len(rows)

    async def retrieve(self, payload: RetrieveRequest) -> list[RetrieveResponseItem]:
        if payload.query_embedding is not None and len(payload.query_embedding) != self.embedding_dim:
            raise ValueError(
                f"query_embedding dim mismatch: expected {self.embedding_dim}, got {len(payload.query_embedding)}"
            )

        score_expr = (
            KnowledgeChunk.embedding.cosine_distance(payload.query_embedding).label("score")
            if payload.query_embedding is not None
            else literal(None).label("score")
        )
        stmt = select(KnowledgeChunk, KnowledgeSource, score_expr).join(
            KnowledgeSource, KnowledgeChunk.source_pk == KnowledgeSource.id
        )

        filters = []
        if not payload.include_inactive:
            filters.append(KnowledgeSource.is_active.is_(True))
        if payload.doc_type:
            filters.append(KnowledgeSource.doc_type == payload.doc_type)
        if payload.source_type:
            filters.append(KnowledgeSource.source_type == payload.source_type)
        if payload.city_code:
            filters.append(KnowledgeSource.city_code == payload.city_code)
        if payload.lot_code:
            filters.append(KnowledgeSource.lot_codes.contains([payload.lot_code]))
        if payload.source_ids:
            filters.append(KnowledgeSource.source_id.in_(payload.source_ids))
        if payload.at_time:
            filters.append(
                and_(
                    or_(KnowledgeSource.effective_from.is_(None), KnowledgeSource.effective_from <= payload.at_time),
                    or_(KnowledgeSource.effective_to.is_(None), KnowledgeSource.effective_to > payload.at_time),
                )
            )

        if filters:
            stmt = stmt.where(and_(*filters))

        if payload.query_embedding is not None:
            stmt = stmt.order_by(
                score_expr.asc(),
                KnowledgeSource.source_id.asc(),
                KnowledgeChunk.chunk_index.asc(),
                KnowledgeChunk.id.asc(),
            )
            rows = (await self.session.execute(stmt.limit(payload.top_k))).all()
        else:
            candidate_limit = max(payload.top_k * 10, 100)
            stmt = stmt.order_by(
                KnowledgeSource.source_id.asc(),
                KnowledgeChunk.chunk_index.asc(),
                KnowledgeChunk.id.asc(),
            )
            rows = (await self.session.execute(stmt.limit(candidate_limit))).all()
            if payload.query.strip():
                rows.sort(
                    key=lambda row: (
                        -_lexical_match_score(payload.query, row[1].title or "", row[0].chunk_text or ""),
                        row[1].source_id,
                        row[0].chunk_index,
                        row[0].id,
                    )
                )
            rows = rows[: payload.top_k]

        return [
            RetrieveResponseItem(
                chunk_id=chunk.id,
                source_pk=chunk.source_pk,
                source_id=source.source_id,
                doc_type=source.doc_type,
                source_type=source.source_type,
                title=source.title,
                content=chunk.chunk_text,
                scenario_id=chunk.scenario_id,
                metadata=chunk.chunk_metadata,
                score=float(score) if score is not None else None,
            )
            for chunk, source, score in rows
        ]

