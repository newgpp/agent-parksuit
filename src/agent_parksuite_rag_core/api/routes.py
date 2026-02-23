from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, delete, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.db.models import KnowledgeChunk, KnowledgeSource
from agent_parksuite_rag_core.db.session import get_db_session
from agent_parksuite_rag_core.schemas.rag import (
    ChunkIngestRequest,
    ChunkIngestResponse,
    KnowledgeSourceResponse,
    KnowledgeSourceUpsertRequest,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResponseItem,
)

router = APIRouter(prefix="/api/v1", tags=["rag-core"])


def _utcnow() -> datetime:
    return datetime.now(UTC)


@router.post("/knowledge/sources", response_model=KnowledgeSourceResponse)
async def upsert_knowledge_source(
    payload: KnowledgeSourceUpsertRequest,
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeSourceResponse:
    existing = (
        await session.execute(select(KnowledgeSource).where(KnowledgeSource.source_id == payload.source_id))
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
        session.add(row)
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

    await session.commit()
    await session.refresh(row)
    return KnowledgeSourceResponse.model_validate(row, from_attributes=True)


@router.post("/knowledge/chunks/batch", response_model=ChunkIngestResponse)
async def ingest_knowledge_chunks(
    payload: ChunkIngestRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ChunkIngestResponse:
    source = (
        await session.execute(select(KnowledgeSource).where(KnowledgeSource.source_id == payload.source_id))
    ).scalar_one_or_none()
    if source is None:
        raise HTTPException(status_code=404, detail=f"source_id not found: {payload.source_id}")

    for idx, item in enumerate(payload.chunks):
        if len(item.embedding) != settings.embedding_dim:
            raise HTTPException(
                status_code=400,
                detail=f"chunk[{idx}] embedding dim mismatch: expected {settings.embedding_dim}, got {len(item.embedding)}",
            )

    if payload.replace_existing:
        await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source_pk == source.id))

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
    session.add_all(rows)
    source.updated_at = _utcnow()
    await session.commit()
    return ChunkIngestResponse(source_pk=source.id, inserted_count=len(rows))


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    payload: RetrieveRequest,
    session: AsyncSession = Depends(get_db_session),
) -> RetrieveResponse:
    if payload.query_embedding is not None and len(payload.query_embedding) != settings.embedding_dim:
        raise HTTPException(
            status_code=400,
            detail=f"query_embedding dim mismatch: expected {settings.embedding_dim}, got {len(payload.query_embedding)}",
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
        stmt = stmt.order_by(score_expr.asc())
    else:
        stmt = stmt.order_by(KnowledgeChunk.created_at.desc(), KnowledgeChunk.id.desc())

    rows = (await session.execute(stmt.limit(payload.top_k))).all()
    items = [
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
    return RetrieveResponse(items=items)
