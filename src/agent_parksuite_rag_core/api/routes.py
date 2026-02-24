from __future__ import annotations

import re
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, delete, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.db.models import KnowledgeChunk, KnowledgeSource
from agent_parksuite_rag_core.db.session import get_db_session
from agent_parksuite_rag_core.schemas.rag import (
    AnswerCitation,
    AnswerRequest,
    AnswerResponse,
    ChunkIngestRequest,
    ChunkIngestResponse,
    KnowledgeSourceResponse,
    KnowledgeSourceUpsertRequest,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResponseItem,
)
from agent_parksuite_rag_core.services.answering import generate_answer_from_chunks

router = APIRouter(prefix="/api/v1", tags=["rag-core"])


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _tokenize_for_match(query: str) -> list[str]:
    # Keep alnum words and contiguous CJK spans for lightweight lexical ranking.
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


async def _retrieve_items(payload: RetrieveRequest, session: AsyncSession) -> list[RetrieveResponseItem]:
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
        stmt = stmt.order_by(
            score_expr.asc(),
            KnowledgeSource.source_id.asc(),
            KnowledgeChunk.chunk_index.asc(),
            KnowledgeChunk.id.asc(),
        )
        rows = (await session.execute(stmt.limit(payload.top_k))).all()
    else:
        candidate_limit = max(payload.top_k * 10, 100)
        stmt = stmt.order_by(
            KnowledgeSource.source_id.asc(),
            KnowledgeChunk.chunk_index.asc(),
            KnowledgeChunk.id.asc(),
        )
        rows = (await session.execute(stmt.limit(candidate_limit))).all()
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


@router.post(
    "/knowledge/sources",
    response_model=KnowledgeSourceResponse,
    summary="新增或更新知识来源",
    description="按 source_id 幂等写入知识来源元数据，用于后续分块入库与检索过滤。",
)
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


@router.post(
    "/knowledge/chunks/batch",
    response_model=ChunkIngestResponse,
    summary="批量写入知识分块",
    description="向指定 source_id 批量写入分块和向量；可选覆盖该来源下历史分块。",
)
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


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="检索知识分块",
    description="支持按业务元数据过滤，可选传入 query_embedding 进行向量相似度排序。",
)
async def retrieve(
    payload: RetrieveRequest,
    session: AsyncSession = Depends(get_db_session),
) -> RetrieveResponse:
    items = await _retrieve_items(payload, session)
    return RetrieveResponse(items=items)


@router.post(
    "/answer",
    response_model=AnswerResponse,
    summary="RAG回答",
    description="先召回证据分块，再调用DeepSeek生成结论与要点，返回可引用证据。",
)
async def answer(
    payload: AnswerRequest,
    session: AsyncSession = Depends(get_db_session),
) -> AnswerResponse:
    retrieve_payload = RetrieveRequest(
        query=payload.query,
        query_embedding=payload.query_embedding,
        top_k=payload.top_k,
        doc_type=payload.doc_type,
        source_type=payload.source_type,
        city_code=payload.city_code,
        lot_code=payload.lot_code,
        at_time=payload.at_time,
        source_ids=payload.source_ids,
        include_inactive=payload.include_inactive,
    )
    items = await _retrieve_items(retrieve_payload, session)
    if not items:
        return AnswerResponse(
            conclusion="未检索到可用证据，暂时无法回答该问题。",
            key_points=[],
            citations=[],
            retrieved_count=0,
            model=settings.deepseek_model,
        )

    try:
        conclusion, key_points, model_used = await generate_answer_from_chunks(payload.query, items)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    citations = [
        AnswerCitation(
            chunk_id=item.chunk_id,
            source_id=item.source_id,
            doc_type=item.doc_type,
            title=item.title,
            snippet=(item.content[:200] + "...") if len(item.content) > 200 else item.content,
            score=item.score,
        )
        for item in items
    ]
    return AnswerResponse(
        conclusion=conclusion,
        key_points=key_points,
        citations=citations,
        retrieved_count=len(items),
        model=model_used,
    )
