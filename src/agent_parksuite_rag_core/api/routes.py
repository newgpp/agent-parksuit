from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.db.session import get_db_session
from agent_parksuite_rag_core.repositories.knowledge import KnowledgeRepository
from agent_parksuite_rag_core.schemas.answer import (
    AnswerCitation,
    AnswerRequest,
    AnswerResponse,
    ClarifyReactDebugRequest,
    ClarifyReactDebugResponse,
    HybridAnswerRequest,
    HybridAnswerResponse,
    IntentSlotParseDebugResponse,
)
from agent_parksuite_rag_core.schemas.knowledge import (
    ChunkIngestRequest,
    ChunkIngestResponse,
    KnowledgeSourceResponse,
    KnowledgeSourceUpsertRequest,
)
from agent_parksuite_rag_core.schemas.retrieve import (
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResponseItem,
)
from agent_parksuite_rag_core.services.answering import generate_answer_from_chunks
from agent_parksuite_rag_core.services.hybrid_answering import run_hybrid_answering
from agent_parksuite_rag_core.workflows.hybrid_answer import HybridExecutionContext
from agent_parksuite_rag_core.services.intent_slot_resolver import debug_clarify_react, debug_intent_slot_parse
from agent_parksuite_rag_core.services.memory import get_session_memory_repo

router = APIRouter(prefix="/api/v1", tags=["rag-core"])


def _build_retrieve_request(payload: AnswerRequest | HybridAnswerRequest) -> RetrieveRequest:
    return RetrieveRequest(
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
    repo = KnowledgeRepository(session=session, embedding_dim=settings.embedding_dim)
    row = await repo.upsert_source(payload)
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
    repo = KnowledgeRepository(session=session, embedding_dim=settings.embedding_dim)
    try:
        source_pk, inserted_count = await repo.ingest_chunks(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChunkIngestResponse(source_pk=source_pk, inserted_count=inserted_count)


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
    repo = KnowledgeRepository(session=session, embedding_dim=settings.embedding_dim)
    try:
        items = await repo.retrieve(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    repo = KnowledgeRepository(session=session, embedding_dim=settings.embedding_dim)
    retrieve_payload = _build_retrieve_request(payload)
    try:
        items = await repo.retrieve(retrieve_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@router.post(
    "/answer/hybrid",
    response_model=HybridAnswerResponse,
    summary="混合回答（RAG + Biz工具）",
    description="规则分流后，执行 RAG 检索与业务工具调用，再综合生成可解释回答。",
)
async def answer_hybrid(
    payload: HybridAnswerRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HybridAnswerResponse:
    repo = KnowledgeRepository(session=session, embedding_dim=settings.embedding_dim)
    turn_id = (payload.turn_id or "").strip() or f"turn-{uuid4().hex[:12]}"
    payload = payload.model_copy(update={"turn_id": turn_id})
    logger.info(
        "hybrid request received session_id={} turn_id={} top_k={} hint={} source_ids={}",
        payload.session_id,
        turn_id,
        payload.top_k,
        payload.intent_hint,
        len(payload.source_ids or []),
    )

    async def _hybrid_retrieve(hybrid_payload: HybridExecutionContext) -> list[RetrieveResponseItem]:
        retrieve_payload = RetrieveRequest(
            query=hybrid_payload.query,
            query_embedding=hybrid_payload.query_embedding,
            top_k=hybrid_payload.top_k,
            doc_type=hybrid_payload.doc_type,
            source_type=hybrid_payload.source_type,
            city_code=hybrid_payload.city_code,
            lot_code=hybrid_payload.lot_code,
            at_time=hybrid_payload.at_time,
            source_ids=hybrid_payload.source_ids,
            include_inactive=hybrid_payload.include_inactive,
        )
        try:
            items = await repo.retrieve(retrieve_payload)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        logger.info("hybrid retrieve done count={}", len(items))
        return items

    try:
        result = await run_hybrid_answering(payload=payload, retrieve_fn=_hybrid_retrieve)
    except RuntimeError as exc:
        logger.exception("hybrid failed with runtime error")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    items = result.get("retrieved_items", [])
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
    response = HybridAnswerResponse(
        session_id=payload.session_id,
        turn_id=turn_id,
        memory_ttl_seconds=settings.memory_ttl_seconds,
        intent=str(result.get("intent", "")),
        conclusion=str(result.get("conclusion", "")),
        key_points=list(result.get("key_points", [])),
        business_facts=dict(result.get("business_facts", {})),
        citations=citations,
        retrieved_count=len(items),
        model=str(result.get("model", settings.deepseek_model)),
        graph_trace=list(result.get("trace", [])),
    )
    logger.info(
        "hybrid response ready session_id={} turn_id={} intent={} retrieved_count={} trace={}",
        response.session_id,
        response.turn_id,
        response.intent,
        response.retrieved_count,
        response.graph_trace,
    )
    return response


@router.post(
    "/debug/intent-slot-parse",
    response_model=IntentSlotParseDebugResponse,
    summary="调试：Step-1 意图与槽位解析",
    description="仅执行 resolver 的 Step-1（LLM one-shot + fallback），不调用后续业务工具链路。",
)
async def debug_intent_slot_parse_route(
    payload: HybridAnswerRequest,
) -> IntentSlotParseDebugResponse:
    parsed = await debug_intent_slot_parse(payload)
    return IntentSlotParseDebugResponse(
        intent=parsed.intent,
        intent_confidence=parsed.intent_confidence,
        field_sources={k: str(v) for k, v in parsed.field_sources.items()},
        missing_required_slots=parsed.missing_required_slots,
        ambiguities=parsed.ambiguities,
        trace=parsed.trace,
        parsed_payload=parsed.payload,
    )


@router.post(
    "/debug/clarify-react",
    response_model=ClarifyReactDebugResponse,
    summary="调试：ReAct澄清循环",
    description="执行 resolver 的 ReAct 澄清路径（含会话历史），不进入 hybrid 业务工具主链路。",
)
async def debug_clarify_react_route(
    payload: ClarifyReactDebugRequest,
) -> ClarifyReactDebugResponse:
    memory_state = None
    repo = get_session_memory_repo()
    session_id = (payload.session_id or "").strip()
    if session_id:
        memory_state = await repo.get_session(session_id)

    request_payload = HybridAnswerRequest(
        session_id=payload.session_id,
        query=payload.query,
        intent_hint=payload.intent,
        city_code=payload.city_code,
        lot_code=payload.lot_code,
        plate_no=payload.plate_no,
        order_no=payload.order_no,
        at_time=payload.at_time,
    )
    debug_result = await debug_clarify_react(
        payload=request_payload,
        memory_state=memory_state,
        required_slots=payload.required_slots,
        max_rounds=payload.max_rounds,
    )

    if session_id:
        old = memory_state or {}
        slots = dict(old.get("slots", {}))
        for key, value in debug_result.resolved_slots.items():
            if value is not None:
                slots[key] = value
        new_state = dict(old)
        new_state["slots"] = slots
        new_state["clarify_messages"] = debug_result.messages
        new_state["pending_clarification"] = {
            "decision": debug_result.decision,
            "error": debug_result.clarify_error,
            "missing_required_slots": debug_result.missing_required_slots,
        }
        new_state["resolved_slots"] = {
            key: value for key, value in debug_result.resolved_slots.items() if value is not None
        }
        await repo.save_session(session_id, new_state, settings.memory_ttl_seconds)

    return ClarifyReactDebugResponse(
        decision=debug_result.decision,
        intent=debug_result.intent,
        clarify_question=debug_result.clarify_question,
        clarify_error=debug_result.clarify_error,
        resolved_slots=debug_result.resolved_slots,
        missing_required_slots=debug_result.missing_required_slots,
        trace=debug_result.trace,
        tool_trace=debug_result.tool_trace,
        messages=debug_result.messages,
        parsed_payload=debug_result.parsed_payload,
    )
