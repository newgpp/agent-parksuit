from __future__ import annotations

from fastapi import APIRouter

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.answer import (
    ClarifyReactDebugRequest,
    ClarifyReactDebugResponse,
    HybridAnswerRequest,
    IntentSlotParseDebugResponse,
)
from agent_parksuite_rag_core.services.intent_slot_resolver import debug_clarify_react, debug_intent_slot_parse
from agent_parksuite_rag_core.services.memory import get_session_memory_repo

router = APIRouter(prefix="/api/v1", tags=["rag-core-debug"])


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
        messages=debug_result.messages,
        parsed_payload=debug_result.parsed_payload,
    )
