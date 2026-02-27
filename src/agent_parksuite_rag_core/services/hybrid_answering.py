from __future__ import annotations

from typing import Any, Awaitable, Callable

from loguru import logger

from agent_parksuite_rag_core.clients.biz_api_client import get_biz_client
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.schemas.retrieve import RetrieveResponseItem
from agent_parksuite_rag_core.services.answering import generate_hybrid_answer
from agent_parksuite_rag_core.services.memory import SessionMemoryState, get_session_memory_repo
from agent_parksuite_rag_core.services.intent_slot_resolver import resolve_turn_context_async
from agent_parksuite_rag_core.tools.biz_fact_tools import BizExecutionContext, BizFactTools
from agent_parksuite_rag_core.workflows.hybrid_answer import HybridGraphState, run_hybrid_workflow

RetrieveFn = Callable[[HybridAnswerRequest], Awaitable[list[RetrieveResponseItem]]]


async def _persist_session_memory(
    payload: HybridAnswerRequest,
    result: HybridGraphState,
    previous_state: SessionMemoryState | None,
) -> None:
    session_id = (payload.session_id or "").strip()
    if not session_id:
        return

    repo = get_session_memory_repo()
    old = previous_state or {}
    slots: dict[str, Any] = dict(old.get("slots", {}))
    for key in ("city_code", "lot_code", "plate_no", "order_no", "at_time"):
        value = getattr(payload, key)
        if value is not None:
            slots[key] = value

    facts = dict(result.get("business_facts", {}))
    if facts.get("order_no"):
        slots["order_no"] = facts["order_no"]
    if facts.get("plate_no"):
        slots["plate_no"] = facts["plate_no"]
    if facts.get("city_code"):
        slots["city_code"] = facts["city_code"]
    resolved_slots = facts.get("resolved_slots", {})
    if isinstance(resolved_slots, dict):
        for key, value in resolved_slots.items():
            if value is not None:
                slots[key] = value

    turns = list(old.get("turns", []))
    turns.append(
        {
            "turn_id": payload.turn_id,
            "query": payload.query,
            "intent": result.get("intent", ""),
            "order_no": slots.get("order_no"),
        }
    )
    if len(turns) > settings.memory_max_turns:
        turns = turns[-settings.memory_max_turns :]

    new_state: SessionMemoryState = {
        "slots": slots,
        "turns": turns,
    }
    clarify_messages = facts.get("clarify_messages")
    if isinstance(clarify_messages, list):
        new_state["clarify_messages"] = clarify_messages
    pending_clarification = facts.get("pending_clarification")
    if isinstance(pending_clarification, dict):
        new_state["pending_clarification"] = pending_clarification
    if isinstance(resolved_slots, dict):
        new_state["resolved_slots"] = resolved_slots
    await repo.save_session(session_id, new_state, settings.memory_ttl_seconds)


async def _classify_intent(payload: HybridAnswerRequest) -> str:
    # Intent authority is resolver/ReAct stage; hybrid workflow consumes resolved intent only.
    if payload.intent_hint in {"rule_explain", "arrears_check", "fee_verify"}:
        logger.info("hybrid classify source=resolver_resolved_intent intent={}", payload.intent_hint)
        return payload.intent_hint

    logger.warning("hybrid classify source=resolver_fallback intent=rule_explain")
    return "rule_explain"


async def run_hybrid_answering(
    payload: HybridAnswerRequest,
    retrieve_fn: RetrieveFn,
) -> HybridGraphState:
    memory_state: SessionMemoryState | None = None
    if payload.session_id:
        memory_state = await get_session_memory_repo().get_session(payload.session_id)
    resolved = await resolve_turn_context_async(payload=payload, memory_state=memory_state)
    payload = resolved.payload
    resolved_intent = resolved.resolved_intent
    resolved_slots_ctx = (resolved.execution_context.slots if resolved.execution_context else {})
    memory_trace: list[str] = resolved.memory_trace
    if resolved.decision in {"clarify_short_circuit", "clarify_react", "clarify_abort"} and resolved.clarify_reason:
        clarified_intent = resolved_intent or payload.intent_hint or ""
        clarify_error = resolved.clarify_error or "clarification_required"
        key_points = ["请提供明确的订单号（order_no），例如 SCN-020。"]
        if clarify_error == "missing_plate_no":
            key_points = ["请提供车牌号（plate_no），例如 沪A12345。"]
        if clarify_error == "missing_intent":
            key_points = ["请先确认问题类型：规则解释、欠费查询，或订单金额核验。"]
        result: HybridGraphState = {
            "intent": clarified_intent,
            "retrieved_items": [],
            "business_facts": {
                "intent": clarified_intent,
                "error": clarify_error,
                "clarify_messages": resolved.clarify_messages or [],
                "clarify_tool_trace": resolved.clarify_tool_trace or [],
                "pending_clarification": {
                    "decision": resolved.decision,
                    "error": clarify_error,
                },
                "resolved_slots": {
                    key: value
                    for key, value in resolved_slots_ctx.items()
                    if value is not None
                },
            },
            "conclusion": resolved.clarify_reason,
            "key_points": key_points,
            "model": "",
            "trace": [*memory_trace, f"answer_synthesizer:{resolved.decision}"],
        }
        if payload.session_id:
            await _persist_session_memory(payload, result, memory_state)
        return result

    logger.info(
        "hybrid start session_id={} turn_id={} query_len={} top_k={} hint={} city_code={} lot_code={}",
        payload.session_id,
        payload.turn_id,
        len(payload.query),
        payload.top_k,
        payload.intent_hint,
        payload.city_code,
        payload.lot_code,
    )
    biz_client = get_biz_client()
    fact_tools = BizFactTools(biz_client=biz_client)
    biz_execution_ctx = BizExecutionContext(
        city_code=payload.city_code,
        lot_code=payload.lot_code,
        plate_no=payload.plate_no,
        order_no=payload.order_no,
        rule_code=payload.rule_code,
        entry_time=payload.entry_time,
        exit_time=payload.exit_time,
    )

    async def _arrears_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await fact_tools.build_arrears_facts(biz_execution_ctx)

    async def _fee_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await fact_tools.build_fee_verify_facts(biz_execution_ctx)

    async def _synthesize_fn(
        query: str,
        items: list[RetrieveResponseItem],
        business_facts: dict[str, Any],
        intent: str,
    ) -> tuple[str, list[str], str]:
        return await generate_hybrid_answer(
            query=query,
            items=items,
            business_facts=business_facts,
            intent=intent,
        )

    async def _classify_fn(p: HybridAnswerRequest) -> str:
        if resolved_intent in {"rule_explain", "arrears_check", "fee_verify"}:
            return resolved_intent
        return await _classify_intent(p)

    result = await run_hybrid_workflow(
        payload=payload,
        retrieve_fn=retrieve_fn,
        classify_fn=_classify_fn,
        arrears_facts_fn=_arrears_facts_fn,
        fee_facts_fn=_fee_facts_fn,
        synthesize_fn=_synthesize_fn,
    )
    if payload.session_id:
        result["trace"] = [*memory_trace, *result.get("trace", []), "memory_persist"]
        await _persist_session_memory(payload, result, memory_state)
    return result
