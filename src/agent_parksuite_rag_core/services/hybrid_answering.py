from __future__ import annotations

from typing import Any, Awaitable, Callable

from loguru import logger

from agent_parksuite_rag_core.clients.biz_api_client import get_biz_client
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.schemas.retrieve import RetrieveResponseItem
from agent_parksuite_rag_core.services.answering import generate_hybrid_answer
from agent_parksuite_rag_core.services.memory import SessionMemoryState, get_session_memory_repo
from agent_parksuite_rag_core.services.intent_slot_resolver import ResolvedTurnContext, resolve_turn_context_async
from agent_parksuite_rag_core.tools.biz_fact_tools import BizExecutionContext, BizFactTools
from agent_parksuite_rag_core.workflows.hybrid_answer import (
    HybridExecutionContext,
    HybridGraphState,
    run_hybrid_workflow,
)

RetrieveFn = Callable[[HybridExecutionContext], Awaitable[list[RetrieveResponseItem]]]
_VALID_INTENTS = {"rule_explain", "arrears_check", "fee_verify"}


def _to_execution_context(payload: HybridAnswerRequest, resolved_intent: str | None) -> HybridExecutionContext:
    return HybridExecutionContext(
        query=payload.query,
        intent_hint=resolved_intent,
        query_embedding=payload.query_embedding,
        top_k=payload.top_k,
        doc_type=payload.doc_type,
        source_type=payload.source_type,
        city_code=payload.city_code,
        lot_code=payload.lot_code,
        at_time=payload.at_time,
        source_ids=payload.source_ids,
        include_inactive=payload.include_inactive,
        plate_no=payload.plate_no,
        order_no=payload.order_no,
        rule_code=payload.rule_code,
        entry_time=payload.entry_time,
        exit_time=payload.exit_time,
    )


async def _persist_session_memory(
    payload: HybridAnswerRequest,
    result: HybridGraphState,
    previous_state: SessionMemoryState | None,
    *,
    pending_clarification: dict[str, Any] | None = None,
    clarify_messages: list[dict[str, Any]] | None = None,
    resolved_slots_override: dict[str, Any] | None = None,
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
    resolved_slots = resolved_slots_override
    if not isinstance(resolved_slots, dict):
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
    if isinstance(pending_clarification, dict):
        new_state["pending_clarification"] = pending_clarification
        if isinstance(clarify_messages, list) and clarify_messages:
            new_state["clarify_messages"] = clarify_messages[-settings.memory_max_clarify_messages :]
    if isinstance(resolved_slots, dict):
        new_state["resolved_slots"] = resolved_slots
    await repo.save_session(session_id, new_state, settings.memory_ttl_seconds)


def _build_clarify_key_points(clarify_error: str) -> list[str]:
    if clarify_error == "missing_plate_no":
        return ["请提供车牌号（plate_no），例如 沪A12345。"]
    if clarify_error == "missing_intent":
        return ["请先确认问题类型：规则解释、欠费查询，或订单金额核验。"]
    return ["请提供明确的订单号（order_no），例如 SCN-020。"]


def _build_clarify_result(resolved: ResolvedTurnContext, resolved_intent: str | None, memory_trace: list[str]) -> HybridGraphState:
    clarify_error = resolved.clarify_error or "clarification_required"
    clarified_intent = resolved_intent or resolved.payload.intent_hint or ""
    return {
        "intent": clarified_intent,
        "retrieved_items": [],
        "business_facts": {
            "intent": clarified_intent,
            "error": clarify_error,
        },
        "conclusion": resolved.clarify_reason or "请补充必要信息后继续。",
        "key_points": _build_clarify_key_points(clarify_error),
        "model": "",
        "trace": [*memory_trace, f"answer_synthesizer:{resolved.decision}"],
    }


def _build_missing_intent_contract_result(memory_trace: list[str]) -> HybridGraphState:
    return {
        "intent": "",
        "retrieved_items": [],
        "business_facts": {
            "error": "missing_intent_contract",
        },
        "conclusion": "当前意图尚未收敛，请补充关键信息后继续。",
        "key_points": ["请先明确问题类型：规则解释、欠费查询或金额核验。"],
        "model": "",
        "trace": [*memory_trace, "intent_router:missing_intent_contract"],
    }


async def _run_business_workflow(
    *,
    execution_ctx: HybridExecutionContext,
    resolved_intent: str,
    retrieve_fn: RetrieveFn,
) -> HybridGraphState:
    biz_client = get_biz_client()
    fact_tools = BizFactTools(biz_client=biz_client)

    async def _arrears_facts_fn(p: HybridExecutionContext) -> dict[str, Any]:
        return await fact_tools.build_arrears_facts(
            BizExecutionContext(
                city_code=p.city_code,
                lot_code=p.lot_code,
                plate_no=p.plate_no,
                order_no=p.order_no,
                rule_code=p.rule_code,
                entry_time=p.entry_time,
                exit_time=p.exit_time,
            )
        )

    async def _fee_facts_fn(p: HybridExecutionContext) -> dict[str, Any]:
        return await fact_tools.build_fee_verify_facts(
            BizExecutionContext(
                city_code=p.city_code,
                lot_code=p.lot_code,
                plate_no=p.plate_no,
                order_no=p.order_no,
                rule_code=p.rule_code,
                entry_time=p.entry_time,
                exit_time=p.exit_time,
            )
        )

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

    async def _classify_fn(_p: HybridExecutionContext) -> str:
        # Downstream routing consumes resolver/clarify contract only.
        return resolved_intent

    return await run_hybrid_workflow(
        payload=execution_ctx,
        retrieve_fn=retrieve_fn,
        classify_fn=_classify_fn,
        arrears_facts_fn=_arrears_facts_fn,
        fee_facts_fn=_fee_facts_fn,
        synthesize_fn=_synthesize_fn,
    )

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
    execution_ctx = _to_execution_context(payload, resolved_intent)
    resolved_slots_ctx = (resolved.execution_context.slots if resolved.execution_context else {})
    memory_trace: list[str] = resolved.memory_trace
    memory_resolved_slots: dict[str, Any] = {
        key: value for key, value in resolved_slots_ctx.items() if value is not None
    }
    if resolved.decision in {"clarify_short_circuit", "clarify_react", "clarify_abort"} and resolved.clarify_reason:
        result = _build_clarify_result(resolved, resolved_intent, memory_trace)
        clarify_error = resolved.clarify_error or "clarification_required"
        memory_pending_clarification = {
            "decision": resolved.decision,
            "error": clarify_error,
        }
        if payload.session_id:
            await _persist_session_memory(
                payload,
                result,
                memory_state,
                pending_clarification=memory_pending_clarification,
                clarify_messages=resolved.clarify_messages or [],
                resolved_slots_override=memory_resolved_slots,
            )
        return result

    if resolved_intent not in _VALID_INTENTS:
        result = _build_missing_intent_contract_result(memory_trace)
        if payload.session_id:
            await _persist_session_memory(
                payload,
                result,
                memory_state,
                pending_clarification={"decision": "clarify_react", "error": "missing_intent_contract"},
                clarify_messages=resolved.clarify_messages or [],
                resolved_slots_override=memory_resolved_slots,
            )
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
    result = await _run_business_workflow(
        execution_ctx=execution_ctx,
        resolved_intent=str(resolved_intent),
        retrieve_fn=retrieve_fn,
    )
    if payload.session_id:
        result["trace"] = [*memory_trace, *result.get("trace", []), "memory_persist"]
        await _persist_session_memory(
            payload,
            result,
            memory_state,
            pending_clarification=None,
            clarify_messages=None,
            resolved_slots_override=memory_resolved_slots,
        )
    return result
