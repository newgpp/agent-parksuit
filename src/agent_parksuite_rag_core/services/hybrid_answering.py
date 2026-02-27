from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from agent_parksuite_rag_core.clients.biz_api_client import BizApiClient
from agent_parksuite_rag_core.clients.llm_client import get_chat_llm
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.schemas.retrieve import RetrieveResponseItem
from agent_parksuite_rag_core.services.answering import _extract_json_payload, generate_hybrid_answer
from agent_parksuite_rag_core.services.memory import SessionMemoryState, get_session_memory_repo
from agent_parksuite_rag_core.services.intent_slot_resolver import resolve_turn_context_async
from agent_parksuite_rag_core.tools.biz_fact_tools import BizFactTools
from agent_parksuite_rag_core.workflows.hybrid_answer import HybridGraphState, run_hybrid_workflow

RetrieveFn = Callable[[HybridAnswerRequest], Awaitable[list[RetrieveResponseItem]]]


def _log_payload_text(text: str) -> str:
    if settings.llm_log_full_payload:
        return text
    return text[: settings.llm_log_max_chars]


def _rule_route_intent(payload: HybridAnswerRequest) -> str:
    if payload.intent_hint in {"rule_explain", "arrears_check", "fee_verify"}:
        return payload.intent_hint

    query = payload.query
    if payload.order_no or any(token in query for token in ("核验", "一致", "算错", "金额", "订单")):
        return "fee_verify"
    if payload.plate_no or any(token in query for token in ("欠费", "补缴", "未缴", "车牌")):
        return "arrears_check"
    return "rule_explain"


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
    if payload.intent_hint in {"rule_explain", "arrears_check", "fee_verify"}:
        logger.info("hybrid classify source=intent_hint intent={}", payload.intent_hint)
        return payload.intent_hint

    if not settings.deepseek_api_key:
        intent = _rule_route_intent(payload)
        logger.info("hybrid classify source=rule_fallback reason=no_api_key intent={}", intent)
        return intent

    llm = get_chat_llm(temperature=0, timeout_seconds=8)

    messages = [
        SystemMessage(
            content=(
                "你是停车业务意图分类器。"
                '请只输出JSON: {"intent": "rule_explain|arrears_check|fee_verify"}。'
                "不要输出其他字段。"
            )
        ),
        HumanMessage(
            content=(
                "根据用户问题选择一个最合适意图:\n"
                "- rule_explain: 解释计费规则/政策\n"
                "- arrears_check: 查询是否欠费/欠费订单\n"
                "- fee_verify: 针对订单金额核验、重算、对账\n\n"
                f"用户问题: {payload.query}"
            )
        ),
    ]
    logger.info("llm[intent] input query={}", payload.query[:200])
    logger.info("llm[intent] input_prompt={}", _log_payload_text(messages[1].content))

    try:
        result = await llm.ainvoke(messages)
    except Exception as exc:
        intent = _rule_route_intent(payload)
        logger.warning(
            "hybrid classify source=rule_fallback reason=llm_error intent={} error={}",
            intent,
            exc.__class__.__name__,
        )
        return intent

    raw_text = str(result.content)
    logger.info("llm[intent] output raw={}", _log_payload_text(raw_text))
    parsed = _extract_json_payload(raw_text)
    intent = str((parsed or {}).get("intent", "")).strip()
    if intent in {"rule_explain", "arrears_check", "fee_verify"}:
        logger.info("hybrid classify source=llm intent={}", intent)
        return intent
    fallback_intent = _rule_route_intent(payload)
    logger.info(
        "hybrid classify source=rule_fallback reason=invalid_output intent={}",
        fallback_intent,
    )
    return fallback_intent


async def run_hybrid_answering(
    payload: HybridAnswerRequest,
    retrieve_fn: RetrieveFn,
) -> HybridGraphState:
    memory_state: SessionMemoryState | None = None
    memory_trace: list[str] = []
    if payload.session_id:
        memory_state = await get_session_memory_repo().get_session(payload.session_id)
        resolved = await resolve_turn_context_async(payload=payload, memory_state=memory_state)
        payload = resolved.payload
        memory_trace = resolved.memory_trace
        if resolved.decision in {"clarify_biz", "clarify_react", "clarify_abort"} and resolved.clarify_reason:
            clarified_intent = payload.intent_hint or ""
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
                    "pending_clarification": {
                        "decision": resolved.decision,
                        "error": clarify_error,
                    },
                    "resolved_slots": {
                        key: getattr(resolved.payload, key)
                        for key in ("city_code", "lot_code", "plate_no", "order_no", "at_time")
                        if getattr(resolved.payload, key) is not None
                    },
                },
                "conclusion": resolved.clarify_reason,
                "key_points": key_points,
                "model": "",
                "trace": [*memory_trace, f"answer_synthesizer:{resolved.decision}"],
            }
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
    biz_client = BizApiClient(
        base_url=settings.biz_api_base_url,
        timeout_seconds=settings.biz_api_timeout_seconds,
    )
    fact_tools = BizFactTools(biz_client=biz_client)

    async def _arrears_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await fact_tools.build_arrears_facts(p)

    async def _fee_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await fact_tools.build_fee_verify_facts(p)

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
