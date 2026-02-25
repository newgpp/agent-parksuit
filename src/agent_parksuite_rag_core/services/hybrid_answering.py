from __future__ import annotations

import re
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
from agent_parksuite_rag_core.tools.biz_fact_tools import BizFactTools
from agent_parksuite_rag_core.workflows.hybrid_answer import HybridGraphState, run_hybrid_workflow

RetrieveFn = Callable[[HybridAnswerRequest], Awaitable[list[RetrieveResponseItem]]]
_ORDER_NO_PATTERN = re.compile(r"\bSCN-\d+\b", re.IGNORECASE)
_ORDER_REF_TOKENS = ("上一单", "上一笔", "这笔", "这单", "第一笔")
_INTENT_CARRY_VALUES = {"rule_explain", "arrears_check", "fee_verify"}
_FIRST_ORDER_REF_TOKENS = ("第一笔", "第一单")


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


def _extract_order_no_from_query(query: str) -> str | None:
    match = _ORDER_NO_PATTERN.search(query)
    if not match:
        return None
    return match.group(0).upper()


def _wants_order_reference(query: str) -> bool:
    return any(token in query for token in _ORDER_REF_TOKENS)


def _wants_first_order_reference(query: str) -> bool:
    return any(token in query for token in _FIRST_ORDER_REF_TOKENS)


def _looks_like_fee_verify_query(payload: HybridAnswerRequest) -> bool:
    query = payload.query
    if payload.order_no:
        return True
    return any(token in query for token in ("核验", "一致", "算错", "金额", "复核", "不对"))


def _apply_memory_hydrate(
    payload: HybridAnswerRequest,
    memory_state: SessionMemoryState | None,
) -> tuple[HybridAnswerRequest, list[str], str | None]:
    if not memory_state:
        return payload, ["memory_hydrate:none"], None

    traces: list[str] = []
    updates: dict[str, Any] = {}
    slots = dict(memory_state.get("slots", {}))

    for key in ("city_code", "lot_code", "plate_no"):
        if getattr(payload, key) is None and slots.get(key):
            updates[key] = slots[key]
            traces.append(f"memory_hydrate:{key}")

    should_force_fee_verify = _looks_like_fee_verify_query(payload) or _wants_order_reference(payload.query)
    if payload.intent_hint not in _INTENT_CARRY_VALUES:
        if should_force_fee_verify:
            updates["intent_hint"] = "fee_verify"
            traces.append("memory_hydrate:intent_hint_fee_verify")
        else:
            last_intent = str(memory_state.get("last_intent", "")).strip()
            if last_intent in _INTENT_CARRY_VALUES:
                updates["intent_hint"] = last_intent
                traces.append("memory_hydrate:intent_hint")

    if payload.order_no is None:
        from_query = _extract_order_no_from_query(payload.query)
        if from_query:
            updates["order_no"] = from_query
            traces.append("memory_hydrate:order_no_from_query")
        else:
            candidates = [str(x) for x in memory_state.get("order_candidates", []) if str(x)]
            if _wants_order_reference(payload.query):
                wants_first = _wants_first_order_reference(payload.query)
                if len(candidates) == 1 and not wants_first:
                    updates["order_no"] = candidates[0]
                    traces.append("memory_hydrate:order_no_from_reference")
                elif len(candidates) > 1 or (wants_first and len(candidates) >= 1):
                    clarify = "检测到候选订单，请明确订单号（order_no）后再核验金额。"
                    if "intent_hint" not in updates and payload.intent_hint not in _INTENT_CARRY_VALUES:
                        updates["intent_hint"] = "fee_verify"
                    hydrated = payload.model_copy(update=updates) if updates else payload
                    traces.append("memory_hydrate:order_reference_ambiguous")
                    return hydrated, traces, clarify
            elif len(candidates) == 1 and (payload.intent_hint == "fee_verify" or _looks_like_fee_verify_query(payload)):
                updates["order_no"] = candidates[0]
                traces.append("memory_hydrate:order_no_from_single_candidate")

    hydrated = payload.model_copy(update=updates) if updates else payload
    return hydrated, traces or ["memory_hydrate:hit"], None


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

    order_candidates = [str(x) for x in old.get("order_candidates", []) if str(x)]
    if isinstance(facts.get("arrears_order_nos"), list):
        order_candidates = [str(x) for x in facts["arrears_order_nos"] if str(x)]
    elif facts.get("order_no"):
        order_candidates = [str(facts["order_no"])]

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
        "last_intent": str(result.get("intent", old.get("last_intent", ""))),
        "order_candidates": order_candidates,
        "turns": turns,
    }
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
        payload, memory_trace, clarify_reason = _apply_memory_hydrate(payload, memory_state)
        if clarify_reason:
            result: HybridGraphState = {
                "intent": payload.intent_hint or "fee_verify",
                "retrieved_items": [],
                "business_facts": {
                    "intent": payload.intent_hint or "fee_verify",
                    "error": "order_reference_ambiguous",
                    "order_candidates": list((memory_state or {}).get("order_candidates", [])),
                },
                "conclusion": clarify_reason,
                "key_points": ["请提供明确的订单号（order_no），例如 SCN-020。"],
                "model": "",
                "trace": [*memory_trace, "answer_synthesizer:memory_clarify"],
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
