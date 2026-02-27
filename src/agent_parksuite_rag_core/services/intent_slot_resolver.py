from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.memory import SessionMemoryState

_ORDER_NO_PATTERN = re.compile(r"\bSCN-\d+\b", re.IGNORECASE)
_ORDER_REF_TOKENS = ("上一单", "上一笔", "这笔", "这单", "第一笔")
_FIRST_ORDER_REF_TOKENS = ("第一笔", "第一单")

ResolverDecision = Literal["continue_business", "clarify_biz"]


@dataclass(frozen=True)
class ResolvedTurnContext:
    payload: HybridAnswerRequest
    decision: ResolverDecision
    memory_trace: list[str]
    clarify_reason: str | None
    clarify_error: str | None = None


def build_request_slots(payload: HybridAnswerRequest) -> dict[str, Any]:
    return {
        "city_code": payload.city_code,
        "lot_code": payload.lot_code,
        "plate_no": payload.plate_no,
        "order_no": payload.order_no,
        "at_time": payload.at_time,
    }


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
) -> tuple[HybridAnswerRequest, list[str], str | None, str | None]:
    if not memory_state:
        return payload, ["memory_hydrate:none"], None, None

    traces: list[str] = []
    updates: dict[str, Any] = {}
    slots = dict(memory_state.get("slots", {}))

    for key in ("city_code", "lot_code", "plate_no"):
        if getattr(payload, key) is None and slots.get(key):
            updates[key] = slots[key]
            traces.append(f"memory_hydrate:{key}")

    should_force_fee_verify = _looks_like_fee_verify_query(payload) or _wants_order_reference(payload.query)
    if should_force_fee_verify and payload.intent_hint not in {"rule_explain", "arrears_check", "fee_verify"}:
        updates["intent_hint"] = "fee_verify"
        traces.append("memory_hydrate:intent_hint_fee_verify")

    if payload.order_no is None:
        from_query = _extract_order_no_from_query(payload.query)
        if from_query:
            updates["order_no"] = from_query
            traces.append("memory_hydrate:order_no_from_query")
        elif _wants_order_reference(payload.query):
            wants_first = _wants_first_order_reference(payload.query)
            clarify = "检测到订单指代，请明确订单号（order_no）后再核验金额。"
            if wants_first:
                clarify = "检测到“第一笔”订单指代，请明确订单号（order_no）后再核验金额。"
            hydrated = payload.model_copy(update=updates) if updates else payload
            traces.append("memory_hydrate:order_reference_needs_clarification")
            return hydrated, traces, clarify, "order_reference_needs_clarification"

    hydrated = payload.model_copy(update=updates) if updates else payload
    return hydrated, traces or ["memory_hydrate:hit"], None, None


def resolve_turn_context(
    payload: HybridAnswerRequest,
    memory_state: SessionMemoryState | None,
) -> ResolvedTurnContext:
    hydrated_payload, trace, clarify_reason, clarify_error = _apply_memory_hydrate(payload, memory_state)
    decision: ResolverDecision = "clarify_biz" if clarify_reason else "continue_business"
    return ResolvedTurnContext(
        payload=hydrated_payload,
        decision=decision,
        memory_trace=trace,
        clarify_reason=clarify_reason,
        clarify_error=clarify_error,
    )
