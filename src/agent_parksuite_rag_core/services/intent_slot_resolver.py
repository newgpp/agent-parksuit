from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.memory import SessionMemoryState

_ORDER_NO_PATTERN = re.compile(r"\bSCN-\d+\b", re.IGNORECASE)
_ORDER_REF_TOKENS = ("上一单", "上一笔", "这笔", "这单", "第一笔")
_FIRST_ORDER_REF_TOKENS = ("第一笔", "第一单")
_VALID_INTENTS = {"rule_explain", "arrears_check", "fee_verify"}
_SLOT_KEYS = ("city_code", "lot_code", "plate_no", "order_no", "at_time")
_MEMORY_HYDRATE_KEYS = ("city_code", "lot_code", "plate_no")

ResolverDecision = Literal["continue_business", "clarify_biz"]
FieldSource = Literal["user", "memory", "inferred"]


@dataclass(frozen=True)
class ResolvedTurnContext:
    payload: HybridAnswerRequest
    decision: ResolverDecision
    memory_trace: list[str]
    clarify_reason: str | None
    clarify_error: str | None = None


@dataclass(frozen=True)
class IntentSlotParseResult:
    payload: HybridAnswerRequest
    intent: str | None
    intent_confidence: float | None
    field_sources: dict[str, FieldSource]
    missing_required_slots: list[str]
    ambiguities: list[str]
    trace: list[str]


@dataclass(frozen=True)
class SlotHydrateResult:
    payload: HybridAnswerRequest
    field_sources: dict[str, FieldSource]
    missing_required_slots: list[str]
    trace: list[str]


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


def _required_slots_for_intent(intent: str | None) -> tuple[str, ...]:
    if intent == "fee_verify":
        return ("order_no",)
    if intent == "arrears_check":
        return ("plate_no",)
    return ()


def _build_field_sources(payload: HybridAnswerRequest) -> dict[str, FieldSource]:
    sources: dict[str, FieldSource] = {}
    for key in _SLOT_KEYS:
        if getattr(payload, key) is not None:
            sources[key] = "user"
    return sources


def _intent_slot_parse(
    payload: HybridAnswerRequest,
) -> IntentSlotParseResult:
    # Step-1: intent_slot_parse
    # 当前阶段使用确定性占位实现，后续将替换为一次LLM调用，
    # 统一输出 intent + 槽位抽取 + 缺参/歧义信号。
    trace = ["intent_slot_parse:deterministic"]
    updates: dict[str, Any] = {}
    field_sources = _build_field_sources(payload)

    intent_raw = (payload.intent_hint or "").strip()
    intent = intent_raw if intent_raw in _VALID_INTENTS else None
    intent_confidence = 1.0 if intent else None

    if payload.order_no is None:
        extracted_order_no = _extract_order_no_from_query(payload.query)
        if extracted_order_no:
            updates["order_no"] = extracted_order_no
            field_sources["order_no"] = "inferred"
            trace.append("intent_slot_parse:order_no_from_query")

    parsed_payload = payload.model_copy(update=updates) if updates else payload
    missing_required_slots = [
        slot for slot in _required_slots_for_intent(intent) if getattr(parsed_payload, slot) is None
    ]
    ambiguities: list[str] = []
    if parsed_payload.order_no is None and _wants_order_reference(parsed_payload.query):
        ambiguities.append("order_reference")
        trace.append("intent_slot_parse:order_reference")
    return IntentSlotParseResult(
        payload=parsed_payload,
        intent=intent,
        intent_confidence=intent_confidence,
        field_sources=field_sources,
        missing_required_slots=missing_required_slots,
        ambiguities=ambiguities,
        trace=trace,
    )


def _slot_hydrate(
    parse_result: IntentSlotParseResult,
    payload: HybridAnswerRequest,
    memory_state: SessionMemoryState | None,
) -> SlotHydrateResult:
    # Step-2: slot_hydrate
    # 无LLM的确定性补槽阶段：根据记忆补充当前轮缺失槽位，
    # 且不覆盖用户本轮显式输入（field_sources 中的 user）。
    if not memory_state:
        return SlotHydrateResult(
            payload=payload,
            field_sources=dict(parse_result.field_sources),
            missing_required_slots=list(parse_result.missing_required_slots),
            trace=["slot_hydrate:none"],
        )

    traces: list[str] = []
    updates: dict[str, Any] = {}
    field_sources: dict[str, FieldSource] = dict(parse_result.field_sources)
    slots = dict(memory_state.get("slots", {}))
    intent = parse_result.intent
    required_slots = set(_required_slots_for_intent(intent))
    for key in _MEMORY_HYDRATE_KEYS:
        if getattr(payload, key) is None and slots.get(key):
            # 当前阶段先保留已有稳定行为（通用槽位继承），后续可收敛为 required-only。
            updates[key] = slots[key]
            field_sources[key] = "memory"
            traces.append(f"slot_hydrate:{key}")
    for key in required_slots:
        if getattr(payload, key) is None and key in slots and slots.get(key):
            updates[key] = slots[key]
            field_sources[key] = "memory"
            traces.append(f"slot_hydrate:required:{key}")
    hydrated = payload.model_copy(update=updates) if updates else payload
    missing_required_slots = [slot for slot in _required_slots_for_intent(intent) if getattr(hydrated, slot) is None]
    return SlotHydrateResult(
        payload=hydrated,
        field_sources=field_sources,
        missing_required_slots=missing_required_slots,
        trace=traces or ["slot_hydrate:hit"],
    )


def _react_clarify_gate(
    parse_result: IntentSlotParseResult,
    hydrate_result: SlotHydrateResult,
) -> tuple[str | None, str | None, list[str]]:
    # Step-3: react_clarify_gate
    # 澄清门控占位：决定是否需要进入后续ReAct澄清链路。
    # PR-1阶段仅保留已上线的订单指代短路，其余场景先透传。
    trace: list[str] = []
    if "order_reference" in parse_result.ambiguities and hydrate_result.payload.order_no is None:
        trace.append("react_clarify_gate:order_reference")
        if _wants_first_order_reference(hydrate_result.payload.query):
            return (
                "检测到“第一笔”订单指代，请明确订单号（order_no）后再核验金额。",
                "order_reference_needs_clarification",
                trace,
            )
        return (
            "检测到订单指代，请明确订单号（order_no）后再核验金额。",
            "order_reference_needs_clarification",
            trace,
        )
    if parse_result.intent is None:
        # PR-1框架阶段: 仅记录状态，继续沿用现有意图分类链路，不做短路。
        trace.append("react_clarify_gate:pending_intent_passthrough")
    if hydrate_result.missing_required_slots:
        # PR-1框架阶段: 仅记录状态，继续沿用现有业务链路错误处理，不做短路。
        trace.append("react_clarify_gate:missing_required_slots_passthrough")
    return None, None, trace or ["react_clarify_gate:pass"]


def resolve_turn_context(
    payload: HybridAnswerRequest,
    memory_state: SessionMemoryState | None,
) -> ResolvedTurnContext:
    # Resolver 主入口：固定编排三阶段
    # intent_slot_parse -> slot_hydrate -> react_clarify_gate
    parse_result = _intent_slot_parse(payload=payload)
    hydrate_result = _slot_hydrate(parse_result=parse_result, payload=parse_result.payload, memory_state=memory_state)
    clarify_reason, clarify_error, gate_trace = _react_clarify_gate(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
    )
    trace = [*parse_result.trace, *hydrate_result.trace, *gate_trace]
    decision: ResolverDecision = "clarify_biz" if clarify_reason else "continue_business"
    return ResolvedTurnContext(
        payload=hydrate_result.payload,
        decision=decision,
        memory_trace=trace,
        clarify_reason=clarify_reason,
        clarify_error=clarify_error,
    )
