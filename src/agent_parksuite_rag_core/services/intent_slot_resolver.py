from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Literal

from agent_parksuite_common.llm_payload import dump_llm_input, dump_llm_output, trim_llm_payload_text
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from agent_parksuite_rag_core.clients.llm_client import get_chat_llm
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.memory import SessionMemoryState
from agent_parksuite_rag_core.services.react_clarify_gate import react_clarify_gate_async

_ORDER_NO_PATTERN = re.compile(r"\bSCN-\d+\b", re.IGNORECASE)
_ORDER_REF_TOKENS = ("上一单", "上一笔", "这笔", "这单", "第一笔")
_VALID_INTENTS = {"rule_explain", "arrears_check", "fee_verify"}
_SLOT_KEYS = ("city_code", "lot_code", "plate_no", "order_no", "at_time")
_MEMORY_HYDRATE_KEYS = ("city_code", "lot_code", "plate_no")

ResolverDecision = Literal["continue_business", "clarify_short_circuit", "clarify_react", "clarify_abort"]
FieldSource = Literal["user", "memory", "inferred"]


@dataclass(frozen=True)
class ResolvedTurnContext:
    """Resolver最终输出：用于驱动后续是继续业务流还是先澄清。"""

    # 经过解析/补槽后的请求对象（供后续Hybrid执行或澄清返回使用）
    payload: HybridAnswerRequest
    # 决策结果：继续业务流或先走业务澄清
    decision: ResolverDecision
    # 解析阶段轨迹（便于排障与审计）
    memory_trace: list[str]
    # 需要澄清时返回给用户的问题文本；可为空表示无需澄清
    clarify_reason: str | None
    # 需要澄清时的结构化错误码
    clarify_error: str | None = None
    # ReAct澄清链路的消息历史（用于后续多轮续接）
    clarify_messages: list[dict[str, Any]] | None = None
    # ReAct工具调用轨迹（用于调试/验收）
    clarify_tool_trace: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class IntentSlotParseResult:
    """阶段1产物：意图与槽位初步解析结果。"""

    # 解析后（含可能从query抽取槽位）的请求对象
    payload: HybridAnswerRequest
    # 识别到的意图；None表示当前无法确定
    intent: str | None
    # 意图置信度（可选）
    intent_confidence: float | None
    # 槽位来源标记：user/memory/inferred
    field_sources: dict[str, FieldSource]
    # 基于当前意图判断的必填缺失槽位
    missing_required_slots: list[str]
    # 解析出的歧义列表（如订单指代歧义）
    ambiguities: list[str]
    # 本阶段轨迹
    trace: list[str]


@dataclass(frozen=True)
class SlotHydrateResult:
    """阶段2产物：结合会话记忆补槽后的结果。"""

    # 补槽后的请求对象
    payload: HybridAnswerRequest
    # 补槽后字段来源标记
    field_sources: dict[str, FieldSource]
    # 补槽后仍缺失的必填槽位
    missing_required_slots: list[str]
    # 本阶段轨迹
    trace: list[str]


@dataclass(frozen=True)
class ClarifyReactDebugResult:
    decision: ResolverDecision
    clarify_question: str | None
    clarify_error: str | None
    resolved_slots: dict[str, Any]
    missing_required_slots: list[str]
    trace: list[str]
    tool_trace: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    parsed_payload: HybridAnswerRequest
    intent: str | None


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


def _intent_slot_parse_deterministic(
    payload: HybridAnswerRequest,
) -> IntentSlotParseResult:
    # Deterministic fallback parser used when LLM is unavailable or failed.
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


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _intent_slot_parse(payload: HybridAnswerRequest) -> IntentSlotParseResult:
    # Step-1: intent_slot_parse (real LLM call + deterministic fallback)
    # 主路径：一次LLM调用统一输出 intent + 槽位抽取 + 缺参/歧义信号；
    # 回退：LLM不可用/输出异常时使用确定性解析。
    deterministic = _intent_slot_parse_deterministic(payload)
    if deterministic.intent in _VALID_INTENTS:
        return deterministic
    llm = get_chat_llm(temperature=0, timeout_seconds=8)
    messages = [
        SystemMessage(
            content=(
                "你是停车业务意图和槽位解析器。"
                '请只输出JSON，格式: {"intent":"rule_explain|arrears_check|fee_verify|unknown",'
                '"intent_confidence":0~1,"slots":{"order_no":string|null,"plate_no":string|null,'
                '"city_code":string|null,"lot_code":string|null},"ambiguities":[string,...]}。'
            )
        ),
        HumanMessage(content=f"用户问题: {payload.query}"),
    ]
    logger.info(
        "llm[intent_slot_parse] input query={} hint={} model={}",
        payload.query[:200],
        payload.intent_hint,
        settings.deepseek_model,
    )
    logger.info(
        "llm[intent_slot_parse] input_payload={}",
        trim_llm_payload_text(
            dump_llm_input(messages=messages, model=settings.deepseek_model, temperature=0),
            full_payload=settings.llm_log_full_payload,
            max_chars=settings.llm_log_max_chars,
        ),
    )
    try:
        result = await llm.ainvoke(messages)
    except Exception as exc:
        logger.warning("intent_slot_parse llm_error fallback=deterministic error={}", exc.__class__.__name__)
        return IntentSlotParseResult(
            payload=deterministic.payload,
            intent=deterministic.intent,
            intent_confidence=deterministic.intent_confidence,
            field_sources=deterministic.field_sources,
            missing_required_slots=deterministic.missing_required_slots,
            ambiguities=deterministic.ambiguities,
            trace=[*deterministic.trace, "intent_slot_parse:llm_error_fallback"],
        )
    raw_text = str(result.content)
    logger.info(
        "llm[intent_slot_parse] output_payload={}",
        trim_llm_payload_text(
            dump_llm_output(result=result, model=settings.deepseek_model, temperature=0),
            full_payload=settings.llm_log_full_payload,
            max_chars=settings.llm_log_max_chars,
        ),
    )
    parsed = _extract_json_payload(raw_text)
    if not parsed:
        logger.info("llm[intent_slot_parse] parse_result=invalid_json fallback=deterministic")
        return IntentSlotParseResult(
            payload=deterministic.payload,
            intent=deterministic.intent,
            intent_confidence=deterministic.intent_confidence,
            field_sources=deterministic.field_sources,
            missing_required_slots=deterministic.missing_required_slots,
            ambiguities=deterministic.ambiguities,
            trace=[*deterministic.trace, "intent_slot_parse:llm_invalid_json_fallback"],
        )

    llm_intent = str(parsed.get("intent", "")).strip()
    intent = llm_intent if llm_intent in _VALID_INTENTS else deterministic.intent
    intent_conf = parsed.get("intent_confidence", deterministic.intent_confidence)
    try:
        intent_confidence = float(intent_conf) if intent_conf is not None else deterministic.intent_confidence
    except Exception:
        intent_confidence = deterministic.intent_confidence

    field_sources = dict(deterministic.field_sources)
    updates: dict[str, Any] = {}
    slots_obj = parsed.get("slots", {})
    if isinstance(slots_obj, dict):
        for key in ("order_no", "plate_no", "city_code", "lot_code"):
            if getattr(deterministic.payload, key) is None and slots_obj.get(key):
                updates[key] = str(slots_obj.get(key))
                field_sources[key] = "inferred"
    merged_payload = deterministic.payload.model_copy(update=updates) if updates else deterministic.payload

    ambiguities = list(deterministic.ambiguities)
    llm_ambiguities = parsed.get("ambiguities", [])
    if isinstance(llm_ambiguities, list):
        for item in llm_ambiguities:
            label = str(item).strip()
            if label and label not in ambiguities:
                ambiguities.append(label)

    missing_required_slots = [slot for slot in _required_slots_for_intent(intent) if getattr(merged_payload, slot) is None]
    logger.info(
        "llm[intent_slot_parse] parse_result=json intent={} missing_required_slots={} ambiguities={}",
        intent,
        missing_required_slots,
        ambiguities,
    )
    return IntentSlotParseResult(
        payload=merged_payload,
        intent=intent,
        intent_confidence=intent_confidence,
        field_sources=field_sources,
        missing_required_slots=missing_required_slots,
        ambiguities=ambiguities,
        trace=[*deterministic.trace, "intent_slot_parse:llm"],
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


async def resolve_turn_context_async(
    payload: HybridAnswerRequest,
    memory_state: SessionMemoryState | None,
) -> ResolvedTurnContext:
    parse_result = await _intent_slot_parse(payload=payload)
    hydrate_result = _slot_hydrate(parse_result=parse_result, payload=parse_result.payload, memory_state=memory_state)
    decision, payload_out, clarify_reason, clarify_error, gate_trace, tool_trace, clarify_messages = await react_clarify_gate_async(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=memory_state,
        llm_factory=lambda: get_chat_llm(temperature=0, timeout_seconds=8),
        required_slots_for_intent=_required_slots_for_intent,
    )
    trace = [*parse_result.trace, *hydrate_result.trace, *gate_trace]
    return ResolvedTurnContext(
        payload=payload_out,
        decision=decision,
        memory_trace=trace,
        clarify_reason=clarify_reason,
        clarify_error=clarify_error,
        clarify_messages=clarify_messages,
        clarify_tool_trace=tool_trace,
    )


async def debug_intent_slot_parse(payload: HybridAnswerRequest) -> IntentSlotParseResult:
    """Debug helper: run only resolver step-1 intent/slot parse."""
    return await _intent_slot_parse(payload=payload)


async def debug_clarify_react(
    payload: HybridAnswerRequest,
    memory_state: SessionMemoryState | None,
    *,
    required_slots: list[str] | None = None,
    max_rounds: int = 3,
) -> ClarifyReactDebugResult:
    parse_result = await _intent_slot_parse(payload=payload)
    hydrate_result = _slot_hydrate(parse_result=parse_result, payload=parse_result.payload, memory_state=memory_state)
    decision, payload_out, clarify_reason, clarify_error, gate_trace, tool_trace, clarify_messages = await react_clarify_gate_async(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=memory_state,
        llm_factory=lambda: get_chat_llm(temperature=0, timeout_seconds=8),
        required_slots_for_intent=_required_slots_for_intent,
        required_slots_override=required_slots,
        max_rounds=max_rounds,
    )
    resolved_slots = build_request_slots(payload_out)
    missing_required_slots = [
        slot
        for slot in (required_slots or list(_required_slots_for_intent(parse_result.intent)))
        if getattr(payload_out, slot, None) is None
    ]
    trace = [*parse_result.trace, *hydrate_result.trace, *gate_trace]
    return ClarifyReactDebugResult(
        decision=decision,
        clarify_question=clarify_reason,
        clarify_error=clarify_error,
        resolved_slots=resolved_slots,
        missing_required_slots=missing_required_slots,
        trace=trace,
        tool_trace=tool_trace or [],
        messages=clarify_messages or [],
        parsed_payload=payload_out,
        intent=parse_result.intent,
    )
