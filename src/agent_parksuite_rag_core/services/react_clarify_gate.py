from __future__ import annotations

from typing import Any, Awaitable, Callable, Literal

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.memory import SessionMemoryState
from agent_parksuite_rag_core.workflows.clarify_react_graph import run_clarify_react_once

ResolverDecision = Literal["continue_business", "clarify_biz", "clarify_react", "clarify_abort"]
RequiredSlotsResolver = Callable[[str | None], tuple[str, ...]]
LLMFactory = Callable[[], Any]


async def react_clarify_gate_async(
    *,
    parse_result: Any,
    hydrate_result: Any,
    memory_state: SessionMemoryState | None,
    llm_factory: LLMFactory,
    required_slots_for_intent: RequiredSlotsResolver,
    required_slots_override: list[str] | None = None,
    max_rounds: int = 3,
) -> tuple[
    ResolverDecision,
    HybridAnswerRequest,
    str | None,
    str | None,
    list[str],
    list[dict[str, Any]] | None,
]:
    trace: list[str] = []
    need_react = bool(hydrate_result.missing_required_slots) or parse_result.intent is None
    if "order_reference" in parse_result.ambiguities and hydrate_result.payload.order_no is None:
        need_react = True
        trace.append("react_clarify_gate_async:order_reference")

    if not need_react:
        return "continue_business", hydrate_result.payload, None, None, (trace or ["react_clarify_gate_async:pass"]), None

    trace.append("react_clarify_gate_async:enter_react")
    required_slots = (
        list(required_slots_override)
        if required_slots_override
        else list(required_slots_for_intent(parse_result.intent))
    )
    react_result = await run_clarify_react_once(
        payload=hydrate_result.payload,
        llm_factory=llm_factory,
        required_slots=required_slots,
        memory_state=memory_state,
        tools=[],
        max_rounds=max_rounds,
    )
    react_decision = str(react_result.get("decision", "clarify_react"))
    react_messages = react_result.get("messages")
    react_trace = list(react_result.get("trace", []))

    merged_payload = hydrate_result.payload.model_copy(update=dict(react_result.get("resolved_slots", {})))
    react_missing = list(react_result.get("missing_required_slots", []))
    if parse_result.intent is None:
        return (
            "clarify_react",
            merged_payload,
            react_result.get("clarify_question") or "请先确认你的问题类型：规则解释、欠费查询，还是订单金额核验？",
            "missing_intent",
            [*trace, *react_trace, "react_clarify_gate_async:pending_intent"],
            react_messages,
        )

    if react_decision == "continue_business" and not react_missing:
        return (
            "continue_business",
            merged_payload,
            None,
            None,
            [*trace, *react_trace, "react_clarify_gate_async:continue_business"],
            react_messages,
        )
    if react_decision == "clarify_abort":
        return (
            "clarify_abort",
            merged_payload,
            react_result.get("clarify_question") or "当前信息仍不足以继续，请补充关键信息后重试。",
            "clarify_abort",
            [*trace, *react_trace, "react_clarify_gate_async:abort"],
            react_messages,
        )
    return (
        "clarify_react",
        merged_payload,
        react_result.get("clarify_question") or "请补充必要信息后继续。",
        "clarify_react_required",
        [*trace, *react_trace, "react_clarify_gate_async:clarify_react"],
        react_messages,
    )
