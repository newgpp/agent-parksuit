from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.clarify_agent import (
    ClarifyAgent,
    ClarifyResult,
    ClarifyTask,
    ReActClarifyAgent,
)
from agent_parksuite_rag_core.services.memory import SessionMemoryState

# Resolver决策枚举：
# continue_business=继续业务执行，clarify_short_circuit=规则短路澄清，clarify_react=进入ReAct澄清，clarify_abort=澄清终止。
ResolverDecision = Literal["continue_business", "clarify_short_circuit", "clarify_react", "clarify_abort"]
RequiredSlotsResolver = Callable[[str | None], tuple[str, ...]]
LLMFactory = Callable[[], Any]
_VALID_INTENTS = {"rule_explain", "arrears_check", "fee_verify"}


@dataclass(frozen=True)
class ReactClarifyGateResult:
    decision: ResolverDecision
    payload: HybridAnswerRequest
    clarify_reason: str | None
    clarify_error: str | None
    trace: list[str]
    clarify_messages: list[dict[str, Any]] | None


def _should_enter_react(parse_result: Any, hydrate_result: Any) -> tuple[bool, list[str]]:
    trace: list[str] = []
    # Gate rule:
    # 1) intent unknown -> enter ReAct for intent clarification
    # 2) intent known but required slots missing -> do deterministic short-circuit clarify
    # 3) otherwise continue business
    if parse_result.intent is None:
        trace.append("react_clarify_gate_async:need_react:missing_intent")
        return True, trace
    if hydrate_result.missing_required_slots:
        trace.append("react_clarify_gate_async:need_react:missing_required_slots")
        return True, trace
    return False, trace


def _short_circuit_if_possible(parse_result: Any, hydrate_result: Any, trace: list[str]) -> ReactClarifyGateResult | None:
    # Fast-path deterministic short-circuit:
    # if intent is already clear and only required slots are missing,
    # return clarify_short_circuit directly without entering ReAct/LLM.
    if parse_result.intent is None or not hydrate_result.missing_required_slots:
        return None

    missing = list(hydrate_result.missing_required_slots)
    if "order_no" in missing:
        return ReactClarifyGateResult(
            decision="clarify_short_circuit",
            payload=hydrate_result.payload,
            clarify_reason="请提供要核验的订单号（order_no，例如 SCN-020）。",
            clarify_error="missing_order_no",
            trace=[*trace, "react_clarify_gate_async:short_circuit:missing_order_no"],
            clarify_messages=None,
        )
    if "plate_no" in missing:
        return ReactClarifyGateResult(
            decision="clarify_short_circuit",
            payload=hydrate_result.payload,
            clarify_reason="请提供要查询欠费的车牌号（plate_no，例如 沪A12345）。",
            clarify_error="missing_plate_no",
            trace=[*trace, "react_clarify_gate_async:short_circuit:missing_plate_no"],
            clarify_messages=None,
        )
    return ReactClarifyGateResult(
        decision="clarify_short_circuit",
        payload=hydrate_result.payload,
        clarify_reason="请补充必要信息后继续。",
        clarify_error="missing_required_slots",
        trace=[*trace, "react_clarify_gate_async:short_circuit:missing_required_slots"],
        clarify_messages=None,
    )


async def _invoke_react_once(
    *,
    parse_result: Any,
    hydrate_result: Any,
    memory_state: SessionMemoryState | None,
    llm_factory: LLMFactory,
    required_slots_for_intent: RequiredSlotsResolver,
    required_slots_override: list[str] | None,
    max_rounds: int,
    trace: list[str],
    clarify_agent: ClarifyAgent,
) -> tuple[ClarifyResult | None, ReactClarifyGateResult | None]:
    required_slots = (
        list(required_slots_override)
        if required_slots_override
        else list(required_slots_for_intent(parse_result.intent))
    )
    try:
        clarify_result = await clarify_agent.run_clarify_task(
            ClarifyTask(
                payload=hydrate_result.payload,
                required_slots=required_slots,
                llm_factory=llm_factory,
                memory_state=memory_state,
                max_rounds=max_rounds,
            )
        )
    except Exception:
        return None, ReactClarifyGateResult(
            decision="clarify_short_circuit",
            payload=hydrate_result.payload,
            clarify_reason="当前澄清流程暂不可用，请补充必要信息后继续。",
            clarify_error="clarify_fallback",
            trace=[*trace, "react_clarify_gate_async:fallback:react_error"],
            clarify_messages=None,
        )
    return clarify_result, None


def _normalize_react_result(
    *,
    parse_result: Any,
    hydrate_result: Any,
    trace: list[str],
    react_result: ClarifyResult,
) -> ReactClarifyGateResult:
    react_decision = react_result.decision
    react_messages = react_result.messages
    react_trace = react_result.trace
    merged_payload = hydrate_result.payload.model_copy(update=dict(react_result.resolved_slots))
    react_missing = list(react_result.missing_required_slots)
    resolved_intent = react_result.resolved_intent if react_result.resolved_intent in _VALID_INTENTS else None
    route_target = react_result.route_target if react_result.route_target in _VALID_INTENTS else None
    intent_evidence = [item for item in react_result.intent_evidence if item]

    if route_target is not None and resolved_intent is not None and route_target != resolved_intent:
        return ReactClarifyGateResult(
            decision="clarify_react",
            payload=merged_payload,
            clarify_reason="当前意图判断仍不稳定，请补充关键信息后继续。",
            clarify_error="intent_route_mismatch",
            trace=[*trace, *react_trace, "react_clarify_gate_async:intent_route_mismatch"],
            clarify_messages=react_messages,
        )

    if react_decision == "continue_business" and not react_missing:
        converged_payload = merged_payload
        extra_trace: list[str] = []
        if resolved_intent is not None:
            converged_payload = converged_payload.model_copy(update={"intent_hint": resolved_intent})
            extra_trace.append(f"react_clarify_gate_async:resolved_intent:{resolved_intent}")
        if route_target is not None:
            extra_trace.append(f"react_clarify_gate_async:route_target:{route_target}")
        if intent_evidence:
            extra_trace.append(f"react_clarify_gate_async:intent_evidence:{'|'.join(intent_evidence)}")
        return ReactClarifyGateResult(
            decision="continue_business",
            payload=converged_payload,
            clarify_reason=None,
            clarify_error=None,
            trace=[*trace, *react_trace, *extra_trace, "react_clarify_gate_async:continue_business"],
            clarify_messages=react_messages,
        )

    if parse_result.intent is None:
        return ReactClarifyGateResult(
            decision="clarify_react",
            payload=merged_payload,
            clarify_reason=react_result.clarify_question or "请先确认你的问题类型：规则解释、欠费查询，还是订单金额核验？",
            clarify_error="missing_intent",
            trace=[*trace, *react_trace, "react_clarify_gate_async:pending_intent"],
            clarify_messages=react_messages,
        )
    if react_decision == "clarify_abort":
        return ReactClarifyGateResult(
            decision="clarify_abort",
            payload=merged_payload,
            clarify_reason=react_result.clarify_question or "当前信息仍不足以继续，请补充关键信息后重试。",
            clarify_error="clarify_abort",
            trace=[*trace, *react_trace, "react_clarify_gate_async:abort"],
            clarify_messages=react_messages,
        )
    return ReactClarifyGateResult(
        decision="clarify_react",
        payload=merged_payload,
        clarify_reason=react_result.clarify_question or "请补充必要信息后继续。",
        clarify_error="clarify_react_required",
        trace=[*trace, *react_trace, "react_clarify_gate_async:clarify_react"],
        clarify_messages=react_messages,
    )


async def react_clarify_gate_async(
    *,
    parse_result: Any,
    hydrate_result: Any,
    memory_state: SessionMemoryState | None,
    llm_factory: LLMFactory,
    required_slots_for_intent: RequiredSlotsResolver,
    required_slots_override: list[str] | None = None,
    max_rounds: int = 3,
    clarify_agent: ClarifyAgent | None = None,
) -> ReactClarifyGateResult:
    # Step-3: react_clarify_gate
    # ReAct澄清编排阶段：当 Step-1/Step-2 仍无法收敛时进入，
    # 通过澄清问答（可含工具调用）输出 continue_business/clarify_react/clarify_abort；
    # 若ReAct执行异常，回退到确定性短路澄清提示，避免中断主链路。
    # Gate rules:
    # 1) intent明确且没有缺失槽位 -> continue_business
    # 2) intent明确但有缺失槽位 -> clarify_short_circuit
    # 3) 其他场景（主要是intent不明确） -> 进入ReAct循环
    need_react, trace = _should_enter_react(parse_result=parse_result, hydrate_result=hydrate_result)

    if not need_react:
        return ReactClarifyGateResult(
            decision="continue_business",
            payload=hydrate_result.payload,
            clarify_reason=None,
            clarify_error=None,
            trace=(trace or ["react_clarify_gate_async:pass"]),
            clarify_messages=None,
        )

    short_circuit = _short_circuit_if_possible(parse_result=parse_result, hydrate_result=hydrate_result, trace=trace)
    if short_circuit is not None:
        return short_circuit

    trace.append("react_clarify_gate_async:enter_react")
    clarify_agent_impl = clarify_agent or ReActClarifyAgent()
    react_result, fallback = await _invoke_react_once(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=memory_state,
        llm_factory=llm_factory,
        required_slots_for_intent=required_slots_for_intent,
        required_slots_override=required_slots_override,
        max_rounds=max_rounds,
        trace=trace,
        clarify_agent=clarify_agent_impl,
    )
    if fallback is not None:
        return fallback
    return _normalize_react_result(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        trace=trace,
        react_result=react_result
        or ClarifyResult(
            decision="clarify_react",
            clarify_question=None,
            resolved_slots={},
            slot_updates={},
            resolved_intent=None,
            route_target=None,
            intent_evidence=[],
            missing_required_slots=[],
            trace=[],
            messages=[],
        ),
    )
