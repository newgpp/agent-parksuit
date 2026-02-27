from __future__ import annotations

import json
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from loguru import logger

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest

ClarifyAction = Literal["ask_user", "finish_clarify", "abort"]
LLMFactory = Callable[[], Any]

CLARIFY_SYSTEM_PROMPT = (
    "你是停车业务澄清助手。"
    "目标是最短路径补齐业务必填槽位并消除歧义。"
    "当用户参数可能同时代表订单或停车场时，优先调用工具先查订单再查停车场后再判断。"
    "最终回复必须是单个 JSON 对象，且只能包含 JSON，禁止输出任何额外说明、前后缀或 Markdown。"
    '仅输出JSON: {"action":"ask_user|finish_clarify|abort",'
    '"clarify_question":string|null,"slot_updates":object,"reason":string|null}。'
)


class ClarifyReactResult(TypedDict, total=False):
    decision: str
    clarify_question: str | None
    resolved_slots: dict[str, Any]
    missing_required_slots: list[str]
    trace: list[str]
    messages: list[BaseMessage]
    tool_trace: list[dict[str, Any]]


class ClarifyGraphState(TypedDict, total=False):
    messages: list[BaseMessage]


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = content.find("{")
        while start >= 0:
            try:
                candidate, _ = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                start = content.find("{", start + 1)
                continue
            return candidate if isinstance(candidate, dict) else None
        return None
    return parsed if isinstance(parsed, dict) else None


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                parts.append(str(text) if text is not None else str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _missing_slots(required_slots: list[str], resolved_slots: dict[str, Any]) -> list[str]:
    return [slot for slot in required_slots if not resolved_slots.get(slot)]


def _merge_slots_from_payload(payload: HybridAnswerRequest) -> dict[str, Any]:
    return {
        "city_code": payload.city_code,
        "lot_code": payload.lot_code,
        "plate_no": payload.plate_no,
        "order_no": payload.order_no,
        "at_time": payload.at_time,
    }


def _extract_tool_trace(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        trace.append(
            {
                "tool_call_id": str(getattr(message, "tool_call_id", "")),
                "content": str(getattr(message, "content", "")),
            }
        )
    return trace


def build_clarify_react_app(
    *,
    llm_factory: LLMFactory,
    tools: list[Any],
):
    llm = llm_factory()
    return create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=CLARIFY_SYSTEM_PROMPT,
    )


async def _invoke_clarify_agent(
    app: Any,
    messages: list[BaseMessage],
    max_rounds: int,
) -> list[BaseMessage]:
    final_state: ClarifyGraphState = await app.ainvoke(
        {"messages": messages},
        config={"recursion_limit": max(4, max_rounds * 2)},
    )
    return list(final_state.get("messages", []))


def _build_clarify_result(
    *,
    decision: str,
    clarify_question: str | None,
    resolved_slots: dict[str, Any],
    required_slots: list[str],
    trace: list[str],
    final_messages: list[BaseMessage],
    tool_trace: list[dict[str, Any]],
) -> ClarifyReactResult:
    return {
        "decision": decision,
        "clarify_question": clarify_question,
        "resolved_slots": resolved_slots,
        "missing_required_slots": _missing_slots(required_slots, resolved_slots),
        "trace": trace,
        "messages": final_messages,
        "tool_trace": tool_trace,
    }


def _parse_action_payload(last_ai: BaseMessage | None) -> tuple[dict[str, Any] | None, str]:
    if not last_ai:
        return None, ""
    ai_content = _message_content_to_text(getattr(last_ai, "content", ""))
    parsed = _extract_json_payload(ai_content)
    return parsed, ai_content


def _normalize_action_and_slots(
    *,
    parsed: dict[str, Any],
    resolved_slots: dict[str, Any],
    required_slots: list[str],
) -> tuple[ClarifyAction, str, str | None, list[str], list[str]]:
    action_raw = str(parsed.get("action", "ask_user")).strip()
    action: ClarifyAction = action_raw if action_raw in {"ask_user", "finish_clarify", "abort"} else "ask_user"
    slot_updates = parsed.get("slot_updates", {})
    if isinstance(slot_updates, dict):
        for key, value in slot_updates.items():
            if value is not None and str(value).strip():
                resolved_slots[key] = value
    missing_required_slots = _missing_slots(required_slots, resolved_slots)
    if action == "finish_clarify" and missing_required_slots:
        action = "ask_user"
    decision = "clarify_react" if action == "ask_user" else ("continue_business" if action == "finish_clarify" else "clarify_abort")
    clarify_question = parsed.get("clarify_question")
    if action == "ask_user" and not clarify_question:
        clarify_question = "请补充关键信息后继续，例如订单号 SCN-020 或车牌号。"
    slot_keys = sorted(slot_updates.keys()) if isinstance(slot_updates, dict) else []
    return action, decision, (str(clarify_question) if clarify_question is not None else None), missing_required_slots, slot_keys


async def run_clarify_react_once(
    *,
    payload: HybridAnswerRequest,
    llm_factory: LLMFactory,
    required_slots: list[str],
    history_messages: list[BaseMessage] | None = None,
    tools: list[Any] | None = None,
    max_rounds: int = 3,
) -> ClarifyReactResult:
    app = build_clarify_react_app(llm_factory=llm_factory, tools=tools or [])
    history = list(history_messages or [])
    logger.info(
        "clarify_react input session_id={} required_slots={} max_rounds={} history_messages={}",
        payload.session_id,
        required_slots,
        max_rounds,
        len(history),
    )
    messages: list[BaseMessage] = [
        *history,
        HumanMessage(content=payload.query),
    ]
    logger.info(
        "clarify_react merge_messages total_messages={} appended_user_query_len={}",
        len(messages),
        len(payload.query),
    )
    resolved_slots = _merge_slots_from_payload(payload)
    logger.info(
        "clarify_react initial_slots keys_with_value={}",
        sorted([key for key, value in resolved_slots.items() if value is not None]),
    )
    trace: list[str] = ["clarify_react:start", "clarify_react:agent:create_react_agent"]
    final_messages = await _invoke_clarify_agent(app=app, messages=messages, max_rounds=max_rounds)
    tool_trace = _extract_tool_trace(final_messages)
    logger.info(
        "clarify_react agent_done final_messages={} recursion_limit={}",
        len(final_messages),
        max(4, max_rounds * 2),
    )
    last_ai = next((msg for msg in reversed(final_messages) if getattr(msg, "type", "") == "ai"), None)

    parsed, ai_content = _parse_action_payload(last_ai)
    if not parsed:
        question = ai_content.strip()
        trace.append("clarify_react:parse:fallback_ask_user")
        trace.append("clarify_react:agent:ask_user")
        logger.info(
            "clarify_react parse_result=fallback_ask_user question_len={} tool_calls={}",
            len(question),
            len(tool_trace),
        )
        return _build_clarify_result(
            decision="clarify_react",
            clarify_question=(question or "请补充必要信息后继续。"),
            resolved_slots=resolved_slots,
            required_slots=required_slots,
            trace=trace,
            final_messages=final_messages,
            tool_trace=tool_trace,
        )

    action, decision, clarify_question, missing_required_slots, slot_update_keys = _normalize_action_and_slots(
        parsed=parsed,
        resolved_slots=resolved_slots,
        required_slots=required_slots,
    )
    if action == "ask_user":
        trace.append("clarify_react:agent:ask_user")
    elif action == "finish_clarify":
        trace.append("clarify_react:agent:finish_clarify")
    else:
        trace.append("clarify_react:agent:abort")

    logger.info(
        "clarify_react result action={} decision={} slot_updates_keys={} missing_required_slots={} rounds={} tool_calls={}",
        action,
        decision,
        slot_update_keys,
        missing_required_slots,
        max_rounds,
        len(tool_trace),
    )
    return _build_clarify_result(
        decision=decision,
        clarify_question=clarify_question,
        resolved_slots=resolved_slots,
        required_slots=required_slots,
        trace=trace,
        final_messages=final_messages,
        tool_trace=tool_trace,
    )
