from __future__ import annotations

import json
from typing import Any, Callable, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from loguru import logger

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.memory import SessionMemoryState

ClarifyAction = Literal["ask_user", "finish_clarify", "abort"]
LLMFactory = Callable[[], Any]

CLARIFY_SYSTEM_PROMPT = (
    "你是停车业务澄清助手。"
    "目标是最短路径补齐业务必填槽位并消除歧义。"
    '仅输出JSON: {"action":"ask_user|finish_clarify|abort",'
    '"clarify_question":string|null,"slot_updates":object,"reason":string|null}。'
)


class ClarifyReactResult(TypedDict, total=False):
    decision: str
    clarify_question: str | None
    resolved_slots: dict[str, Any]
    missing_required_slots: list[str]
    trace: list[str]
    messages: list[dict[str, Any]]


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
        return None
    return parsed if isinstance(parsed, dict) else None


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


def _load_history_messages(memory_state: SessionMemoryState | None) -> list[BaseMessage]:
    if not memory_state:
        return []
    raw_messages = memory_state.get("clarify_messages", [])
    messages: list[BaseMessage] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", ""))
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "tool":
            tool_call_id = str(item.get("tool_call_id", ""))
            if tool_call_id:
                messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        elif role == "system":
            messages.append(SystemMessage(content=content))
    return messages


def _dump_history_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        msg_type = getattr(message, "type", "")
        if msg_type == "human":
            role = "user"
        elif msg_type == "ai":
            role = "assistant"
        elif msg_type == "tool":
            role = "tool"
        else:
            role = "system"
        item: dict[str, Any] = {"role": role, "content": str(getattr(message, "content", ""))}
        if role == "tool":
            item["tool_call_id"] = str(getattr(message, "tool_call_id", ""))
        serialized.append(item)
    return serialized


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


async def run_clarify_react_once(
    *,
    payload: HybridAnswerRequest,
    llm_factory: LLMFactory,
    required_slots: list[str],
    memory_state: SessionMemoryState | None = None,
    tools: list[Any] | None = None,
    max_rounds: int = 3,
) -> ClarifyReactResult:
    app = build_clarify_react_app(llm_factory=llm_factory, tools=tools or [])
    history = _load_history_messages(memory_state)
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
    final_state: ClarifyGraphState = await app.ainvoke(
        {"messages": messages},
        config={"recursion_limit": max(4, max_rounds * 2)},
    )
    final_messages = list(final_state.get("messages", []))
    logger.info(
        "clarify_react agent_done final_messages={} recursion_limit={}",
        len(final_messages),
        max(4, max_rounds * 2),
    )
    last_ai = next((msg for msg in reversed(final_messages) if getattr(msg, "type", "") == "ai"), None)

    parsed = _extract_json_payload(str(getattr(last_ai, "content", ""))) if last_ai else None
    if not parsed:
        trace.append("clarify_react:parse:invalid_json")
        question = str(getattr(last_ai, "content", "")).strip() if last_ai else ""
        logger.info(
            "clarify_react parse_result=invalid_json fallback_question_len={}",
            len(question),
        )
        return {
            "decision": "clarify_react",
            "clarify_question": question or "请补充必要信息后继续。",
            "resolved_slots": resolved_slots,
            "missing_required_slots": _missing_slots(required_slots, resolved_slots),
            "trace": trace,
            "messages": _dump_history_messages(final_messages),
        }

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
    if action == "ask_user":
        trace.append("clarify_react:agent:ask_user")
    elif action == "finish_clarify":
        trace.append("clarify_react:agent:finish_clarify")
    else:
        trace.append("clarify_react:agent:abort")

    decision = "clarify_react" if action == "ask_user" else ("continue_business" if action == "finish_clarify" else "clarify_abort")
    clarify_question = parsed.get("clarify_question")
    if action == "ask_user" and not clarify_question:
        clarify_question = "请补充关键信息后继续，例如订单号 SCN-020 或车牌号。"

    logger.info(
        "clarify_react result action={} decision={} slot_updates_keys={} missing_required_slots={} rounds={}",
        action,
        decision,
        sorted(slot_updates.keys()) if isinstance(slot_updates, dict) else [],
        missing_required_slots,
        max_rounds,
    )
    return {
        "decision": decision,
        "clarify_question": str(clarify_question) if clarify_question is not None else None,
        "resolved_slots": resolved_slots,
        "missing_required_slots": missing_required_slots,
        "trace": trace,
        "messages": _dump_history_messages(final_messages),
    }
