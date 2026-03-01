from __future__ import annotations

import ast
import json
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agent_parksuite_rag_core.clients.llm_client import get_default_chat_llm
from agent_parksuite_rag_core.tools.clarify_react_tools import build_clarify_react_tools

CLARIFY_SYSTEM_PROMPT = (
    "你是停车业务澄清助手。"
    "目标是最短路径补齐业务必填槽位并消除歧义。"
    "每轮最多只允许一次工具调用；拿到可用结果后不要继续调用第二个工具，直接给出最终JSON。"
    "当用户参数可能同时代表订单或停车场时，优先调用工具先查订单再查停车场后再判断。"
    "最终回复必须是单个 JSON 对象，且只能包含 JSON，禁止输出任何额外说明、前后缀或 Markdown。"
    '仅输出JSON: {"action":"ask_user|finish_clarify|abort",'
    '"clarify_question":string|null,"slot_updates":object,'
    '"resolved_intent":"rule_explain|arrears_check|fee_verify|null",'
    '"intent_evidence":[string,...],"reason":string|null}。'
)


class ClarifyGraphState(TypedDict, total=False):
    messages: list[BaseMessage]


def build_clarify_react_app(
    *,
    tools: list[Any],
):
    llm = get_default_chat_llm()
    return create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=CLARIFY_SYSTEM_PROMPT,
    )


async def run_clarify_react_graph(
    messages: list[BaseMessage],
    max_rounds: int,
    tools: list[Any] | None = None,
) -> list[BaseMessage]:
    def _tool_content_to_obj(content: Any) -> dict[str, Any] | None:
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            raw = content.strip()
            if not raw:
                return None
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
            try:
                parsed = ast.literal_eval(raw)
                return parsed if isinstance(parsed, dict) else None
            except (ValueError, SyntaxError):
                return None
        return None

    def _has_successful_tool_result(new_messages: list[BaseMessage]) -> bool:
        for msg in reversed(new_messages):
            if not isinstance(msg, ToolMessage):
                continue
            payload = _tool_content_to_obj(getattr(msg, "content", ""))
            if isinstance(payload, dict) and payload.get("hit") is True:
                return True
        return False

    current_messages = list(messages)
    app = build_clarify_react_app(
        tools=(tools if tools is not None else build_clarify_react_tools()),
    )
    app_no_tools = build_clarify_react_app(tools=[])
    recursion_limit = max(4, max_rounds * 2)
    for _ in range(max(1, max_rounds)):
        final_state: ClarifyGraphState = await app.ainvoke(
            {
                "messages": current_messages,
                # One tool cycle per round; multi-round behavior is managed by outer loop.
                "remaining_steps": 2,
            },
            config={"recursion_limit": recursion_limit},
        )
        next_messages = list(final_state.get("messages", []))
        if len(next_messages) <= len(current_messages):
            return next_messages
        added_messages = next_messages[len(current_messages):]
        if _has_successful_tool_result(added_messages):
            # A tool has already returned a successful hit; force final output without tools.
            final_no_tools_state: ClarifyGraphState = await app_no_tools.ainvoke(
                {"messages": next_messages},
                config={"recursion_limit": recursion_limit},
            )
            return list(final_no_tools_state.get("messages", []))
        current_messages = next_messages
    return current_messages
