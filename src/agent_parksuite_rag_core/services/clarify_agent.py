from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.memory import SessionMemoryState
from agent_parksuite_rag_core.tools.clarify_react_tools import build_clarify_react_tools
from agent_parksuite_rag_core.workflows.clarify_react_graph import run_clarify_react_once

LLMFactory = Callable[[], Any]


@dataclass(frozen=True)
class ClarifyTask:
    payload: HybridAnswerRequest
    required_slots: list[str]
    llm_factory: LLMFactory
    memory_state: SessionMemoryState | None = None
    max_rounds: int = 3


@dataclass(frozen=True)
class ClarifyResult:
    decision: str
    clarify_question: str | None
    resolved_slots: dict[str, Any]
    slot_updates: dict[str, Any]
    resolved_intent: str | None
    route_target: str | None
    intent_evidence: list[str]
    missing_required_slots: list[str]
    trace: list[str]
    messages: list[dict[str, Any]] | None


class ClarifyAgent(Protocol):
    async def run_clarify_task(self, task: ClarifyTask) -> ClarifyResult:
        ...


class ReActClarifyAgent:
    @staticmethod
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

    @staticmethod
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

    async def run_clarify_task(self, task: ClarifyTask) -> ClarifyResult:
        # Clarify memory (de)serialization is encapsulated inside sub-agent.
        history_messages = self._load_history_messages(task.memory_state)
        react_result = await run_clarify_react_once(
            payload=task.payload,
            llm_factory=task.llm_factory,
            required_slots=task.required_slots,
            history_messages=history_messages,
            tools=build_clarify_react_tools(),
            max_rounds=task.max_rounds,
        )
        messages = react_result.get("messages", [])
        serialized_messages = (
            self._dump_history_messages(messages)
            if isinstance(messages, list)
            else []
        )
        return ClarifyResult(
            decision=str(react_result.get("decision", "clarify_react")),
            clarify_question=react_result.get("clarify_question"),
            resolved_slots=dict(react_result.get("resolved_slots", {})),
            slot_updates=dict(react_result.get("slot_updates", {})),
            resolved_intent=(
                str(react_result.get("resolved_intent")).strip()
                if react_result.get("resolved_intent") is not None
                else None
            ),
            route_target=(
                str(react_result.get("route_target")).strip()
                if react_result.get("route_target") is not None
                else None
            ),
            intent_evidence=[str(item) for item in react_result.get("intent_evidence", []) if str(item).strip()],
            missing_required_slots=list(react_result.get("missing_required_slots", [])),
            trace=list(react_result.get("trace", [])),
            messages=serialized_messages,
        )
