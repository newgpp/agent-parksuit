from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

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
    missing_required_slots: list[str]
    trace: list[str]
    messages: list[dict[str, Any]] | None
    tool_trace: list[dict[str, Any]]


class ClarifyAgent(Protocol):
    async def run_clarify_task(self, task: ClarifyTask) -> ClarifyResult:
        ...


class ReActClarifyAgent:
    async def run_clarify_task(self, task: ClarifyTask) -> ClarifyResult:
        # PR-1 scaffold:
        # keep existing ReAct workflow behavior behind a sub-agent contract.
        react_result = await run_clarify_react_once(
            payload=task.payload,
            llm_factory=task.llm_factory,
            required_slots=task.required_slots,
            memory_state=task.memory_state,
            tools=build_clarify_react_tools(),
            max_rounds=task.max_rounds,
        )
        return ClarifyResult(
            decision=str(react_result.get("decision", "clarify_react")),
            clarify_question=react_result.get("clarify_question"),
            resolved_slots=dict(react_result.get("resolved_slots", {})),
            missing_required_slots=list(react_result.get("missing_required_slots", [])),
            trace=list(react_result.get("trace", [])),
            messages=react_result.get("messages"),
            tool_trace=list(react_result.get("tool_trace", [])),
        )
