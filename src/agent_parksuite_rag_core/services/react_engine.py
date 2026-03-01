from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from agent_parksuite_common.llm_payload import dump_llm_input, dump_llm_output, trim_llm_payload_text
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.memory import SessionMemoryState
from agent_parksuite_rag_core.workflows.clarify_react_graph import run_clarify_react_graph

ClarifyAction = Literal["ask_user", "finish_clarify", "abort"]


@dataclass(frozen=True)
class ReActTask:
    payload: HybridAnswerRequest
    required_slots: list[str]
    memory_state: SessionMemoryState | None = None
    max_rounds: int = 3


@dataclass(frozen=True)
class ReActResult:
    decision: str
    clarify_question: str | None
    resolved_slots: dict[str, Any]
    slot_updates: dict[str, Any]
    resolved_intent: str | None
    intent_evidence: list[str]
    missing_required_slots: list[str]
    trace: list[str]
    messages: list[dict[str, Any]] | None


class ReActEngine(Protocol):
    async def run(self, task: ReActTask) -> ReActResult:
        ...


class DefaultReActEngine:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _missing_slots(required_slots: list[str], resolved_slots: dict[str, Any]) -> list[str]:
        return [slot for slot in required_slots if not resolved_slots.get(slot)]

    @staticmethod
    def _merge_slots_from_payload(payload: HybridAnswerRequest) -> dict[str, Any]:
        return {
            "city_code": payload.city_code,
            "lot_code": payload.lot_code,
            "plate_no": payload.plate_no,
            "order_no": payload.order_no,
            "at_time": payload.at_time,
        }

    @staticmethod
    def _parse_action_payload(last_ai: BaseMessage | None) -> tuple[dict[str, Any] | None, str]:
        if not last_ai:
            return None, ""
        ai_content = DefaultReActEngine._message_content_to_text(getattr(last_ai, "content", ""))
        parsed = DefaultReActEngine._extract_json_payload(ai_content)
        return parsed, ai_content

    @staticmethod
    def _normalize_action_and_slots(
        *,
        parsed: dict[str, Any],
        resolved_slots: dict[str, Any],
        required_slots: list[str],
    ) -> tuple[ClarifyAction, str, str | None, list[str], dict[str, Any], str | None, list[str]]:
        action_raw = str(parsed.get("action", "ask_user")).strip()
        action: ClarifyAction = action_raw if action_raw in {"ask_user", "finish_clarify", "abort"} else "ask_user"
        slot_updates_raw = parsed.get("slot_updates", {})
        slot_updates: dict[str, Any] = {}
        if isinstance(slot_updates_raw, dict):
            for key, value in slot_updates_raw.items():
                if value is not None and str(value).strip():
                    slot_updates[key] = value
                    resolved_slots[key] = value
        missing_required_slots = DefaultReActEngine._missing_slots(required_slots, resolved_slots)
        if action == "finish_clarify" and missing_required_slots:
            action = "ask_user"
        decision = (
            "clarify_react" if action == "ask_user" else ("continue_business" if action == "finish_clarify" else "clarify_abort")
        )
        clarify_question = parsed.get("clarify_question")
        if action == "ask_user" and not clarify_question:
            clarify_question = "请补充关键信息后继续，例如订单号 SCN-020 或车牌号。"
        resolved_intent_raw = parsed.get("resolved_intent")
        resolved_intent = str(resolved_intent_raw).strip() if resolved_intent_raw is not None else None
        intent_evidence_raw = parsed.get("intent_evidence", [])
        intent_evidence = (
            [str(item).strip() for item in intent_evidence_raw if str(item).strip()]
            if isinstance(intent_evidence_raw, list)
            else []
        )
        return (
            action,
            decision,
            (str(clarify_question) if clarify_question is not None else None),
            missing_required_slots,
            slot_updates,
            resolved_intent,
            intent_evidence,
        )

    @staticmethod
    def _log_react_llm_hops(initial_messages: list[BaseMessage], final_messages: list[BaseMessage]) -> None:
        logger.info(
            "llm[clarify_react] hop1_input_payload={}",
            trim_llm_payload_text(
                dump_llm_input(messages=initial_messages, model=settings.deepseek_model, temperature=0),
                full_payload=settings.llm_log_full_payload,
                max_chars=settings.llm_log_max_chars,
            ),
        )
        ai_with_index: list[tuple[int, BaseMessage]] = [
            (idx, msg) for idx, msg in enumerate(final_messages) if getattr(msg, "type", "") == "ai"
        ]
        if not ai_with_index:
            return
        logger.info(
            "llm[clarify_react] hop1_output_payload={}",
            trim_llm_payload_text(
                dump_llm_output(result=ai_with_index[0][1], model=settings.deepseek_model, temperature=0),
                full_payload=settings.llm_log_full_payload,
                max_chars=settings.llm_log_max_chars,
            ),
        )
        if len(ai_with_index) < 2:
            return
        second_ai_index, second_ai_message = ai_with_index[1]
        logger.info(
            "llm[clarify_react] hop2_input_payload={}",
            trim_llm_payload_text(
                dump_llm_input(
                    messages=final_messages[:second_ai_index],
                    model=settings.deepseek_model,
                    temperature=0,
                ),
                full_payload=settings.llm_log_full_payload,
                max_chars=settings.llm_log_max_chars,
            ),
        )
        logger.info(
            "llm[clarify_react] hop2_output_payload={}",
            trim_llm_payload_text(
                dump_llm_output(result=second_ai_message, model=settings.deepseek_model, temperature=0),
                full_payload=settings.llm_log_full_payload,
                max_chars=settings.llm_log_max_chars,
            ),
        )

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

    async def run(self, task: ReActTask) -> ReActResult:
        # Clarify memory (de)serialization is encapsulated inside ReAct engine.
        history_messages = self._load_history_messages(task.memory_state)
        payload = task.payload
        required_slots = task.required_slots
        history = list(history_messages or [])
        logger.info(
            "clarify_react input session_id={} required_slots={} max_rounds={} history_messages={}",
            payload.session_id,
            required_slots,
            task.max_rounds,
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
        resolved_slots = self._merge_slots_from_payload(payload)
        logger.info(
            "clarify_react initial_slots keys_with_value={}",
            sorted([key for key, value in resolved_slots.items() if value is not None]),
        )
        trace: list[str] = ["clarify_react:start", "clarify_react:agent:create_react_agent"]
        final_messages = await run_clarify_react_graph(messages=messages, max_rounds=task.max_rounds)
        logger.info(
            "clarify_react agent_done final_messages={} recursion_limit={}",
            len(final_messages),
            max(4, task.max_rounds * 2),
        )
        self._log_react_llm_hops(initial_messages=messages, final_messages=final_messages)
        last_ai = next((msg for msg in reversed(final_messages) if getattr(msg, "type", "") == "ai"), None)
        parsed, ai_content = self._parse_action_payload(last_ai)

        decision = "clarify_react"
        clarify_question: str | None = None
        slot_updates: dict[str, Any] = {}
        resolved_intent: str | None = None
        intent_evidence: list[str] = []

        if not parsed:
            clarify_question = ai_content.strip() or "请补充必要信息后继续。"
            trace.append("clarify_react:parse:fallback_ask_user")
            trace.append("clarify_react:agent:ask_user")
            logger.info(
                "clarify_react parse_result=fallback_ask_user question_len={}",
                len(clarify_question),
            )
        else:
            action, decision, clarify_question, missing_required_slots, slot_updates, resolved_intent, intent_evidence = (
                self._normalize_action_and_slots(
                    parsed=parsed,
                    resolved_slots=resolved_slots,
                    required_slots=required_slots,
                )
            )
            if action == "ask_user":
                trace.append("clarify_react:agent:ask_user")
            elif action == "finish_clarify":
                trace.append("clarify_react:agent:finish_clarify")
            else:
                trace.append("clarify_react:agent:abort")
            logger.info(
                "clarify_react result action={} decision={} slot_updates_keys={} missing_required_slots={} rounds={}",
                action,
                decision,
                sorted(slot_updates.keys()),
                missing_required_slots,
                task.max_rounds,
            )

        serialized_messages = self._dump_history_messages(final_messages) if isinstance(final_messages, list) else []
        return ReActResult(
            decision=decision,
            clarify_question=clarify_question,
            resolved_slots=resolved_slots,
            slot_updates=slot_updates,
            resolved_intent=resolved_intent,
            intent_evidence=intent_evidence,
            missing_required_slots=self._missing_slots(required_slots, resolved_slots),
            trace=trace,
            messages=serialized_messages,
        )


# Backward-compatible aliases for old naming.
ClarifyTask = ReActTask
ClarifyResult = ReActResult
ClarifyAgent = ReActEngine
ReActClarifyAgent = DefaultReActEngine
