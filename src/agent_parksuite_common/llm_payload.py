from __future__ import annotations

import json
from typing import Any


def _normalize_role(raw_role: str) -> str:
    role = (raw_role or "").strip().lower()
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    if role in {"system", "user", "assistant", "tool"}:
        return role
    return "user"


def _stringify_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_content(content: Any) -> Any:
    if isinstance(content, (str, list, dict)) or content is None:
        return content
    return str(content)


def _extract_message_payload(message: Any) -> dict[str, Any]:
    role = _normalize_role(getattr(message, "type", message.__class__.__name__.replace("Message", "").lower()))
    payload: dict[str, Any] = {
        "role": role,
        "content": _normalize_content(getattr(message, "content", "")),
    }

    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    openai_role = additional_kwargs.get("__openai_role__")
    if isinstance(openai_role, str) and openai_role.strip():
        payload["role"] = openai_role.strip()

    name = getattr(message, "name", None) or additional_kwargs.get("name")
    if name:
        payload["name"] = str(name)

    if payload["role"] == "assistant":
        tool_calls = getattr(message, "tool_calls", None) or additional_kwargs.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_tool_calls: list[dict[str, Any]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function_name = call.get("name") or ((call.get("function") or {}).get("name"))
                function_args = call.get("args")
                if function_args is None:
                    function_args = (call.get("function") or {}).get("arguments")
                normalized_tool_calls.append(
                    {
                        "id": call.get("id"),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": _stringify_json(function_args if function_args is not None else {}),
                        },
                    }
                )
            if normalized_tool_calls:
                payload["tool_calls"] = normalized_tool_calls
        function_call = additional_kwargs.get("function_call")
        if function_call is not None:
            payload["function_call"] = function_call

    if payload["role"] == "tool":
        tool_call_id = getattr(message, "tool_call_id", None) or additional_kwargs.get("tool_call_id")
        if tool_call_id:
            payload["tool_call_id"] = str(tool_call_id)

    return payload


def dump_llm_input(messages: list[Any], model: str, temperature: float) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [_extract_message_payload(message) for message in messages],
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def dump_llm_output(result: Any, model: str, temperature: float) -> str:
    response_metadata = getattr(result, "response_metadata", {}) or {}
    usage = response_metadata.get("token_usage", {})
    finish_reason = response_metadata.get("finish_reason")
    system_fingerprint = response_metadata.get("system_fingerprint")
    logprobs = response_metadata.get("logprobs")
    payload = {
        "id": getattr(result, "id", None),
        "object": "chat.completion",
        "model": model,
        "temperature": temperature,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": str(getattr(result, "content", "")),
                },
                "logprobs": logprobs,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
        "system_fingerprint": system_fingerprint,
        # Keep LangChain-normalized fields for troubleshooting.
        "response_metadata": response_metadata,
        "usage_metadata": getattr(result, "usage_metadata", None),
        "additional_kwargs": getattr(result, "additional_kwargs", None),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def trim_llm_payload_text(text: str, *, full_payload: bool, max_chars: int) -> str:
    if full_payload:
        return text
    return text[:max_chars]
