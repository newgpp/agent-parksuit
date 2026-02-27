from __future__ import annotations

import pytest

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.intent_slot_resolver import (
    _intent_slot_parse,
    resolve_turn_context_async,
)

@pytest.mark.anyio
async def test_resolve_turn_context_async_should_use_llm_slots_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        async def ainvoke(self, _messages):
            class _Resp:
                content = (
                    '{"intent":"fee_verify","intent_confidence":0.91,'
                    '"slots":{"order_no":"SCN-020"},"ambiguities":[]}'
                )

            return _Resp()

    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.get_chat_llm", lambda **_: _FakeLLM())
    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.settings.deepseek_api_key", "x-test")

    payload = HybridAnswerRequest(query="帮我核验一下这单金额", intent_hint=None)
    resolved = await resolve_turn_context_async(payload=payload, memory_state=None)

    assert resolved.payload.order_no == "SCN-020"
    assert resolved.decision == "continue_business"
    assert "intent_slot_parse:llm" in resolved.memory_trace


@pytest.mark.anyio
async def test_intent_slot_parse_should_skip_llm_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.settings.deepseek_api_key", "")
    payload = HybridAnswerRequest(query="这笔订单帮我核验下")

    result = await _intent_slot_parse(payload)

    assert result.intent is None
    assert result.payload.order_no is None
    assert "order_reference" in result.ambiguities
    assert result.trace[-1] == "intent_slot_parse:llm_skip:no_api_key"


@pytest.mark.anyio
async def test_intent_slot_parse_should_merge_llm_json_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        async def ainvoke(self, _messages):
            class _Resp:
                content = (
                    '{"intent":"fee_verify","intent_confidence":0.93,'
                    '"slots":{"order_no":"SCN-020","plate_no":null,"city_code":null,"lot_code":null},'
                    '"ambiguities":["need_order_context"]}'
                )

            return _Resp()

    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.get_chat_llm", lambda **_: _FakeLLM())
    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.settings.deepseek_api_key", "x-test")
    payload = HybridAnswerRequest(query="这笔订单帮我核验下")

    result = await _intent_slot_parse(payload)

    assert result.intent == "fee_verify"
    assert result.payload.order_no == "SCN-020"
    assert result.field_sources["order_no"] == "inferred"
    assert result.missing_required_slots == []
    assert "order_reference" in result.ambiguities
    assert "need_order_context" in result.ambiguities
    assert result.trace[-1] == "intent_slot_parse:llm"


@pytest.mark.anyio
async def test_intent_slot_parse_should_fallback_when_llm_returns_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        async def ainvoke(self, _messages):
            class _Resp:
                content = "not-a-json"

            return _Resp()

    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.get_chat_llm", lambda **_: _FakeLLM())
    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.settings.deepseek_api_key", "x-test")
    payload = HybridAnswerRequest(query="这笔订单帮我核验下")

    result = await _intent_slot_parse(payload)

    assert result.intent is None
    assert result.payload.order_no is None
    assert "order_reference" in result.ambiguities
    assert result.trace[-1] == "intent_slot_parse:llm_invalid_json_fallback"


@pytest.mark.anyio
async def test_intent_slot_parse_should_fallback_when_llm_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        async def ainvoke(self, _messages):
            raise RuntimeError("llm down")

    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.get_chat_llm", lambda **_: _FakeLLM())
    monkeypatch.setattr("agent_parksuite_rag_core.services.intent_slot_resolver.settings.deepseek_api_key", "x-test")
    payload = HybridAnswerRequest(query="这笔订单帮我核验下")

    result = await _intent_slot_parse(payload)

    assert result.intent is None
    assert result.payload.order_no is None
    assert "order_reference" in result.ambiguities
    assert result.trace[-1] == "intent_slot_parse:llm_error_fallback"
