from __future__ import annotations

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.intent_slot_resolver import (
    build_request_slots,
    resolve_turn_context,
)


def test_resolve_turn_context_without_memory_keeps_payload() -> None:
    payload = HybridAnswerRequest(query="解释一下停车费规则")
    resolved = resolve_turn_context(payload=payload, memory_state=None)
    assert resolved.decision == "continue_business"
    assert resolved.payload.query == payload.query
    assert resolved.memory_trace == ["memory_hydrate:none"]
    assert resolved.clarify_reason is None


def test_resolve_turn_context_should_short_circuit_when_order_reference_ambiguous() -> None:
    payload = HybridAnswerRequest(query="这笔订单帮我核验下")
    memory_state = {
        "slots": {"city_code": "310100"},
    }
    resolved = resolve_turn_context(payload=payload, memory_state=memory_state)
    assert resolved.decision == "clarify_biz"
    assert resolved.payload.intent_hint == "fee_verify"
    assert resolved.clarify_error == "order_reference_needs_clarification"
    assert resolved.clarify_reason is not None
    assert "memory_hydrate:order_reference_needs_clarification" in resolved.memory_trace


def test_build_request_slots_should_not_auto_fill_order_no_from_memory() -> None:
    payload = HybridAnswerRequest(query="金额不对，帮我复核")
    memory_state = {
        "slots": {"city_code": "310100", "plate_no": "沪SCN020"},
    }
    resolved = resolve_turn_context(payload=payload, memory_state=memory_state)
    slots = build_request_slots(resolved.payload)
    assert resolved.decision == "continue_business"
    assert slots["city_code"] == "310100"
    assert slots["plate_no"] == "沪SCN020"
    assert slots["order_no"] is None
