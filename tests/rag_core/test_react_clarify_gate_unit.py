from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.clarify_agent import ClarifyResult
from agent_parksuite_rag_core.services.react_clarify_gate import react_clarify_gate_async


class _FakeClarifyAgent:
    async def run_clarify_task(self, _task):
        return ClarifyResult(
            decision="continue_business",
            clarify_question=None,
            resolved_slots={"order_no": "SCN-006"},
            slot_updates={"order_no": "SCN-006"},
            resolved_intent="arrears_check",
            route_target="arrears_check",
            intent_evidence=["lookup_order_hit"],
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[],
        )


class _FakeClarifyAgentRuleExplain:
    async def run_clarify_task(self, _task):
        return ClarifyResult(
            decision="continue_business",
            clarify_question=None,
            resolved_slots={
                "lot_code": "SCN-LOT-B",
                "city_code": "310100",
                "matched_rule_count": 1,
                "rule_codes": ["SCN-RULE-DAY-NIGHT"],
            },
            slot_updates={"lot_code": "SCN-LOT-B", "city_code": "310100"},
            resolved_intent="rule_explain",
            route_target="rule_explain",
            intent_evidence=["billing_rules_hit"],
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[],
        )


class _FakeClarifyAgentRouteDefault:
    async def run_clarify_task(self, _task):
        return ClarifyResult(
            decision="continue_business",
            clarify_question=None,
            resolved_slots={"order_no": "SCN-009"},
            slot_updates={"order_no": "SCN-009"},
            resolved_intent="arrears_check",
            route_target=None,
            intent_evidence=["lookup_order_hit"],
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[],
        )


@pytest.mark.anyio
async def test_react_clarify_gate_should_accept_contract_intent_when_continue_business() -> None:
    parse_result = SimpleNamespace(intent=None, ambiguities=[])
    hydrate_result = SimpleNamespace(
        payload=HybridAnswerRequest(query="编码是 SCN-006，帮我看下"),
        missing_required_slots=[],
    )

    result = await react_clarify_gate_async(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=None,
        llm_factory=lambda: None,
        required_slots_for_intent=lambda _intent: (),
        clarify_agent=_FakeClarifyAgent(),
    )

    assert result.decision == "continue_business"
    assert result.payload.intent_hint == "arrears_check"
    assert result.route_target == "arrears_check"
    assert result.payload.order_no == "SCN-006"
    assert "react_clarify_gate_async:resolved_intent:arrears_check" in result.trace
    assert "react_clarify_gate_async:intent_evidence:lookup_order_hit" in result.trace


@pytest.mark.anyio
async def test_react_clarify_gate_should_accept_contract_intent_rule_explain() -> None:
    parse_result = SimpleNamespace(intent=None, ambiguities=[])
    hydrate_result = SimpleNamespace(
        payload=HybridAnswerRequest(query="编码是 SCN-LOT-B，帮我看下"),
        missing_required_slots=[],
    )

    result = await react_clarify_gate_async(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=None,
        llm_factory=lambda: None,
        required_slots_for_intent=lambda _intent: (),
        clarify_agent=_FakeClarifyAgentRuleExplain(),
    )

    assert result.decision == "continue_business"
    assert result.payload.intent_hint == "rule_explain"
    assert result.route_target == "rule_explain"
    assert result.payload.lot_code == "SCN-LOT-B"
    assert "react_clarify_gate_async:resolved_intent:rule_explain" in result.trace
    assert "react_clarify_gate_async:intent_evidence:billing_rules_hit" in result.trace


@pytest.mark.anyio
async def test_react_clarify_gate_should_default_route_target_to_resolved_intent() -> None:
    parse_result = SimpleNamespace(intent=None, ambiguities=[])
    hydrate_result = SimpleNamespace(
        payload=HybridAnswerRequest(query="订单 SCN-009 看下"),
        missing_required_slots=[],
    )

    result = await react_clarify_gate_async(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=None,
        llm_factory=lambda: None,
        required_slots_for_intent=lambda _intent: (),
        clarify_agent=_FakeClarifyAgentRouteDefault(),
    )

    assert result.decision == "continue_business"
    assert result.payload.intent_hint == "arrears_check"
    assert result.route_target == "arrears_check"


class _FakeClarifyAgentMismatch:
    async def run_clarify_task(self, _task):
        return ClarifyResult(
            decision="continue_business",
            clarify_question=None,
            resolved_slots={"order_no": "SCN-006"},
            slot_updates={"order_no": "SCN-006"},
            resolved_intent="arrears_check",
            route_target="rule_explain",
            intent_evidence=["mismatch_case"],
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[],
        )


@pytest.mark.anyio
async def test_react_clarify_gate_should_fallback_when_route_target_mismatch() -> None:
    parse_result = SimpleNamespace(intent=None, ambiguities=[])
    hydrate_result = SimpleNamespace(
        payload=HybridAnswerRequest(query="编码是 SCN-006，帮我看下"),
        missing_required_slots=[],
    )

    result = await react_clarify_gate_async(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=None,
        llm_factory=lambda: None,
        required_slots_for_intent=lambda _intent: (),
        clarify_agent=_FakeClarifyAgentMismatch(),
    )

    assert result.decision == "clarify_react"
    assert result.clarify_error == "intent_route_mismatch"


class _FakeClarifyAgentMissingIntent:
    async def run_clarify_task(self, _task):
        return ClarifyResult(
            decision="continue_business",
            clarify_question=None,
            resolved_slots={"order_no": "SCN-006"},
            slot_updates={"order_no": "SCN-006"},
            resolved_intent=None,
            route_target=None,
            intent_evidence=[],
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[],
        )


@pytest.mark.anyio
async def test_react_clarify_gate_should_not_continue_when_intent_still_missing() -> None:
    parse_result = SimpleNamespace(intent=None, ambiguities=[])
    hydrate_result = SimpleNamespace(
        payload=HybridAnswerRequest(query="编码是 SCN-006，帮我看下"),
        missing_required_slots=[],
    )

    result = await react_clarify_gate_async(
        parse_result=parse_result,
        hydrate_result=hydrate_result,
        memory_state=None,
        llm_factory=lambda: None,
        required_slots_for_intent=lambda _intent: (),
        clarify_agent=_FakeClarifyAgentMissingIntent(),
    )

    assert result.decision == "clarify_react"
    assert result.clarify_error == "missing_intent"
