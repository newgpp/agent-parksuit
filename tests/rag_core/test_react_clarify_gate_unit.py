from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.services.react_engine import ReActResult
from agent_parksuite_rag_core.services.react_clarify_gate import react_clarify_gate_async


class _FakeReActEngine:
    async def run(self, _task):
        return ReActResult(
            decision="continue_business",
            clarify_question=None,
            resolved_slots={"order_no": "SCN-006"},
            slot_updates={"order_no": "SCN-006"},
            resolved_intent="arrears_check",
            intent_evidence=["lookup_order_hit"],
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[],
        )


class _FakeReActEngineRuleExplain:
    async def run(self, _task):
        return ReActResult(
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
            intent_evidence=["billing_rules_hit"],
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
        required_slots_for_intent=lambda _intent: (),
        react_engine=_FakeReActEngine(),
    )

    assert result.decision == "continue_business"
    assert result.payload.intent_hint == "arrears_check"
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
        required_slots_for_intent=lambda _intent: (),
        react_engine=_FakeReActEngineRuleExplain(),
    )

    assert result.decision == "continue_business"
    assert result.payload.intent_hint == "rule_explain"
    assert result.payload.lot_code == "SCN-LOT-B"
    assert "react_clarify_gate_async:resolved_intent:rule_explain" in result.trace
    assert "react_clarify_gate_async:intent_evidence:billing_rules_hit" in result.trace


class _FakeReActEngineMissingIntent:
    async def run(self, _task):
        return ReActResult(
            decision="continue_business",
            clarify_question=None,
            resolved_slots={"order_no": "SCN-006"},
            slot_updates={"order_no": "SCN-006"},
            resolved_intent=None,
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
        required_slots_for_intent=lambda _intent: (),
        react_engine=_FakeReActEngineMissingIntent(),
    )

    assert result.decision == "clarify_react"
    assert result.clarify_error == "missing_intent"
