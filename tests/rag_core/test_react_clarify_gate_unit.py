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
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[
                {
                    "role": "tool",
                    "content": (
                        '{"tool":"lookup_order","hit":true,'
                        '"order_no":"SCN-006","plate_no":"沪SCN006","city_code":"310100","lot_code":"SCN-LOT-B"}'
                    ),
                }
            ],
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
            missing_required_slots=[],
            trace=["clarify_react:agent:finish_clarify"],
            messages=[],
        )


@pytest.mark.anyio
async def test_react_clarify_gate_should_infer_arrears_check_when_intent_missing_but_order_resolved() -> None:
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
    assert result.payload.order_no == "SCN-006"
    assert "react_clarify_gate_async:infer_intent:arrears_check" in result.trace


@pytest.mark.anyio
async def test_react_clarify_gate_should_infer_rule_explain_when_billing_rule_hit() -> None:
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
    assert result.payload.lot_code == "SCN-LOT-B"
    assert "react_clarify_gate_async:infer_intent:rule_explain" in result.trace
