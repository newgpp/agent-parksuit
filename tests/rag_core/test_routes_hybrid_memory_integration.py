from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_hybrid_should_carry_order_no_from_previous_turn(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_arrears_orders(self, plate_no: str | None, city_code: str | None) -> list[dict[str, Any]]:
        assert plate_no == "沪SCN020"
        assert city_code == "310100"
        return [{"order_no": "SCN-020", "arrears_amount": "6.00"}]

    async def _fake_get_parking_order(self, order_no: str) -> dict[str, Any]:
        assert order_no == "SCN-020"
        return {
            "order_no": "SCN-020",
            "billing_rule_code": "SCN-RULE-PERIODIC",
            "entry_time": "2026-02-01T01:00:00+00:00",
            "exit_time": "2026-02-01T02:00:00+00:00",
            "total_amount": "6.00",
            "paid_amount": "0.00",
            "arrears_amount": "6.00",
        }

    async def _fake_simulate_billing(self, rule_code, entry_time, exit_time) -> dict[str, Any]:
        return {
            "total_amount": "4.00",
            "duration_minutes": 60,
            "matched_version_no": 1,
            "breakdown": [],
        }

    async def _fake_generate_hybrid_answer(query: str, items: list, business_facts: dict[str, Any], intent: str):
        if intent == "arrears_check":
            return ("存在欠费订单。", ["命中欠费单"], "deepseek-chat")
        return ("该订单金额不一致。", ["需人工复核"], "deepseek-chat")

    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.BizApiClient.get_arrears_orders",
        _fake_get_arrears_orders,
    )
    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.BizApiClient.get_parking_order",
        _fake_get_parking_order,
    )
    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.BizApiClient.simulate_billing",
        _fake_simulate_billing,
    )
    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.generate_hybrid_answer",
        _fake_generate_hybrid_answer,
    )

    resp1 = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": "rag009-ses-carry-001",
            "turn_id": "t1",
            "query": "帮我查下车牌沪SCN020有没有欠费",
            "intent_hint": "arrears_check",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "city_code": "310100",
            "plate_no": "沪SCN020",
        },
    )
    assert resp1.status_code == 200

    resp2 = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": "rag009-ses-carry-001",
            "turn_id": "t2",
            "query": "这笔订单金额为什么不一致，帮我核验下",
            "intent_hint": "fee_verify",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
        },
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["session_id"] == "rag009-ses-carry-001"
    assert body2["business_facts"]["order_no"] == "SCN-020"
    assert any("memory_hydrate:order_no" in item for item in body2["graph_trace"])


@pytest.mark.anyio
async def test_hybrid_should_not_carry_memory_across_sessions(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_arrears_orders(self, plate_no: str | None, city_code: str | None) -> list[dict[str, Any]]:
        return [{"order_no": "SCN-020", "arrears_amount": "6.00"}]

    async def _fake_generate_hybrid_answer(query: str, items: list, business_facts: dict[str, Any], intent: str):
        if business_facts.get("error"):
            return ("缺少订单号，无法核验。", ["请提供order_no"], "deepseek-chat")
        return ("存在欠费订单。", ["命中欠费单"], "deepseek-chat")

    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.BizApiClient.get_arrears_orders",
        _fake_get_arrears_orders,
    )
    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.generate_hybrid_answer",
        _fake_generate_hybrid_answer,
    )

    resp1 = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": "rag009-ses-iso-A",
            "turn_id": "t1",
            "query": "帮我查下车牌沪SCN020有没有欠费",
            "intent_hint": "arrears_check",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "city_code": "310100",
            "plate_no": "沪SCN020",
        },
    )
    assert resp1.status_code == 200

    resp2 = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": "rag009-ses-iso-B",
            "turn_id": "t2",
            "query": "这笔订单金额为什么不一致，帮我核验下",
            "intent_hint": "fee_verify",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
        },
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["session_id"] == "rag009-ses-iso-B"
    assert body2["business_facts"]["error"] == "order_no is required for fee_verify"
    assert "memory_hydrate:none" in body2["graph_trace"]
