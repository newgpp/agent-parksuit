from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_hybrid_should_not_auto_carry_order_no_from_previous_turn(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_arrears_orders(self, plate_no: str | None, city_code: str | None) -> list[dict[str, Any]]:
        assert plate_no == "沪SCN020"
        assert city_code == "310100"
        return [{"order_no": "SCN-020", "arrears_amount": "6.00"}]

    async def _fake_generate_hybrid_answer(query: str, items: list, business_facts: dict[str, Any], intent: str):
        if intent == "arrears_check":
            return ("存在欠费订单。", ["命中欠费单"], "deepseek-chat")
        return ("缺少订单号，无法核验。", ["请提供order_no"], "deepseek-chat")

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
    assert body2["business_facts"]["error"] == "order_reference_needs_clarification"
    assert not any("memory_hydrate:order_no" in item for item in body2["graph_trace"])


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
    assert body2["business_facts"]["error"] == "order_reference_needs_clarification"
    assert "slot_hydrate:none" in body2["graph_trace"]
    assert "react_clarify_gate:order_reference" in body2["graph_trace"]


@pytest.mark.anyio
async def test_hybrid_should_short_circuit_when_arrears_check_missing_plate_no(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"arrears_called": False}

    async def _fake_get_arrears_orders(self, plate_no: str | None, city_code: str | None) -> list[dict[str, Any]]:
        called["arrears_called"] = True
        return []

    async def _fake_generate_hybrid_answer(query: str, items: list, business_facts: dict[str, Any], intent: str):
        return ("占位", [], "deepseek-chat")

    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.BizApiClient.get_arrears_orders",
        _fake_get_arrears_orders,
    )
    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.generate_hybrid_answer",
        _fake_generate_hybrid_answer,
    )

    resp = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": "rag010-ses-missing-plate-001",
            "turn_id": "t1",
            "query": "帮我查下有没有欠费",
            "intent_hint": "arrears_check",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "city_code": "310100",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["business_facts"]["error"] == "missing_plate_no"
    assert "react_clarify_gate:missing_required_slots:plate_no" in body["graph_trace"]
    assert called["arrears_called"] is False
