from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from httpx import AsyncClient

from agent_parksuite_rag_core.services.memory import get_session_memory_repo


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
        "agent_parksuite_rag_core.clients.biz_api_client.BizApiClient.get_arrears_orders",
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
    assert body2["business_facts"]["error"] == "missing_order_no"
    assert not any("memory_hydrate:order_no" in item for item in body2["graph_trace"])


@pytest.mark.anyio
async def test_hybrid_should_apply_resolver_chain_without_session_id(
    rag_async_client: AsyncClient,
) -> None:
    resp = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "query": "这笔订单金额为什么不一致，帮我核验下",
            "intent_hint": "fee_verify",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["business_facts"]["error"] == "missing_order_no"
    assert "intent_slot_parse:order_reference" in body["graph_trace"]
    assert "slot_hydrate:none" in body["graph_trace"]
    assert "react_clarify_gate_async:short_circuit:missing_order_no" in body["graph_trace"]


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
        "agent_parksuite_rag_core.clients.biz_api_client.BizApiClient.get_arrears_orders",
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
    assert body2["business_facts"]["error"] == "missing_order_no"
    assert "slot_hydrate:none" in body2["graph_trace"]
    assert "react_clarify_gate_async:short_circuit:missing_order_no" in body2["graph_trace"]


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
        "agent_parksuite_rag_core.clients.biz_api_client.BizApiClient.get_arrears_orders",
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
    assert "react_clarify_gate_async:short_circuit:missing_plate_no" in body["graph_trace"]
    assert called["arrears_called"] is False


@pytest.mark.anyio
async def test_hybrid_should_clear_clarify_memory_after_continue_business(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_arrears_orders(self, plate_no: str | None, city_code: str | None) -> list[dict[str, Any]]:
        return [{"order_no": "SCN-020", "arrears_amount": "6.00"}]

    async def _fake_generate_hybrid_answer(query: str, items: list, business_facts: dict[str, Any], intent: str):
        return ("存在欠费订单。", ["命中欠费订单"], "deepseek-chat")

    monkeypatch.setattr(
        "agent_parksuite_rag_core.clients.biz_api_client.BizApiClient.get_arrears_orders",
        _fake_get_arrears_orders,
    )
    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.generate_hybrid_answer",
        _fake_generate_hybrid_answer,
    )

    session_id = "rag011-ses-clear-clarify-001"
    resp1 = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": session_id,
            "turn_id": "t1",
            "query": "帮我查下有没有欠费",
            "intent_hint": "arrears_check",
            "city_code": "310100",
        },
    )
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["business_facts"]["error"] == "missing_plate_no"
    assert "pending_clarification" not in body1["business_facts"]
    assert "clarify_messages" not in body1["business_facts"]

    repo = get_session_memory_repo()
    state1 = await repo.get_session(session_id)
    assert state1 is not None
    assert "pending_clarification" in state1
    assert "clarify_messages" not in state1

    resp2 = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": session_id,
            "turn_id": "t2",
            "query": "车牌沪SCN020帮我查下欠费",
            "intent_hint": "arrears_check",
            "city_code": "310100",
            "plate_no": "沪SCN020",
        },
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["business_facts"]["arrears_count"] == 1

    state2 = await repo.get_session(session_id)
    assert state2 is not None
    assert "pending_clarification" not in state2
    assert "clarify_messages" not in state2


@pytest.mark.anyio
async def test_hybrid_should_not_fallback_intent_when_contract_missing(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_resolve_turn_context_async(payload, memory_state):
        return SimpleNamespace(
            payload=payload,
            decision="continue_business",
            memory_trace=["intent_slot_parse:deterministic"],
            clarify_reason=None,
            clarify_error=None,
            clarify_messages=None,
            resolved_intent=None,
            execution_context=None,
        )

    monkeypatch.setattr(
        "agent_parksuite_rag_core.services.hybrid_answering.resolve_turn_context_async",
        _fake_resolve_turn_context_async,
    )

    resp = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": "rag012-ses-missing-intent-contract-001",
            "turn_id": "t1",
            "query": "这单怎么收费",
            "city_code": "310100",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["business_facts"]["error"] == "missing_intent_contract"
    assert "intent_router:missing_intent_contract" in body["graph_trace"]
