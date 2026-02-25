from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient

from agent_parksuite_rag_core.services.ingestion import DeterministicEmbedder


def _load_scenario_by_id(scenario_id: str) -> dict[str, Any]:
    dataset = Path(__file__).resolve().parents[2] / "data" / "rag000" / "scenarios.jsonl"
    with dataset.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("scenario_id") == scenario_id:
                return row
    raise AssertionError(f"scenario not found: {scenario_id}")


def _scenario_chunk_text(row: dict[str, Any]) -> str:
    context = row.get("context") or {}
    gt = row.get("ground_truth") or {}
    return "\n".join(
        [
            f"scenario_id: {row.get('scenario_id')}",
            f"query: {row.get('query', '')}",
            f"city_code: {context.get('city_code')}",
            f"lot_code: {context.get('lot_code')}",
            f"order_no: {context.get('order_no')}",
            f"entry_time: {context.get('entry_time')}",
            f"exit_time: {context.get('exit_time')}",
            f"matched_rule_code: {gt.get('matched_rule_code')}",
            f"matched_version_no: {gt.get('matched_version_no')}",
            f"expected_total_amount: {gt.get('expected_total_amount')}",
            f"amount_check_result: {gt.get('amount_check_result')}",
            f"amount_check_action: {gt.get('amount_check_action')}",
            f"notes: {row.get('notes', '')}",
        ]
    )


@pytest.mark.anyio
async def test_hybrid_answer_fee_verify_should_combine_tool_facts_and_rag(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scn_020 = _load_scenario_by_id("SCN-020")
    chunk_text = _scenario_chunk_text(scn_020)
    emb = DeterministicEmbedder(dim=1536).embed_documents([chunk_text])[0]

    source_payload = {
        "source_id": "SRC-HYB-FEE-020",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "RAG000 SCN-020",
        "city_code": "310100",
        "lot_codes": ["SCN-LOT-A"],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-HYB-FEE-020",
                "replace_existing": True,
                "chunks": [
                    {
                        "scenario_id": "SCN-020",
                        "chunk_index": 0,
                        "chunk_text": chunk_text,
                        "embedding": emb,
                        "metadata": {"doc_type": "rule_explain"},
                    }
                ],
            },
        )
    ).status_code == 200

    context = scn_020["context"]
    gt = scn_020["ground_truth"]

    async def _fake_get_parking_order(self, order_no: str) -> dict[str, Any]:
        assert order_no == context["order_no"]
        return {
            "order_no": context["order_no"],
            "billing_rule_code": gt["matched_rule_code"],
            "entry_time": context["entry_time"],
            "exit_time": context["exit_time"],
            "total_amount": gt["order_total_amount"],
            "paid_amount": gt["expected_paid_amount"],
            "arrears_amount": gt["expected_arrears_amount"],
        }

    async def _fake_simulate_billing(self, rule_code, entry_time, exit_time) -> dict[str, Any]:
        assert rule_code == gt["matched_rule_code"]
        return {
            "total_amount": gt["expected_total_amount"],
            "duration_minutes": 60,
            "matched_version_no": gt["matched_version_no"],
            "breakdown": [],
        }

    async def _fake_generate_hybrid_answer(query: str, items: list, business_facts: dict[str, Any], intent: str):
        assert query == scn_020["query"]
        assert len(items) == 1
        assert intent == "fee_verify"
        assert business_facts["amount_check_result"] == "不一致"
        return ("结论：订单金额与模拟金额不一致，需人工复核。", ["核验结果不一致"], "deepseek-chat")

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

    resp = await rag_async_client.post(
        "/api/v1/answer/hybrid",
        json={
            "session_id": "ses-hybrid-001",
            "turn_id": "turn-hybrid-001",
            "query": scn_020["query"],
            "intent_hint": "fee_verify",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "city_code": context["city_code"],
            "lot_code": context["lot_code"],
            "order_no": context["order_no"],
            "source_ids": ["SRC-HYB-FEE-020"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "ses-hybrid-001"
    assert body["turn_id"] == "turn-hybrid-001"
    assert body["memory_ttl_seconds"] > 0
    assert body["intent"] == "fee_verify"
    assert body["conclusion"].startswith("结论：订单金额与模拟金额不一致")
    assert body["business_facts"]["amount_check_result"] == "不一致"
    assert body["business_facts"]["amount_check_action"] == "需人工复核"
    assert body["retrieved_count"] == 1
    assert any(item.startswith("intent_classifier:fee_verify") for item in body["graph_trace"])


@pytest.mark.anyio
async def test_hybrid_answer_arrears_check_should_call_biz_tool(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scn_009 = _load_scenario_by_id("SCN-009")
    chunk_text = _scenario_chunk_text(scn_009)
    emb = DeterministicEmbedder(dim=1536).embed_documents([chunk_text])[0]
    context = scn_009["context"]
    gt = scn_009["ground_truth"]

    source_payload = {
        "source_id": "SRC-HYB-ARR-009",
        "doc_type": "faq",
        "source_type": "biz_derived",
        "title": "RAG000 SCN-009",
        "city_code": context["city_code"],
        "lot_codes": [context["lot_code"]],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-HYB-ARR-009",
                "replace_existing": True,
                "chunks": [
                    {
                        "scenario_id": "SCN-009",
                        "chunk_index": 0,
                        "chunk_text": chunk_text,
                        "embedding": emb,
                        "metadata": {"doc_type": "faq"},
                    }
                ],
            },
        )
    ).status_code == 200

    arrears_orders = [item for item in gt["orders"] if item.get("arrears_status") == "HAS_ARREARS"]

    async def _fake_get_arrears_orders(self, plate_no: str | None, city_code: str | None):
        assert plate_no == context["plate_no"]
        assert city_code == context["city_code"]
        return arrears_orders

    async def _fake_generate_hybrid_answer(query: str, items: list, business_facts: dict[str, Any], intent: str):
        assert intent == "arrears_check"
        assert len(items) == 0
        assert business_facts["arrears_count"] == len(arrears_orders)
        return ("结论：该车牌存在欠费记录。", ["命中欠费订单"], "deepseek-chat")

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
            "session_id": "ses-hybrid-002",
            "query": scn_009["query"],
            "intent_hint": "arrears_check",
            "top_k": 3,
            "doc_type": "faq",
            "source_type": "biz_derived",
            "city_code": context["city_code"],
            "lot_code": context["lot_code"],
            "plate_no": context["plate_no"],
            "source_ids": ["SRC-HYB-ARR-009"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "ses-hybrid-002"
    assert body["turn_id"].startswith("turn-")
    assert body["memory_ttl_seconds"] > 0
    assert body["intent"] == "arrears_check"
    assert body["business_facts"]["arrears_count"] == len(arrears_orders)
    assert body["retrieved_count"] == 0
    assert any(item.startswith("intent_classifier:arrears_check") for item in body["graph_trace"])
