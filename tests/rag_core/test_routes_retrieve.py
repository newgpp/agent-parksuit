from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from httpx import AsyncClient

from agent_parksuite_rag_core.services.ingestion import DeterministicEmbedder


def _vec(value: float = 0.01) -> list[float]:
    return [value] * 1536


def _load_scenario_by_id(scenario_id: str) -> dict:
    dataset = Path(__file__).resolve().parents[2] / "data" / "rag000" / "scenarios.jsonl"
    with dataset.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("scenario_id") == scenario_id:
                return row
    raise AssertionError(f"scenario not found: {scenario_id}")


def _scenario_chunk_text(row: dict) -> str:
    context = row.get("context") or {}
    gt = row.get("ground_truth") or {}
    return "\n".join(
        [
            f"scenario_id: {row.get('scenario_id')}",
            f"query: {row.get('query', '')}",
            f"city_code: {context.get('city_code')}",
            f"lot_code: {context.get('lot_code')}",
            f"matched_rule_code: {gt.get('matched_rule_code')}",
            f"matched_version_no: {gt.get('matched_version_no')}",
            f"expected_total_amount: {gt.get('expected_total_amount')}",
            f"notes: {row.get('notes', '')}",
        ]
    )


@pytest.mark.anyio
async def test_retrieve_should_filter_by_metadata_and_time(rag_async_client: AsyncClient) -> None:
    source_rule = {
        "source_id": "SRC-RULE-SH-001",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "上海A场规则",
        "city_code": "310100",
        "lot_codes": ["LOT-A"],
        "effective_from": "2026-01-01T00:00:00+08:00",
        "effective_to": "2026-03-01T00:00:00+08:00",
        "version": "v1",
        "is_active": True,
    }
    source_policy = {
        "source_id": "SRC-POLICY-NJ-001",
        "doc_type": "policy_doc",
        "source_type": "manual",
        "title": "南京政策",
        "city_code": "320100",
        "lot_codes": ["LOT-NJ"],
        "effective_from": "2026-01-01T00:00:00+08:00",
        "effective_to": None,
        "version": "v1",
        "is_active": False,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_rule)).status_code == 200
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_policy)).status_code == 200

    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-RULE-SH-001",
                "replace_existing": True,
                "chunks": [
                    {
                        "scenario_id": "SCN-017",
                        "chunk_index": 0,
                        "chunk_text": "上海A场日间30分钟2元，封顶20元",
                        "embedding": _vec(0.11),
                        "metadata": {"doc_type": "rule_explain"},
                    }
                ],
            },
        )
    ).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-POLICY-NJ-001",
                "replace_existing": True,
                "chunks": [
                    {
                        "scenario_id": "SCN-X",
                        "chunk_index": 0,
                        "chunk_text": "南京欠费政策文档",
                        "embedding": _vec(0.99),
                        "metadata": {"doc_type": "policy_doc"},
                    }
                ],
            },
        )
    ).status_code == 200

    retrieve_resp = await rag_async_client.post(
        "/api/v1/retrieve",
        json={
            "query": "上海A场怎么计费",
            "top_k": 5,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "source_ids": ["SRC-RULE-SH-001"],
            "city_code": "310100",
            "lot_code": "LOT-A",
            "at_time": "2026-02-10T10:00:00+08:00",
        },
    )
    assert retrieve_resp.status_code == 200
    items = retrieve_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["source_id"] == "SRC-RULE-SH-001"
    assert items[0]["doc_type"] == "rule_explain"

    # inactive source should be excluded by default
    inactive_resp = await rag_async_client.post(
        "/api/v1/retrieve",
        json={
            "query": "南京政策",
            "top_k": 5,
            "source_ids": ["SRC-POLICY-NJ-001"],
            "city_code": "320100",
            "lot_code": "LOT-NJ",
        },
    )
    assert inactive_resp.status_code == 200
    assert inactive_resp.json()["items"] == []

    # include inactive should return policy source
    include_inactive_resp = await rag_async_client.post(
        "/api/v1/retrieve",
        json={
            "query": "南京政策",
            "top_k": 5,
            "source_ids": ["SRC-POLICY-NJ-001"],
            "city_code": "320100",
            "lot_code": "LOT-NJ",
            "include_inactive": True,
        },
    )
    assert include_inactive_resp.status_code == 200
    rows = include_inactive_resp.json()["items"]
    assert len(rows) == 1
    assert rows[0]["source_id"] == "SRC-POLICY-NJ-001"

    # at_time outside effective window should not match
    out_of_window = await rag_async_client.post(
        "/api/v1/retrieve",
        json={
            "query": "上海A场怎么计费",
            "top_k": 5,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "source_ids": ["SRC-RULE-SH-001"],
            "city_code": "310100",
            "lot_code": "LOT-A",
            "at_time": datetime.fromisoformat("2026-03-01T00:00:00+08:00").isoformat(),
        },
    )
    assert out_of_window.status_code == 200
    assert out_of_window.json()["items"] == []


@pytest.mark.anyio
async def test_retrieve_should_prefer_query_matched_chunks_without_embedding(rag_async_client: AsyncClient) -> None:
    source_payload = {
        "source_id": "SRC-SCN-RETRIEVE-001",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "场景召回测试",
        "city_code": "310100",
        "lot_codes": ["LOT-A"],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-SCN-RETRIEVE-001",
                "replace_existing": True,
                "chunks": [
                    {
                        "scenario_id": "SCN-017",
                        "chunk_index": 0,
                        "chunk_text": "SCN-017 同城不同lot_code差异-A，上海A场白天30分钟2元",
                        "embedding": _vec(0.21),
                        "metadata": {},
                    },
                    {
                        "scenario_id": "SCN-018",
                        "chunk_index": 1,
                        "chunk_text": "SCN-018 同城不同lot_code差异-C，上海C场阶梯计费",
                        "embedding": _vec(0.21),
                        "metadata": {},
                    },
                ],
            },
        )
    ).status_code == 200

    retrieve_resp = await rag_async_client.post(
        "/api/v1/retrieve",
        json={
            "query": "SCN-018 差异-C",
            "top_k": 1,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "city_code": "310100",
            "lot_code": "LOT-A",
        },
    )
    assert retrieve_resp.status_code == 200
    items = retrieve_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["scenario_id"] == "SCN-018"


@pytest.mark.anyio
async def test_retrieve_with_embedding_should_match_rag000_scenario(rag_async_client: AsyncClient) -> None:
    scn_017 = _load_scenario_by_id("SCN-017")
    scn_018 = _load_scenario_by_id("SCN-018")
    chunk_017 = _scenario_chunk_text(scn_017)
    chunk_018 = _scenario_chunk_text(scn_018)
    embedder = DeterministicEmbedder(dim=1536)
    emb_017, emb_018, query_emb = embedder.embed_documents([chunk_017, chunk_018, scn_018["query"]])

    source_payload = {
        "source_id": "SRC-RAG000-EMB-001",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "RAG000 场景召回样本",
        "city_code": "310100",
        "lot_codes": ["SCN-LOT-A", "SCN-LOT-C"],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-RAG000-EMB-001",
                "replace_existing": True,
                "chunks": [
                    {
                        "scenario_id": "SCN-017",
                        "chunk_index": 0,
                        "chunk_text": chunk_017,
                        "embedding": emb_017,
                        "metadata": {},
                    },
                    {
                        "scenario_id": "SCN-018",
                        "chunk_index": 1,
                        "chunk_text": chunk_018,
                        "embedding": emb_018,
                        "metadata": {},
                    },
                ],
            },
        )
    ).status_code == 200

    retrieve_resp = await rag_async_client.post(
        "/api/v1/retrieve",
        json={
            "query": scn_018["query"],
            "query_embedding": query_emb,
            "top_k": 2,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "source_ids": ["SRC-RAG000-EMB-001"],
            "city_code": "310100",
            "lot_code": "SCN-LOT-C",
        },
    )
    assert retrieve_resp.status_code == 200
    items = retrieve_resp.json()["items"]
    assert len(items) == 2
    assert items[0]["scenario_id"] == "SCN-018"
    assert items[0]["score"] is not None
    assert items[1]["score"] is not None
    assert items[0]["score"] <= items[1]["score"]
