from __future__ import annotations

from datetime import datetime

import pytest
from httpx import AsyncClient


def _vec(value: float = 0.01) -> list[float]:
    return [value] * 1536


@pytest.mark.anyio
async def test_upsert_knowledge_source_create_and_update(rag_async_client: AsyncClient) -> None:
    create_payload = {
        "source_id": "SRC-RULE-001",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "规则说明V1",
        "city_code": "310100",
        "lot_codes": ["LOT-A"],
        "effective_from": "2026-01-01T00:00:00+08:00",
        "effective_to": None,
        "version": "v1",
        "source_uri": "biz://billing-rule/SRC-RULE-001",
        "is_active": True,
    }
    create_resp = await rag_async_client.post("/api/v1/knowledge/sources", json=create_payload)
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["source_id"] == "SRC-RULE-001"
    assert created["source_type"] == "biz_derived"
    first_id = created["id"]

    update_payload = {
        **create_payload,
        "title": "规则说明V2",
        "lot_codes": ["LOT-A", "LOT-B"],
        "version": "v2",
    }
    update_resp = await rag_async_client.post("/api/v1/knowledge/sources", json=update_payload)
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["id"] == first_id
    assert updated["title"] == "规则说明V2"
    assert updated["version"] == "v2"
    assert updated["lot_codes"] == ["LOT-A", "LOT-B"]


@pytest.mark.anyio
async def test_ingest_chunks_should_validate_source_and_embedding_dim(rag_async_client: AsyncClient) -> None:
    not_found_payload = {
        "source_id": "NOT-EXIST",
        "replace_existing": False,
        "chunks": [
            {
                "scenario_id": "SCN-001",
                "chunk_index": 0,
                "chunk_text": "测试分块",
                "embedding": _vec(),
                "metadata": {},
            }
        ],
    }
    not_found_resp = await rag_async_client.post("/api/v1/knowledge/chunks/batch", json=not_found_payload)
    assert not_found_resp.status_code == 404

    source_payload = {
        "source_id": "SRC-FAQ-001",
        "doc_type": "faq",
        "source_type": "manual",
        "title": "FAQ",
        "city_code": "310100",
        "lot_codes": ["LOT-A"],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200

    bad_dim_payload = {
        "source_id": "SRC-FAQ-001",
        "replace_existing": False,
        "chunks": [
            {
                "scenario_id": "SCN-002",
                "chunk_index": 0,
                "chunk_text": "维度错误",
                "embedding": [0.1, 0.2],
                "metadata": {},
            }
        ],
    }
    bad_dim_resp = await rag_async_client.post("/api/v1/knowledge/chunks/batch", json=bad_dim_payload)
    assert bad_dim_resp.status_code == 400

    good_payload = {
        "source_id": "SRC-FAQ-001",
        "replace_existing": True,
        "chunks": [
            {
                "scenario_id": "SCN-003",
                "chunk_index": 0,
                "chunk_text": "欠费说明A",
                "embedding": _vec(0.2),
                "metadata": {"tag": "arrears"},
            },
            {
                "scenario_id": "SCN-004",
                "chunk_index": 1,
                "chunk_text": "欠费说明B",
                "embedding": _vec(0.3),
                "metadata": {"tag": "arrears"},
            },
        ],
    }
    good_resp = await rag_async_client.post("/api/v1/knowledge/chunks/batch", json=good_payload)
    assert good_resp.status_code == 200
    body = good_resp.json()
    assert body["inserted_count"] == 2


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
