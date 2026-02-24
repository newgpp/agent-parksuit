from __future__ import annotations

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

