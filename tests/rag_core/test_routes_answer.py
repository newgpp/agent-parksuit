from __future__ import annotations

import pytest
from httpx import AsyncClient


def _vec(value: float = 0.01) -> list[float]:
    return [value] * 1536


@pytest.mark.anyio
async def test_answer_should_return_citations_with_mocked_llm(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_generate_answer(query: str, items: list) -> tuple[str, list[str], str]:
        assert query == "上海A场怎么计费"
        assert len(items) == 1
        return ("结论：按A场规则计费。", ["30分钟2元", "日间封顶20元"], "deepseek-chat")

    monkeypatch.setattr(
        "agent_parksuite_rag_core.api.routes.generate_answer_from_chunks",
        _fake_generate_answer,
    )

    source_payload = {
        "source_id": "SRC-ANS-001",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "上海A场规则",
        "city_code": "310100",
        "lot_codes": ["LOT-A"],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-ANS-001",
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

    resp = await rag_async_client.post(
        "/api/v1/answer",
        json={
            "query": "上海A场怎么计费",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "city_code": "310100",
            "lot_code": "LOT-A",
            "source_ids": ["SRC-ANS-001"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["conclusion"] == "结论：按A场规则计费。"
    assert body["key_points"] == ["30分钟2元", "日间封顶20元"]
    assert body["retrieved_count"] == 1
    assert body["model"] == "deepseek-chat"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_id"] == "SRC-ANS-001"


@pytest.mark.anyio
async def test_answer_should_return_503_when_llm_unavailable(
    rag_async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_generate_answer(_: str, __: list) -> tuple[str, list[str], str]:
        raise RuntimeError("RAG_DEEPSEEK_API_KEY is not configured")

    monkeypatch.setattr(
        "agent_parksuite_rag_core.api.routes.generate_answer_from_chunks",
        _fake_generate_answer,
    )

    source_payload = {
        "source_id": "SRC-ANS-ERR-001",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "上海A场规则",
        "city_code": "310100",
        "lot_codes": ["LOT-A"],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-ANS-ERR-001",
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

    resp = await rag_async_client.post(
        "/api/v1/answer",
        json={
            "query": "上海A场怎么计费",
            "top_k": 3,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "city_code": "310100",
            "lot_code": "LOT-A",
            "source_ids": ["SRC-ANS-ERR-001"],
        },
    )
    assert resp.status_code == 503
    assert "RAG_DEEPSEEK_API_KEY" in resp.json()["detail"]
