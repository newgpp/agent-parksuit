from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from httpx import AsyncClient


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


def _openai_embed_documents_or_skip(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("semantic retrieval test skipped: OPENAI_API_KEY not set")

    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("RAG_TEST_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAIEmbeddings(model=model)
    return client.embed_documents(texts)


@pytest.mark.anyio
async def test_retrieve_with_openai_embedding_should_hit_semantic_paraphrase(rag_async_client: AsyncClient) -> None:
    scn_017 = _load_scenario_by_id("SCN-017")
    scn_018 = _load_scenario_by_id("SCN-018")
    chunk_017 = _scenario_chunk_text(scn_017)
    chunk_018 = _scenario_chunk_text(scn_018)

    # Paraphrase SCN-018 intent: same-city, lot-C pricing difference explanation.
    semantic_query = "同城里 C 场为什么和其他停车场收费不一样？"
    emb_017, emb_018, query_emb = _openai_embed_documents_or_skip([chunk_017, chunk_018, semantic_query])

    source_payload = {
        "source_id": "SRC-RAG000-SEMANTIC-001",
        "doc_type": "rule_explain",
        "source_type": "biz_derived",
        "title": "RAG000 语义召回样本",
        "city_code": "310100",
        "lot_codes": ["SCN-LOT-A", "SCN-LOT-C"],
        "is_active": True,
    }
    assert (await rag_async_client.post("/api/v1/knowledge/sources", json=source_payload)).status_code == 200
    assert (
        await rag_async_client.post(
            "/api/v1/knowledge/chunks/batch",
            json={
                "source_id": "SRC-RAG000-SEMANTIC-001",
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
            "query": semantic_query,
            "query_embedding": query_emb,
            "top_k": 2,
            "doc_type": "rule_explain",
            "source_type": "biz_derived",
            "source_ids": ["SRC-RAG000-SEMANTIC-001"],
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
