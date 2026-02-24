from __future__ import annotations

from agent_parksuite_rag_core.services.ingestion import (
    DeterministicEmbedder,
    build_sources_from_scenarios,
    split_text,
)


def test_split_text_should_return_overlap_chunks() -> None:
    text = "A" * 120
    chunks = split_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 3
    assert len(chunks[0]) == 50
    assert len(chunks[1]) == 50
    assert len(chunks[2]) == 40


def test_build_sources_from_scenarios_should_generate_doc_type_sources() -> None:
    rows = [
        {
            "scenario_id": "SCN-001",
            "intent_tags": ["rule_explain"],
            "query": "怎么计费",
            "context": {
                "city_code": "310100",
                "lot_code": "LOT-A",
                "entry_time": "2026-02-01T08:00:00+08:00",
            },
            "ground_truth": {
                "matched_rule_code": "RULE-A",
                "matched_version_no": 1,
                "expected_total_amount": "4.00",
            },
            "expected_citations": {"doc_type": ["rule_explain", "faq"]},
            "notes": "test",
        }
    ]

    drafts = build_sources_from_scenarios(rows=rows, source_uri="data/rag000/scenarios.jsonl", chunk_size=120, overlap=20)
    assert len(drafts) == 2
    assert {item.doc_type for item in drafts} == {"rule_explain", "faq"}
    assert all(item.source_id.startswith("RAG000-SCN-001-") for item in drafts)
    assert all(item.lot_codes == ["LOT-A"] for item in drafts)
    assert all(len(item.chunks) >= 1 for item in drafts)


def test_deterministic_embedder_dimension() -> None:
    embedder = DeterministicEmbedder(dim=1536)
    vectors = embedder.embed_documents(["hello world", "停车 计费 规则"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
    assert len(vectors[1]) == 1536
