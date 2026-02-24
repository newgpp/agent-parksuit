from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_parksuite_rag_core.db.models import KnowledgeChunk, KnowledgeSource


@dataclass
class ChunkDraft:
    scenario_id: str | None
    chunk_index: int
    chunk_text: str
    metadata: dict[str, Any]


@dataclass
class SourceDraft:
    source_id: str
    doc_type: str
    source_type: str
    title: str
    city_code: str | None
    lot_codes: list[str]
    effective_from: datetime | None
    effective_to: datetime | None
    version: str | None
    source_uri: str | None
    is_active: bool
    chunks: list[ChunkDraft]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if chunk_size <= 0:
        return [cleaned]
    if overlap >= chunk_size:
        overlap = 0

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += step
    return chunks


def _build_scenario_text(row: dict[str, Any], doc_type: str) -> str:
    context = row.get("context") or {}
    gt = row.get("ground_truth") or {}
    parts = [
        f"scenario_id: {row.get('scenario_id')}",
        f"doc_type: {doc_type}",
        f"query: {row.get('query', '')}",
        f"city_code: {context.get('city_code')}",
        f"lot_code: {context.get('lot_code')}",
        f"entry_time: {context.get('entry_time')}",
        f"exit_time: {context.get('exit_time')}",
        f"matched_rule_code: {gt.get('matched_rule_code')}",
        f"matched_version_no: {gt.get('matched_version_no')}",
        f"expected_total_amount: {gt.get('expected_total_amount')}",
        f"order_total_amount: {gt.get('order_total_amount')}",
        f"amount_check_result: {gt.get('amount_check_result')}",
        f"amount_check_action: {gt.get('amount_check_action')}",
        f"expected_arrears_amount: {gt.get('expected_arrears_amount')}",
        f"expected_arrears_status: {gt.get('expected_arrears_status')}",
        f"notes: {row.get('notes', '')}",
    ]
    return "\n".join(parts)


def build_sources_from_scenarios(
    rows: list[dict[str, Any]],
    source_uri: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[SourceDraft]:
    drafts: list[SourceDraft] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", "")).strip()
        if not scenario_id:
            continue

        citations = row.get("expected_citations") or {}
        doc_types = citations.get("doc_type") or ["rule_explain"]
        context = row.get("context") or {}
        gt = row.get("ground_truth") or {}

        for doc_type in doc_types:
            text = _build_scenario_text(row, doc_type)
            raw_chunks = split_text(text=text, chunk_size=chunk_size, overlap=overlap)
            if not raw_chunks:
                continue
            chunk_rows = [
                ChunkDraft(
                    scenario_id=scenario_id,
                    chunk_index=idx,
                    chunk_text=item,
                    metadata={
                        "scenario_id": scenario_id,
                        "intent_tags": row.get("intent_tags", []),
                        "doc_type": doc_type,
                    },
                )
                for idx, item in enumerate(raw_chunks)
            ]
            drafts.append(
                SourceDraft(
                    source_id=f"RAG000-{scenario_id}-{doc_type}",
                    doc_type=str(doc_type),
                    source_type="biz_derived",
                    title=f"RAG000 {scenario_id} {doc_type}",
                    city_code=context.get("city_code"),
                    lot_codes=[context.get("lot_code")] if context.get("lot_code") else [],
                    effective_from=_parse_dt(context.get("entry_time")),
                    effective_to=None,
                    version=(str(gt.get("matched_version_no")) if gt.get("matched_version_no") is not None else None),
                    source_uri=source_uri,
                    is_active=True,
                    chunks=chunk_rows,
                )
            )
    return drafts


def build_sources_from_markdown(
    files: list[Path],
    source_prefix: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[SourceDraft]:
    drafts: list[SourceDraft] = []
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        title = file_path.stem
        chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue
        chunk_rows = [
            ChunkDraft(
                scenario_id=None,
                chunk_index=idx,
                chunk_text=item,
                metadata={"source_file": str(file_path), "doc_type": "policy_doc"},
            )
            for idx, item in enumerate(chunks)
        ]
        drafts.append(
            SourceDraft(
                source_id=f"{source_prefix}-{title}",
                doc_type="policy_doc",
                source_type="manual",
                title=title,
                city_code=None,
                lot_codes=[],
                effective_from=None,
                effective_to=None,
                version=None,
                source_uri=str(file_path),
                is_active=True,
                chunks=chunk_rows,
            )
        )
    return drafts


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


class DeterministicEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAIEmbedder:
    def __init__(self, model: str) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.client = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed_documents(texts)


async def upsert_sources_and_chunks(
    session: AsyncSession,
    drafts: list[SourceDraft],
    embedder: DeterministicEmbedder | OpenAIEmbedder,
    replace_existing: bool,
) -> tuple[int, int]:
    source_count = 0
    chunk_count = 0

    for draft in drafts:
        existing = (
            await session.execute(select(KnowledgeSource).where(KnowledgeSource.source_id == draft.source_id))
        ).scalar_one_or_none()
        if existing is None:
            source = KnowledgeSource(
                source_id=draft.source_id,
                doc_type=draft.doc_type,
                source_type=draft.source_type,
                title=draft.title,
                city_code=draft.city_code,
                lot_codes=draft.lot_codes,
                effective_from=draft.effective_from,
                effective_to=draft.effective_to,
                version=draft.version,
                source_uri=draft.source_uri,
                is_active=draft.is_active,
            )
            session.add(source)
            await session.flush()
        else:
            source = existing
            source.doc_type = draft.doc_type
            source.source_type = draft.source_type
            source.title = draft.title
            source.city_code = draft.city_code
            source.lot_codes = draft.lot_codes
            source.effective_from = draft.effective_from
            source.effective_to = draft.effective_to
            source.version = draft.version
            source.source_uri = draft.source_uri
            source.is_active = draft.is_active
            if replace_existing:
                await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source_pk == source.id))

        embeddings = embedder.embed_documents([item.chunk_text for item in draft.chunks])
        for item, emb in zip(draft.chunks, embeddings, strict=True):
            session.add(
                KnowledgeChunk(
                    source_pk=source.id,
                    scenario_id=item.scenario_id,
                    chunk_index=item.chunk_index,
                    chunk_text=item.chunk_text,
                    embedding=emb,
                    chunk_metadata=item.metadata,
                )
            )
            chunk_count += 1
        source_count += 1

    await session.commit()
    return source_count, chunk_count

