from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.services.ingestion import (
    DeterministicEmbedder,
    OpenAIEmbedder,
    build_sources_from_markdown,
    build_sources_from_scenarios,
    read_jsonl,
    upsert_sources_and_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-002 ingestion pipeline")
    parser.add_argument(
        "--database-url",
        default=os.getenv("RAG_DATABASE_URL", settings.database_url),
        help="target rag database url",
    )
    parser.add_argument(
        "--input-type",
        choices=["scenarios_jsonl", "markdown"],
        default="scenarios_jsonl",
        help="input source type",
    )
    parser.add_argument(
        "--input-path",
        default="data/rag000/scenarios.jsonl",
        help="jsonl file path or markdown directory path",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["deterministic", "openai"],
        default=os.getenv("RAG_INGEST_EMBEDDING_PROVIDER", "deterministic"),
        help="embedding provider",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAG_INGEST_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="embedding model when provider=openai",
    )
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace chunks for existing source_id",
    )
    parser.add_argument(
        "--source-prefix",
        default="MD",
        help="source prefix for markdown import",
    )
    return parser.parse_args()


def _build_drafts(args: argparse.Namespace):
    input_path = Path(args.input_path)
    if args.input_type == "scenarios_jsonl":
        rows = read_jsonl(input_path)
        return build_sources_from_scenarios(
            rows=rows,
            source_uri=str(input_path),
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )

    md_files = sorted(input_path.rglob("*.md")) if input_path.is_dir() else [input_path]
    return build_sources_from_markdown(
        files=md_files,
        source_prefix=args.source_prefix,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )


def _build_embedder(args: argparse.Namespace):
    if args.embedding_provider == "openai":
        return OpenAIEmbedder(model=args.embedding_model)
    return DeterministicEmbedder(dim=settings.embedding_dim)


async def run() -> None:
    args = parse_args()
    drafts = _build_drafts(args)
    if not drafts:
        print("[warn] no drafts generated")
        return

    embedder = _build_embedder(args)
    engine = create_async_engine(args.database_url, echo=False, future=True)
    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

    try:
        async with session_maker() as session:
            source_count, chunk_count = await upsert_sources_and_chunks(
                session=session,
                drafts=drafts,
                embedder=embedder,
                replace_existing=args.replace_existing,
            )
        print(f"[ok] sources={source_count}, chunks={chunk_count}, db={args.database_url}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())

