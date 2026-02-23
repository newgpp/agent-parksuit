from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from agent_parksuite_rag_core.api.routes import router
from agent_parksuite_rag_core.db.base import Base
from agent_parksuite_rag_core.db.session import get_db_session


def _keep_test_data_enabled() -> bool:
    return os.getenv("KEEP_TEST_DATA", "0") == "1"


@pytest.fixture(scope="module")
async def rag_engine() -> AsyncGenerator[AsyncEngine, None]:
    rag_test_database_url = os.getenv(
        "RAG_TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test",
    )
    test_engine = create_async_engine(rag_test_database_url, echo=False, future=True)
    try:
        async with test_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        await test_engine.dispose()
        pytest.skip(f"RAG core integration tests skipped: cannot connect test DB ({exc})")

    yield test_engine

    if not _keep_test_data_enabled():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture(scope="function")
async def rag_db_session(rag_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    session_maker = async_sessionmaker(bind=rag_engine, expire_on_commit=False, class_=AsyncSession)
    if not _keep_test_data_enabled():
        async with rag_engine.begin() as conn:
            for table in ("knowledge_chunks", "knowledge_sources"):
                await conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
    async with session_maker() as session:
        yield session


@pytest.fixture(scope="function")
async def rag_async_client(rag_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        yield rag_db_session

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db_session] = _override_get_db_session
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
