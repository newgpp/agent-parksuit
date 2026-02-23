from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from agent_parksuite_biz_api.db.base import Base
from agent_parksuite_biz_api.db.session import get_db_session
from agent_parksuite_biz_api.main import app


def _keep_test_data_enabled() -> bool:
    return os.getenv("KEEP_TEST_DATA", "0") == "1"


@pytest.fixture(scope="function")
def uniq() -> str:
    return uuid4().hex[:8]


@pytest.fixture(scope="session")
def biz_test_database_url() -> str:
    return os.getenv(
        "BIZ_TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_test",
    )


@pytest.fixture(scope="module")
async def engine(biz_test_database_url: str) -> AsyncGenerator[AsyncEngine, None]:
    test_engine = create_async_engine(biz_test_database_url, echo=False, future=True)
    try:
        async with test_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:
        await test_engine.dispose()
        pytest.skip(f"Biz API integration tests skipped: cannot connect test DB ({exc})")

    yield test_engine

    if not _keep_test_data_enabled():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture(scope="function")
async def db_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

    if not _keep_test_data_enabled():
        async with engine.begin() as conn:
            for table in ("parking_orders", "billing_rule_versions", "billing_rules"):
                await conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))

    async with session_maker() as session:
        yield session


@pytest.fixture(scope="function")
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db_session] = _override_get_db_session
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
