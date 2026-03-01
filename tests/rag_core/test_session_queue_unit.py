from __future__ import annotations

import asyncio
from time import perf_counter

import pytest

from agent_parksuite_rag_core.services.session_queue import SessionExecutionQueue


@pytest.mark.anyio
async def test_session_queue_should_serialize_same_session() -> None:
    queue = SessionExecutionQueue()
    events: list[str] = []
    acquire_info: dict[str, tuple[bool, int]] = {}

    async def _worker(name: str, hold_seconds: float) -> None:
        async with queue.session("ses-001") as info:
            acquire_info[name] = (info.waited, info.waited_ms)
            events.append(f"start:{name}")
            await asyncio.sleep(hold_seconds)
            events.append(f"end:{name}")

    t1 = asyncio.create_task(_worker("w1", 0.05))
    await asyncio.sleep(0.005)
    t2 = asyncio.create_task(_worker("w2", 0.0))
    await asyncio.gather(t1, t2)

    assert events == ["start:w1", "end:w1", "start:w2", "end:w2"]
    assert acquire_info["w1"][0] is False
    assert acquire_info["w2"][0] is True
    assert queue.active_session_count() == 0


@pytest.mark.anyio
async def test_session_queue_should_allow_parallel_different_sessions() -> None:
    queue = SessionExecutionQueue()

    async def _worker(session_id: str, hold_seconds: float) -> None:
        async with queue.session(session_id):
            await asyncio.sleep(hold_seconds)

    started = perf_counter()
    await asyncio.gather(
        _worker("ses-A", 0.05),
        _worker("ses-B", 0.05),
    )
    elapsed = perf_counter() - started

    assert elapsed < 0.09
    assert queue.active_session_count() == 0
