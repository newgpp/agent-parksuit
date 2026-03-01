from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import AsyncIterator


@dataclass
class SessionQueueAcquireInfo:
    waited: bool
    waited_ms: int


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock
    refs: int = 0


class SessionExecutionQueue:
    """Serialize execution for the same session_id while keeping cross-session concurrency."""

    def __init__(self) -> None:
        self._entries: dict[str, _SessionLockEntry] = {}
        self._guard = asyncio.Lock()

    async def _get_or_create_entry(self, session_id: str) -> _SessionLockEntry:
        async with self._guard:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = _SessionLockEntry(lock=asyncio.Lock())
                self._entries[session_id] = entry
            entry.refs += 1
            return entry

    async def _release_entry_ref(self, session_id: str, entry: _SessionLockEntry) -> None:
        async with self._guard:
            entry.refs -= 1
            if entry.refs <= 0 and not entry.lock.locked():
                self._entries.pop(session_id, None)

    def active_session_count(self) -> int:
        return len(self._entries)

    @asynccontextmanager
    async def session(self, session_id: str | None) -> AsyncIterator[SessionQueueAcquireInfo]:
        sid = (session_id or "").strip()
        if not sid:
            yield SessionQueueAcquireInfo(waited=False, waited_ms=0)
            return

        entry = await self._get_or_create_entry(sid)
        waited = entry.lock.locked()
        started = perf_counter()
        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            waited_ms = max(0, int((perf_counter() - started) * 1000))
            yield SessionQueueAcquireInfo(waited=waited, waited_ms=waited_ms)
        finally:
            if acquired:
                entry.lock.release()
            await self._release_entry_ref(sid, entry)


_HYBRID_SESSION_QUEUE = SessionExecutionQueue()


def get_hybrid_session_queue() -> SessionExecutionQueue:
    return _HYBRID_SESSION_QUEUE
