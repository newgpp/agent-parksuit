from __future__ import annotations

import time
from typing import Any, Protocol, TypedDict


class SessionMemoryState(TypedDict, total=False):
    slots: dict[str, Any]
    last_intent: str
    order_candidates: list[str]
    turns: list[dict[str, Any]]


class SessionMemoryRepo(Protocol):
    async def get_session(self, session_id: str) -> SessionMemoryState | None:
        ...

    async def save_session(self, session_id: str, state: SessionMemoryState, ttl_seconds: int) -> None:
        ...


class InMemorySessionMemoryRepo:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, SessionMemoryState]] = {}

    async def get_session(self, session_id: str) -> SessionMemoryState | None:
        now = time.time()
        item = self._store.get(session_id)
        if not item:
            return None
        expires_at, state = item
        if expires_at <= now:
            self._store.pop(session_id, None)
            return None
        return state

    async def save_session(self, session_id: str, state: SessionMemoryState, ttl_seconds: int) -> None:
        ttl = max(int(ttl_seconds), 1)
        self._store[session_id] = (time.time() + ttl, state)


_repo_singleton: SessionMemoryRepo = InMemorySessionMemoryRepo()


def get_session_memory_repo() -> SessionMemoryRepo:
    return _repo_singleton
