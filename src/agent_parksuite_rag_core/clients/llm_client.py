from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from agent_parksuite_rag_core.config import settings


@lru_cache(maxsize=16)
def _build_chat_llm(
    model: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    timeout_seconds: float | None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout_seconds,
    )


def get_chat_llm(*, temperature: float = 0, timeout_seconds: float | None = None) -> ChatOpenAI:
    return _build_chat_llm(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )
