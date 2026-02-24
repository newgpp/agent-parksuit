from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.retrieve import RetrieveResponseItem


def _log_payload_text(text: str) -> str:
    if settings.llm_log_full_payload:
        return text
    return text[: settings.llm_log_max_chars]


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _render_context(items: list[RetrieveResponseItem]) -> str:
    blocks = []
    for idx, item in enumerate(items, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] source_id={item.source_id} chunk_id={item.chunk_id}",
                    f"doc_type={item.doc_type} title={item.title}",
                    f"score={item.score}",
                    f"content={item.content}",
                ]
            )
        )
    return "\n\n".join(blocks)


async def generate_answer_from_chunks(
    query: str,
    items: list[RetrieveResponseItem],
) -> tuple[str, list[str], str]:
    if not settings.deepseek_api_key:
        raise RuntimeError("RAG_DEEPSEEK_API_KEY is not configured")

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )
    context = _render_context(items)
    messages = [
        SystemMessage(
            content=(
                "你是停车业务知识助手。仅基于给定证据回答，禁止编造。"
                "输出严格 JSON: {\"conclusion\": string, \"key_points\": [string,...]}。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题:\n{query}\n\n"
                f"证据片段:\n{context}\n\n"
                "请给出结论和要点。"
            )
        ),
    ]
    logger.info(
        "llm[answer] input query={} evidence_count={} model={}",
        query[:200],
        len(items),
        settings.deepseek_model,
    )
    logger.info(
        "llm[answer] input_prompt={}",
        _log_payload_text(messages[1].content),
    )
    result = await llm.ainvoke(messages)
    raw_text = str(result.content)
    logger.info("llm[answer] output raw={}", _log_payload_text(raw_text))
    parsed = _extract_json_payload(raw_text)
    if not parsed:
        logger.info("llm[answer] parse_result=raw_text_fallback")
        return raw_text.strip(), [], settings.deepseek_model

    conclusion = str(parsed.get("conclusion", "")).strip() or "未生成结论"
    key_points_raw = parsed.get("key_points", [])
    if isinstance(key_points_raw, list):
        key_points = [str(item).strip() for item in key_points_raw if str(item).strip()]
    else:
        key_points = []
    logger.info("llm[answer] parse_result=json key_points={}", len(key_points))
    return conclusion, key_points, settings.deepseek_model


async def generate_hybrid_answer(
    query: str,
    items: list[RetrieveResponseItem],
    business_facts: dict[str, Any],
    intent: str = "rule_explain",
    request_id: str = "",
) -> tuple[str, list[str], str]:
    if not settings.deepseek_api_key:
        raise RuntimeError("RAG_DEEPSEEK_API_KEY is not configured")

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
    )
    context = _render_context(items)
    facts = json.dumps(business_facts, ensure_ascii=False)
    messages = [
        SystemMessage(
            content=(
                "你是停车业务混合助手。必须同时参考业务事实和证据片段，禁止编造。"
                "输出严格 JSON: {\"conclusion\": string, \"key_points\": [string,...]}。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题:\n{query}\n\n"
                f"意图类型:\n{intent}\n\n"
                f"业务事实:\n{facts}\n\n"
                f"证据片段:\n{context}\n\n"
                "请生成结论和要点。"
            )
        ),
    ]
    logger.info(
        "llm[hybrid][{}] input intent={} query={} evidence_count={} facts={} model={}",
        request_id,
        intent,
        query[:200],
        len(items),
        sorted(business_facts.keys()),
        settings.deepseek_model,
    )
    logger.info(
        "llm[hybrid][{}] input_prompt={}",
        request_id,
        _log_payload_text(messages[1].content),
    )
    result = await llm.ainvoke(messages)
    raw_text = str(result.content)
    logger.info("llm[hybrid][{}] output raw={}", request_id, _log_payload_text(raw_text))
    parsed = _extract_json_payload(raw_text)
    if not parsed:
        logger.info("llm[hybrid][{}] parse_result=raw_text_fallback", request_id)
        return raw_text.strip(), [], settings.deepseek_model

    conclusion = str(parsed.get("conclusion", "")).strip() or "未生成结论"
    key_points_raw = parsed.get("key_points", [])
    if isinstance(key_points_raw, list):
        key_points = [str(item).strip() for item in key_points_raw if str(item).strip()]
    else:
        key_points = []
    logger.info("llm[hybrid][{}] parse_result=json key_points={}", request_id, len(key_points))
    return conclusion, key_points, settings.deepseek_model
