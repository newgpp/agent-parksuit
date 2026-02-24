from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from agent_parksuite_rag_core.clients.biz_api_client import BizApiClient
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest
from agent_parksuite_rag_core.schemas.retrieve import RetrieveResponseItem
from agent_parksuite_rag_core.services.answering import _extract_json_payload, generate_hybrid_answer
from agent_parksuite_rag_core.tools.biz_fact_tools import BizFactTools
from agent_parksuite_rag_core.workflows.hybrid_answer import HybridGraphState, run_hybrid_workflow

RetrieveFn = Callable[[HybridAnswerRequest], Awaitable[list[RetrieveResponseItem]]]


def _log_payload_text(text: str) -> str:
    if settings.llm_log_full_payload:
        return text
    return text[: settings.llm_log_max_chars]


def _rule_route_intent(payload: HybridAnswerRequest) -> str:
    if payload.intent_hint in {"rule_explain", "arrears_check", "fee_verify"}:
        return payload.intent_hint

    query = payload.query
    if payload.order_no or any(token in query for token in ("核验", "一致", "算错", "金额", "订单")):
        return "fee_verify"
    if payload.plate_no or any(token in query for token in ("欠费", "补缴", "未缴", "车牌")):
        return "arrears_check"
    return "rule_explain"


async def _classify_intent(payload: HybridAnswerRequest, request_id: str = "") -> str:
    if payload.intent_hint in {"rule_explain", "arrears_check", "fee_verify"}:
        logger.info("hybrid[{}] classify source=intent_hint intent={}", request_id, payload.intent_hint)
        return payload.intent_hint

    if not settings.deepseek_api_key:
        intent = _rule_route_intent(payload)
        logger.info("hybrid[{}] classify source=rule_fallback reason=no_api_key intent={}", request_id, intent)
        return intent

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0,
        timeout=8,
    )

    messages = [
        SystemMessage(
            content=(
                "你是停车业务意图分类器。"
                '请只输出JSON: {"intent": "rule_explain|arrears_check|fee_verify"}。'
                "不要输出其他字段。"
            )
        ),
        HumanMessage(
            content=(
                "根据用户问题选择一个最合适意图:\n"
                "- rule_explain: 解释计费规则/政策\n"
                "- arrears_check: 查询是否欠费/欠费订单\n"
                "- fee_verify: 针对订单金额核验、重算、对账\n\n"
                f"用户问题: {payload.query}"
            )
        ),
    ]
    logger.info("llm[intent][{}] input query={}", request_id, payload.query[:200])
    logger.info("llm[intent][{}] input_prompt={}", request_id, _log_payload_text(messages[1].content))

    try:
        result = await llm.ainvoke(messages)
    except Exception as exc:
        intent = _rule_route_intent(payload)
        logger.warning(
            "hybrid[{}] classify source=rule_fallback reason=llm_error intent={} error={}",
            request_id,
            intent,
            exc.__class__.__name__,
        )
        return intent

    raw_text = str(result.content)
    logger.info("llm[intent][{}] output raw={}", request_id, _log_payload_text(raw_text))
    parsed = _extract_json_payload(raw_text)
    intent = str((parsed or {}).get("intent", "")).strip()
    if intent in {"rule_explain", "arrears_check", "fee_verify"}:
        logger.info("hybrid[{}] classify source=llm intent={}", request_id, intent)
        return intent
    fallback_intent = _rule_route_intent(payload)
    logger.info(
        "hybrid[{}] classify source=rule_fallback reason=invalid_output intent={}",
        request_id,
        fallback_intent,
    )
    return fallback_intent


async def run_hybrid_answering(
    payload: HybridAnswerRequest,
    retrieve_fn: RetrieveFn,
    request_id: str = "",
) -> HybridGraphState:
    logger.info(
        "hybrid[{}] start query_len={} top_k={} hint={} city_code={} lot_code={}",
        request_id,
        len(payload.query),
        payload.top_k,
        payload.intent_hint,
        payload.city_code,
        payload.lot_code,
    )
    biz_client = BizApiClient(
        base_url=settings.biz_api_base_url,
        timeout_seconds=settings.biz_api_timeout_seconds,
    )
    fact_tools = BizFactTools(biz_client=biz_client)

    async def _arrears_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await fact_tools.build_arrears_facts(p)

    async def _fee_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await fact_tools.build_fee_verify_facts(p)

    async def _synthesize_fn(
        query: str,
        items: list[RetrieveResponseItem],
        business_facts: dict[str, Any],
        intent: str,
    ) -> tuple[str, list[str], str]:
        return await generate_hybrid_answer(
            query=query,
            items=items,
            business_facts=business_facts,
            intent=intent,
            request_id=request_id,
        )

    async def _classify_fn(p: HybridAnswerRequest) -> str:
        return await _classify_intent(p, request_id=request_id)

    return await run_hybrid_workflow(
        payload=payload,
        retrieve_fn=retrieve_fn,
        classify_fn=_classify_fn,
        arrears_facts_fn=_arrears_facts_fn,
        fee_facts_fn=_fee_facts_fn,
        synthesize_fn=_synthesize_fn,
        request_id=request_id,
    )
