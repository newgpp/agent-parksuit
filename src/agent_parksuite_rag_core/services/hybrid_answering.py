from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.schemas.rag import HybridAnswerRequest, RetrieveResponseItem
from agent_parksuite_rag_core.services.answering import _extract_json_payload, generate_hybrid_answer
from agent_parksuite_rag_core.services.biz_tools import BizApiClient
from agent_parksuite_rag_core.workflows.hybrid_answer import HybridGraphState, run_hybrid_workflow

RetrieveFn = Callable[[HybridAnswerRequest], Awaitable[list[RetrieveResponseItem]]]


def _log_payload_text(text: str) -> str:
    if settings.llm_log_full_payload:
        return text
    return text[: settings.llm_log_max_chars]


def _normalize_decimal_str(value: Any) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01")))


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


async def _build_arrears_facts(payload: HybridAnswerRequest, biz_client: BizApiClient) -> dict[str, Any]:
    logger.info("hybrid biz_tool=arrears_orders start plate_no={} city_code={}", payload.plate_no, payload.city_code)
    rows = await biz_client.get_arrears_orders(plate_no=payload.plate_no, city_code=payload.city_code)
    logger.info("hybrid biz_tool=arrears_orders done count={}", len(rows))
    return {
        "intent": "arrears_check",
        "plate_no": payload.plate_no,
        "city_code": payload.city_code,
        "arrears_count": len(rows),
        "arrears_order_nos": [str(item.get("order_no", "")) for item in rows],
        "orders": rows,
    }


async def _build_fee_verify_facts(payload: HybridAnswerRequest, biz_client: BizApiClient) -> dict[str, Any]:
    if not payload.order_no:
        logger.info("hybrid biz_tool=fee_verify skip reason=missing_order_no")
        return {"intent": "fee_verify", "error": "order_no is required for fee_verify"}

    logger.info("hybrid biz_tool=fee_verify start order_no={}", payload.order_no)
    order = await biz_client.get_parking_order(order_no=payload.order_no)
    rule_code = payload.rule_code or str(order.get("billing_rule_code", ""))

    try:
        entry_time = payload.entry_time or datetime.fromisoformat(str(order.get("entry_time")))
    except Exception:
        logger.warning("hybrid biz_tool=fee_verify invalid_entry_time order_no={}", payload.order_no)
        return {"intent": "fee_verify", "error": "entry_time is invalid for fee_verify"}

    exit_raw = payload.exit_time or order.get("exit_time")
    if exit_raw is None:
        logger.warning("hybrid biz_tool=fee_verify missing_exit_time order_no={}", payload.order_no)
        return {"intent": "fee_verify", "error": "exit_time is required for fee_verify"}

    try:
        exit_time = exit_raw if isinstance(exit_raw, datetime) else datetime.fromisoformat(str(exit_raw))
    except Exception:
        logger.warning("hybrid biz_tool=fee_verify invalid_exit_time order_no={}", payload.order_no)
        return {"intent": "fee_verify", "error": "exit_time is invalid for fee_verify"}

    sim = await biz_client.simulate_billing(rule_code=rule_code, entry_time=entry_time, exit_time=exit_time)
    order_total = _normalize_decimal_str(order.get("total_amount", "0"))
    sim_total = _normalize_decimal_str(sim.get("total_amount", "0"))
    is_consistent = order_total == sim_total
    logger.info(
        "hybrid biz_tool=fee_verify done order_no={} amount_check_result={}",
        payload.order_no,
        "一致" if is_consistent else "不一致",
    )
    return {
        "intent": "fee_verify",
        "order_no": payload.order_no,
        "rule_code": rule_code,
        "entry_time": entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "order_total_amount": order_total,
        "sim_total_amount": sim_total,
        "amount_check_result": "一致" if is_consistent else "不一致",
        "amount_check_action": "自动通过" if is_consistent else "需人工复核",
        "order": order,
        "simulation": sim,
    }


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

    async def _arrears_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await _build_arrears_facts(p, biz_client)

    async def _fee_facts_fn(p: HybridAnswerRequest) -> dict[str, Any]:
        return await _build_fee_verify_facts(p, biz_client)

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
