from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from loguru import logger

from agent_parksuite_rag_core.clients.biz_api_client import BizApiClient
from agent_parksuite_rag_core.schemas.rag import HybridAnswerRequest


def _normalize_decimal_str(value: Any) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01")))


class BizFactTools:
    def __init__(self, biz_client: BizApiClient) -> None:
        self.biz_client = biz_client

    async def build_arrears_facts(self, payload: HybridAnswerRequest) -> dict[str, Any]:
        logger.info("tool[arrears_check] start plate_no={} city_code={}", payload.plate_no, payload.city_code)
        rows = await self.biz_client.get_arrears_orders(plate_no=payload.plate_no, city_code=payload.city_code)
        logger.info("tool[arrears_check] done count={}", len(rows))
        return {
            "intent": "arrears_check",
            "plate_no": payload.plate_no,
            "city_code": payload.city_code,
            "arrears_count": len(rows),
            "arrears_order_nos": [str(item.get("order_no", "")) for item in rows],
            "orders": rows,
        }

    async def build_fee_verify_facts(self, payload: HybridAnswerRequest) -> dict[str, Any]:
        if not payload.order_no:
            logger.info("tool[fee_verify] skip reason=missing_order_no")
            return {"intent": "fee_verify", "error": "order_no is required for fee_verify"}

        logger.info("tool[fee_verify] start order_no={}", payload.order_no)
        order = await self.biz_client.get_parking_order(order_no=payload.order_no)
        rule_code = payload.rule_code or str(order.get("billing_rule_code", ""))

        try:
            entry_time = payload.entry_time or datetime.fromisoformat(str(order.get("entry_time")))
        except Exception:
            logger.warning("tool[fee_verify] invalid_entry_time order_no={}", payload.order_no)
            return {"intent": "fee_verify", "error": "entry_time is invalid for fee_verify"}

        exit_raw = payload.exit_time or order.get("exit_time")
        if exit_raw is None:
            logger.warning("tool[fee_verify] missing_exit_time order_no={}", payload.order_no)
            return {"intent": "fee_verify", "error": "exit_time is required for fee_verify"}

        try:
            exit_time = exit_raw if isinstance(exit_raw, datetime) else datetime.fromisoformat(str(exit_raw))
        except Exception:
            logger.warning("tool[fee_verify] invalid_exit_time order_no={}", payload.order_no)
            return {"intent": "fee_verify", "error": "exit_time is invalid for fee_verify"}

        sim = await self.biz_client.simulate_billing(rule_code=rule_code, entry_time=entry_time, exit_time=exit_time)
        order_total = _normalize_decimal_str(order.get("total_amount", "0"))
        sim_total = _normalize_decimal_str(sim.get("total_amount", "0"))
        is_consistent = order_total == sim_total
        logger.info(
            "tool[fee_verify] done order_no={} amount_check_result={}",
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

