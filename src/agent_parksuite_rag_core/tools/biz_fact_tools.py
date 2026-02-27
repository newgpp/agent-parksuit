from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx
from loguru import logger

from agent_parksuite_rag_core.clients.biz_api_client import BizApiClient


def _normalize_decimal_str(value: Any) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01")))


class BizFactTools:
    def __init__(self, biz_client: BizApiClient) -> None:
        self.biz_client = biz_client

    async def build_arrears_facts(self, ctx: "BizExecutionContext") -> dict[str, Any]:
        logger.info("tool[arrears_check] start plate_no={} city_code={}", ctx.plate_no, ctx.city_code)
        attempted_tools = ["GET /api/v1/arrears-orders"]
        try:
            rows = await self.biz_client.get_arrears_orders(plate_no=ctx.plate_no, city_code=ctx.city_code)
        except httpx.HTTPStatusError as exc:
            logger.warning("tool[arrears_check] http_error status={}", getattr(exc.response, "status_code", None))
            return {
                "intent": "arrears_check",
                "plate_no": ctx.plate_no,
                "city_code": ctx.city_code,
                "error": "arrears_tool_http_error",
                "attempted_tools": attempted_tools,
            }
        except httpx.RequestError:
            logger.warning("tool[arrears_check] request_error")
            return {
                "intent": "arrears_check",
                "plate_no": ctx.plate_no,
                "city_code": ctx.city_code,
                "error": "arrears_tool_request_error",
                "attempted_tools": attempted_tools,
            }
        logger.info("tool[arrears_check] done count={}", len(rows))
        return {
            "intent": "arrears_check",
            "plate_no": ctx.plate_no,
            "city_code": ctx.city_code,
            "arrears_count": len(rows),
            "arrears_order_nos": [str(item.get("order_no", "")) for item in rows],
            "orders": rows,
            "attempted_tools": attempted_tools,
        }

    async def build_fee_verify_facts(self, ctx: "BizExecutionContext") -> dict[str, Any]:
        if not ctx.order_no:
            logger.info("tool[fee_verify] skip reason=missing_order_no")
            return {
                "intent": "fee_verify",
                "error": "order_no is required for fee_verify",
                "error_detail": "需要提供order_no后才能执行金额核验。",
                "attempted_tools": [],
            }

        logger.info("tool[fee_verify] start order_no={}", ctx.order_no)
        attempted_tools = ["GET /api/v1/parking-orders/{order_no}"]
        try:
            order = await self.biz_client.get_parking_order(order_no=ctx.order_no)
        except httpx.HTTPStatusError as exc:
            status_code = getattr(exc.response, "status_code", None)
            logger.warning("tool[fee_verify] get_order_http_error status={} order_no={}", status_code, ctx.order_no)
            if status_code == 404:
                return {
                    "intent": "fee_verify",
                    "order_no": ctx.order_no,
                    "error": "order_not_found",
                    "attempted_tools": attempted_tools,
                }
            return {
                "intent": "fee_verify",
                "order_no": ctx.order_no,
                "error": "order_tool_http_error",
                "attempted_tools": attempted_tools,
            }
        except httpx.RequestError:
            logger.warning("tool[fee_verify] get_order_request_error order_no={}", ctx.order_no)
            return {
                "intent": "fee_verify",
                "order_no": ctx.order_no,
                "error": "order_tool_request_error",
                "attempted_tools": attempted_tools,
            }
        rule_code = ctx.rule_code or str(order.get("billing_rule_code", ""))

        try:
            entry_time = ctx.entry_time or datetime.fromisoformat(str(order.get("entry_time")))
        except Exception:
            logger.warning("tool[fee_verify] invalid_entry_time order_no={}", ctx.order_no)
            return {"intent": "fee_verify", "error": "entry_time is invalid for fee_verify", "attempted_tools": attempted_tools}

        exit_raw = ctx.exit_time or order.get("exit_time")
        if exit_raw is None:
            logger.warning("tool[fee_verify] missing_exit_time order_no={}", ctx.order_no)
            return {"intent": "fee_verify", "error": "exit_time is required for fee_verify", "attempted_tools": attempted_tools}

        try:
            exit_time = exit_raw if isinstance(exit_raw, datetime) else datetime.fromisoformat(str(exit_raw))
        except Exception:
            logger.warning("tool[fee_verify] invalid_exit_time order_no={}", ctx.order_no)
            return {"intent": "fee_verify", "error": "exit_time is invalid for fee_verify", "attempted_tools": attempted_tools}

        try:
            sim = await self.biz_client.simulate_billing(rule_code=rule_code, entry_time=entry_time, exit_time=exit_time)
            attempted_tools.append("POST /api/v1/billing-rules/simulate")
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "tool[fee_verify] simulate_http_error status={} order_no={}",
                getattr(exc.response, "status_code", None),
                ctx.order_no,
            )
            return {
                "intent": "fee_verify",
                "order_no": ctx.order_no,
                "rule_code": rule_code,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "error": "simulate_tool_http_error",
                "order": order,
                "attempted_tools": attempted_tools,
            }
        except httpx.RequestError:
            logger.warning("tool[fee_verify] simulate_request_error order_no={}", ctx.order_no)
            return {
                "intent": "fee_verify",
                "order_no": ctx.order_no,
                "rule_code": rule_code,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "error": "simulate_tool_request_error",
                "order": order,
                "attempted_tools": attempted_tools,
            }
        order_total = _normalize_decimal_str(order.get("total_amount", "0"))
        sim_total = _normalize_decimal_str(sim.get("total_amount", "0"))
        is_consistent = order_total == sim_total
        logger.info(
            "tool[fee_verify] done order_no={} amount_check_result={}",
            ctx.order_no,
            "一致" if is_consistent else "不一致",
        )
        return {
            "intent": "fee_verify",
            "order_no": ctx.order_no,
            "rule_code": rule_code,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "order_total_amount": order_total,
            "sim_total_amount": sim_total,
            "amount_check_result": "一致" if is_consistent else "不一致",
            "amount_check_action": "自动通过" if is_consistent else "需人工复核",
            "order": order,
            "simulation": sim,
            "attempted_tools": attempted_tools,
        }


@dataclass(frozen=True)
class BizExecutionContext:
    city_code: str | None = None
    lot_code: str | None = None
    plate_no: str | None = None
    order_no: str | None = None
    rule_code: str | None = None
    entry_time: datetime | None = None
    exit_time: datetime | None = None
