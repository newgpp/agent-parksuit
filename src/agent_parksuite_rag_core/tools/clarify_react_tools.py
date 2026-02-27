from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import tool

from agent_parksuite_rag_core.clients.biz_api_client import BizApiClient, get_biz_client


def get_clarify_react_biz_client() -> BizApiClient:
    return get_biz_client()


@tool("lookup_order")
async def lookup_order(order_no: str) -> dict[str, Any]:
    """按订单号查询订单是否存在。"""
    biz_client = get_clarify_react_biz_client()
    normalized_order_no = (order_no or "").strip().upper()
    if not normalized_order_no:
        return {"tool": "lookup_order", "hit": False, "reason": "missing_order_no"}
    try:
        order = await biz_client.get_parking_order(order_no=normalized_order_no)
    except httpx.HTTPStatusError as exc:
        status = getattr(exc.response, "status_code", None)
        return {
            "tool": "lookup_order",
            "hit": False,
            "order_no": normalized_order_no,
            "reason": f"http_{status}",
        }
    except httpx.RequestError:
        return {
            "tool": "lookup_order",
            "hit": False,
            "order_no": normalized_order_no,
            "reason": "request_error",
        }
    return {
        "tool": "lookup_order",
        "hit": True,
        "order_no": normalized_order_no,
        "plate_no": order.get("plate_no"),
        "city_code": order.get("city_code"),
        "lot_code": order.get("lot_code"),
    }


@tool("query_billing_rules_by_params")
async def query_billing_rules_by_params(lot_code: str, city_code: str | None = None) -> dict[str, Any]:
    """按停车场编码（可选城市）通过 /billing-rules 查询是否存在匹配规则。"""
    biz_client = get_clarify_react_biz_client()
    normalized_lot_code = (lot_code or "").strip().upper()
    normalized_city_code = (city_code or "").strip() or None
    if not normalized_lot_code:
        return {"tool": "query_billing_rules_by_params", "hit": False, "reason": "missing_lot_code"}
    try:
        rows = await biz_client.get_billing_rules(city_code=normalized_city_code, lot_code=normalized_lot_code)
    except httpx.HTTPStatusError as exc:
        status = getattr(exc.response, "status_code", None)
        return {
            "tool": "query_billing_rules_by_params",
            "hit": False,
            "lot_code": normalized_lot_code,
            "city_code": normalized_city_code,
            "reason": f"http_{status}",
        }
    except httpx.RequestError:
        return {
            "tool": "query_billing_rules_by_params",
            "hit": False,
            "lot_code": normalized_lot_code,
            "city_code": normalized_city_code,
            "reason": "request_error",
        }
    if not rows:
        return {
            "tool": "query_billing_rules_by_params",
            "hit": False,
            "lot_code": normalized_lot_code,
            "city_code": normalized_city_code,
            "reason": "rule_not_found",
        }
    return {
        "tool": "query_billing_rules_by_params",
        "hit": True,
        "lot_code": normalized_lot_code,
        "city_code": normalized_city_code,
        "matched_rule_count": len(rows),
        "rule_codes": [str(item.get("rule_code", "")) for item in rows if isinstance(item, dict)],
    }


def build_clarify_react_tools(
) -> list[Any]:
    return [lookup_order, query_billing_rules_by_params]
