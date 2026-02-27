from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import httpx
from loguru import logger

from agent_parksuite_common.observability import current_trace_headers
from agent_parksuite_rag_core.config import settings


class BizApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def get_arrears_orders(self, plate_no: str | None, city_code: str | None) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if plate_no:
            params["plate_no"] = plate_no
        if city_code:
            params["city_code"] = city_code
        url = f"{self.base_url}/api/v1/arrears-orders"
        headers = current_trace_headers()
        logger.info("client[biz_api] request method=GET url={} params={} headers={}", url, params, headers)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            resp = await client.get(url, params=params, headers=headers)
            logger.info(
                "client[biz_api] response method=GET url={} status={} body={}",
                url,
                resp.status_code,
                resp.text,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []

    async def get_billing_rules(self, city_code: str | None, lot_code: str | None) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if city_code:
            params["city_code"] = city_code
        if lot_code:
            params["lot_code"] = lot_code
        url = f"{self.base_url}/api/v1/billing-rules"
        headers = current_trace_headers()
        logger.info("client[biz_api] request method=GET url={} params={} headers={}", url, params, headers)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            resp = await client.get(url, params=params, headers=headers)
            logger.info(
                "client[biz_api] response method=GET url={} status={} body={}",
                url,
                resp.status_code,
                resp.text,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []

    async def get_parking_order(self, order_no: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/v1/parking-orders/{order_no}"
        headers = current_trace_headers()
        logger.info("client[biz_api] request method=GET url={} headers={}", url, headers)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            resp = await client.get(url, headers=headers)
            logger.info(
                "client[biz_api] response method=GET url={} status={} body={}",
                url,
                resp.status_code,
                resp.text,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def build_biz_client(base_url: str, timeout_seconds: float) -> BizApiClient:
    return BizApiClient(base_url=base_url, timeout_seconds=timeout_seconds)


def get_biz_client() -> BizApiClient:
    return build_biz_client(
        base_url=settings.biz_api_base_url,
        timeout_seconds=settings.biz_api_timeout_seconds,
    )

    async def simulate_billing(self, rule_code: str, entry_time: datetime, exit_time: datetime) -> dict[str, Any]:
        payload = {
            "rule_code": rule_code,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
        }
        url = f"{self.base_url}/api/v1/billing-rules/simulate"
        headers = current_trace_headers()
        logger.info("client[biz_api] request method=POST url={} json={} headers={}", url, payload, headers)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, trust_env=False) as client:
            resp = await client.post(url, json=payload, headers=headers)
            logger.info(
                "client[biz_api] response method=POST url={} status={} body={}",
                url,
                resp.status_code,
                resp.text,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}
