from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
        logger.info("tool[biz_api] request method=GET url=%s params=%s", url, params)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.get(url, params=params)
            logger.info(
                "tool[biz_api] response method=GET url=%s status=%d body=%s",
                url,
                resp.status_code,
                resp.text,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []

    async def get_parking_order(self, order_no: str) -> dict[str, Any]:
        url = f"{self.base_url}/api/v1/parking-orders/{order_no}"
        logger.info("tool[biz_api] request method=GET url=%s", url)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.get(url)
            logger.info(
                "tool[biz_api] response method=GET url=%s status=%d body=%s",
                url,
                resp.status_code,
                resp.text,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}

    async def simulate_billing(self, rule_code: str, entry_time: datetime, exit_time: datetime) -> dict[str, Any]:
        payload = {
            "rule_code": rule_code,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
        }
        url = f"{self.base_url}/api/v1/billing-rules/simulate"
        logger.info("tool[biz_api] request method=POST url=%s json=%s", url, payload)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            logger.info(
                "tool[biz_api] response method=POST url=%s status=%d body=%s",
                url,
                resp.status_code,
                resp.text,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}
