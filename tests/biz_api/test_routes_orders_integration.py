from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _create_billing_rule_for_lot(async_client: AsyncClient, rule_code: str, lot_code: str) -> None:
    payload = {
        "rule_code": rule_code,
        "name": f"{lot_code} 规则",
        "status": "enabled",
        "scope": {
            "scope_type": "lot_code",
            "city_code": "310100",
            "lot_codes": [lot_code],
        },
        "version": {
            "effective_from": "2026-01-01T00:00:00+08:00",
            "effective_to": None,
            "priority": 100,
            "rule_payload": [],
        },
    }
    resp = await async_client.post("/api/v1/billing-rules", json=payload)
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_create_parking_order_should_compute_arrears(async_client: AsyncClient, uniq: str) -> None:
    lot_code = f"LOT-A-{uniq}"
    rule_code = f"RULE-LOT-A-{uniq}"
    await _create_billing_rule_for_lot(async_client, rule_code, lot_code)

    payload = {
        "order_no": f"ORDER-001-{uniq}",
        "plate_no": "沪A12345",
        "city_code": "310100",
        "lot_code": lot_code,
        "billing_rule_code": rule_code,
        "billing_rule_version_no": 1,
        "entry_time": "2026-02-01T09:00:00+08:00",
        "exit_time": "2026-02-01T10:00:00+08:00",
        "total_amount": "50.00",
        "paid_amount": "20.00",
        "status": "UNPAID",
    }

    resp = await async_client.post("/api/v1/parking-orders", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["order_no"] == f"ORDER-001-{uniq}"
    assert data["arrears_amount"] == "30.00"


@pytest.mark.anyio
async def test_get_parking_order_detail(async_client: AsyncClient, uniq: str) -> None:
    lot_code = f"LOT-B-{uniq}"
    rule_code = f"RULE-LOT-B-{uniq}"
    order_no = f"ORDER-DETAIL-{uniq}"
    await _create_billing_rule_for_lot(async_client, rule_code, lot_code)

    payload = {
        "order_no": order_no,
        "plate_no": "沪B67890",
        "city_code": "310100",
        "lot_code": lot_code,
        "billing_rule_code": rule_code,
        "billing_rule_version_no": 1,
        "entry_time": "2026-02-02T09:00:00+08:00",
        "exit_time": "2026-02-02T11:00:00+08:00",
        "total_amount": "40.00",
        "paid_amount": "40.00",
        "status": "PAID",
    }
    assert (await async_client.post("/api/v1/parking-orders", json=payload)).status_code == 200

    detail = await async_client.get(f"/api/v1/parking-orders/{order_no}")
    assert detail.status_code == 200
    data = detail.json()
    assert data["order_no"] == order_no
    assert data["status"] == "PAID"


@pytest.mark.anyio
async def test_list_arrears_orders_filter_by_plate_and_city(async_client: AsyncClient, uniq: str) -> None:
    lot_c = f"LOT-C-{uniq}"
    lot_d = f"LOT-D-{uniq}"
    lot_e = f"LOT-E-{uniq}"
    rule_c = f"RULE-LOT-C-{uniq}"
    rule_d = f"RULE-LOT-D-{uniq}"
    rule_e = f"RULE-LOT-E-{uniq}"
    plate_no = f"沪C{uniq[:5]}"

    await _create_billing_rule_for_lot(async_client, rule_c, lot_c)
    await _create_billing_rule_for_lot(async_client, rule_d, lot_d)
    await _create_billing_rule_for_lot(async_client, rule_e, lot_e)

    arrears_order = {
        "order_no": f"ORDER-ARREARS-1-{uniq}",
        "plate_no": plate_no,
        "city_code": "310100",
        "lot_code": lot_c,
        "billing_rule_code": rule_c,
        "billing_rule_version_no": 1,
        "entry_time": "2026-02-03T09:00:00+08:00",
        "exit_time": "2026-02-03T10:00:00+08:00",
        "total_amount": "60.00",
        "paid_amount": "10.00",
        "status": "UNPAID",
    }
    paid_order = {
        "order_no": f"ORDER-PAID-1-{uniq}",
        "plate_no": plate_no,
        "city_code": "310100",
        "lot_code": lot_d,
        "billing_rule_code": rule_d,
        "billing_rule_version_no": 1,
        "entry_time": "2026-02-03T11:00:00+08:00",
        "exit_time": "2026-02-03T12:00:00+08:00",
        "total_amount": "20.00",
        "paid_amount": "20.00",
        "status": "PAID",
    }
    other_city_order = {
        "order_no": f"ORDER-ARREARS-OTHER-{uniq}",
        "plate_no": plate_no,
        "city_code": "320100",
        "lot_code": lot_e,
        "billing_rule_code": rule_e,
        "billing_rule_version_no": 1,
        "entry_time": "2026-02-03T13:00:00+08:00",
        "exit_time": "2026-02-03T14:00:00+08:00",
        "total_amount": "30.00",
        "paid_amount": "0.00",
        "status": "UNPAID",
    }

    assert (await async_client.post("/api/v1/parking-orders", json=arrears_order)).status_code == 200
    assert (await async_client.post("/api/v1/parking-orders", json=paid_order)).status_code == 200
    assert (await async_client.post("/api/v1/parking-orders", json=other_city_order)).status_code == 200

    resp = await async_client.get(
        "/api/v1/arrears-orders",
        params={"plate_no": plate_no, "city_code": "310100"},
    )
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["order_no"] == f"ORDER-ARREARS-1-{uniq}"
    assert rows[0]["arrears_amount"] == "50.00"
