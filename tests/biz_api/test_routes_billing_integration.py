from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_upsert_billing_rule_and_get_detail(async_client: AsyncClient, uniq: str) -> None:
    rule_code = f"RULE-SH-A-{uniq}"
    payload = {
        "rule_code": rule_code,
        "name": "上海A场规则",
        "status": "enabled",
        "scope": {
            "scope_type": "lot_code",
            "city_code": "310100",
            "lot_codes": ["LOT-A", "LOT-B"],
        },
        "version": {
            "effective_from": "2026-02-01T00:00:00+08:00",
            "effective_to": "2026-12-31T23:59:59+08:00",
            "priority": 100,
            "rule_payload": [
                {
                    "name": "day_periodic",
                    "type": "periodic",
                    "time_window": {"start": "08:00", "end": "20:00"},
                    "unit_minutes": 30,
                    "unit_price": 2,
                    "max_charge": 20,
                }
            ],
        },
    }

    create_resp = await async_client.post("/api/v1/billing-rules", json=payload)
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["rule_code"] == rule_code
    assert len(created["versions"]) == 1
    assert created["versions"][0]["version_no"] == 1

    detail_resp = await async_client.get(f"/api/v1/billing-rules/{rule_code}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["scope"]["lot_codes"] == ["LOT-A", "LOT-B"]


@pytest.mark.anyio
async def test_upsert_billing_rule_overlap_version_should_409(async_client: AsyncClient, uniq: str) -> None:
    rule_code = f"RULE-SH-OVERLAP-{uniq}"
    base_payload = {
        "rule_code": rule_code,
        "name": "规则重叠测试",
        "status": "enabled",
        "scope": {
            "scope_type": "lot_code",
            "city_code": "310100",
            "lot_codes": ["LOT-X"],
        },
        "version": {
            "effective_from": "2026-02-01T00:00:00+08:00",
            "effective_to": "2026-06-30T23:59:59+08:00",
            "priority": 100,
            "rule_payload": [],
        },
    }
    first = await async_client.post("/api/v1/billing-rules", json=base_payload)
    assert first.status_code == 200

    overlap_payload = {
        **base_payload,
        "version": {
            "effective_from": "2026-06-01T00:00:00+08:00",
            "effective_to": "2026-12-31T23:59:59+08:00",
            "priority": 100,
            "rule_payload": [],
        },
    }
    overlap = await async_client.post("/api/v1/billing-rules", json=overlap_payload)
    assert overlap.status_code == 409


@pytest.mark.anyio
async def test_list_billing_rules_filter_by_lot_code(async_client: AsyncClient, uniq: str) -> None:
    lot_a = f"LOT-A-{uniq}"
    lot_b = f"LOT-B-{uniq}"
    payload_a = {
        "rule_code": f"RULE-A-{uniq}",
        "name": "规则A",
        "status": "enabled",
        "scope": {
            "scope_type": "lot_code",
            "city_code": "310100",
            "lot_codes": [lot_a],
        },
        "version": {
            "effective_from": "2026-01-01T00:00:00+08:00",
            "effective_to": None,
            "priority": 100,
            "rule_payload": [],
        },
    }
    payload_b = {
        "rule_code": f"RULE-B-{uniq}",
        "name": "规则B",
        "status": "enabled",
        "scope": {
            "scope_type": "lot_code",
            "city_code": "310100",
            "lot_codes": [lot_b],
        },
        "version": {
            "effective_from": "2026-01-01T00:00:00+08:00",
            "effective_to": None,
            "priority": 100,
            "rule_payload": [],
        },
    }

    assert (await async_client.post("/api/v1/billing-rules", json=payload_a)).status_code == 200
    assert (await async_client.post("/api/v1/billing-rules", json=payload_b)).status_code == 200

    query = await async_client.get("/api/v1/billing-rules", params={"lot_code": lot_a})
    assert query.status_code == 200
    rows = query.json()
    assert len(rows) == 1
    assert rows[0]["rule_code"] == f"RULE-A-{uniq}"


@pytest.mark.anyio
async def test_simulate_billing_hit_version_and_return_breakdown(async_client: AsyncClient, uniq: str) -> None:
    rule_code = f"RULE-SIM-{uniq}"
    payload = {
        "rule_code": rule_code,
        "name": "模拟计费规则",
        "status": "enabled",
        "scope": {
            "scope_type": "lot_code",
            "city_code": "310100",
            "lot_codes": ["LOT-SIM"],
        },
        "version": {
            "effective_from": "2026-01-01T00:00:00+08:00",
            "effective_to": None,
            "priority": 100,
            "rule_payload": [
                {
                    "name": "day_periodic",
                    "type": "periodic",
                    "time_window": {"start": "08:00", "end": "20:00"},
                    "unit_minutes": 30,
                    "unit_price": 2,
                    "max_charge": 20,
                }
            ],
        },
    }
    assert (await async_client.post("/api/v1/billing-rules", json=payload)).status_code == 200

    simulate_req = {
        "rule_code": rule_code,
        "entry_time": "2026-02-01T09:00:00+08:00",
        "exit_time": "2026-02-01T11:10:00+08:00",
    }
    simulate_resp = await async_client.post("/api/v1/billing-rules/simulate", json=simulate_req)
    assert simulate_resp.status_code == 200
    data = simulate_resp.json()
    assert data["matched_version_no"] == 1
    assert data["total_amount"] == "10.00"
    assert data["breakdown"][0]["segment_name"] == "day_periodic"
