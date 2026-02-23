from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_parksuite_biz_api.db.base import Base
from agent_parksuite_biz_api.db.models import BillingRule, BillingRuleVersion, ParkingOrder
from agent_parksuite_biz_api.services.billing_engine import simulate_fee


@dataclass
class RuleSeed:
    rule_code: str
    name: str
    city_code: str
    lot_codes: list[str]
    versions: list[dict]


@dataclass
class ScenarioSeed:
    scenario_id: str
    intent_tags: list[str]
    query: str
    city_code: str
    lot_code: str
    plate_no: str
    order_no: str | None
    entry_time: datetime
    exit_time: datetime
    rule_code: str
    paid_amount: Decimal
    force_order_total: Decimal | None = None
    notes: str = ""


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def q(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _scenario_rules() -> list[RuleSeed]:
    return [
        RuleSeed(
            rule_code="SCN-RULE-PERIODIC",
            name="场景规则-日间周期计费",
            city_code="310100",
            lot_codes=["SCN-LOT-A"],
            versions=[
                {
                    "version_no": 1,
                    "effective_from": dt("2026-01-01T00:00:00+08:00"),
                    "effective_to": None,
                    "priority": 100,
                    "rule_payload": [
                        {
                            "name": "day_periodic",
                            "type": "periodic",
                            "time_window": {"start": "08:00", "end": "20:00"},
                            "unit_minutes": 30,
                            "unit_price": 2,
                            "free_minutes": 0,
                            "max_charge": 20,
                        }
                    ],
                }
            ],
        ),
        RuleSeed(
            rule_code="SCN-RULE-DAY-NIGHT",
            name="场景规则-日间收费夜间免费",
            city_code="310100",
            lot_codes=["SCN-LOT-B"],
            versions=[
                {
                    "version_no": 1,
                    "effective_from": dt("2026-01-01T00:00:00+08:00"),
                    "effective_to": None,
                    "priority": 100,
                    "rule_payload": [
                        {
                            "name": "day_periodic",
                            "type": "periodic",
                            "time_window": {"start": "08:00", "end": "20:00"},
                            "unit_minutes": 30,
                            "unit_price": 2,
                            "free_minutes": 0,
                            "max_charge": 20,
                        },
                        {
                            "name": "night_free",
                            "type": "free",
                            "time_window": {"start": "20:00", "end": "08:00"},
                        },
                    ],
                }
            ],
        ),
        RuleSeed(
            rule_code="SCN-RULE-TIERED",
            name="场景规则-阶梯计费",
            city_code="310100",
            lot_codes=["SCN-LOT-C"],
            versions=[
                {
                    "version_no": 1,
                    "effective_from": dt("2026-01-01T00:00:00+08:00"),
                    "effective_to": None,
                    "priority": 100,
                    "rule_payload": [
                        {
                            "name": "day_tiered",
                            "type": "tiered",
                            "time_window": {"start": "08:00", "end": "20:00"},
                            "unit_minutes": 30,
                            "free_minutes": 30,
                            "tiers": [
                                {"start_minute": 0, "end_minute": 120, "unit_price": 2},
                                {"start_minute": 120, "end_minute": None, "unit_price": 3},
                            ],
                            "max_charge": 20,
                        },
                        {
                            "name": "night_free",
                            "type": "free",
                            "time_window": {"start": "20:00", "end": "08:00"},
                        },
                    ],
                }
            ],
        ),
        RuleSeed(
            rule_code="SCN-RULE-DUAL-CAP",
            name="场景规则-日夜双时段封顶",
            city_code="310100",
            lot_codes=["SCN-LOT-D"],
            versions=[
                {
                    "version_no": 1,
                    "effective_from": dt("2026-01-01T00:00:00+08:00"),
                    "effective_to": None,
                    "priority": 100,
                    "rule_payload": [
                        {
                            "name": "day_periodic",
                            "type": "periodic",
                            "time_window": {"start": "08:00", "end": "20:00"},
                            "unit_minutes": 30,
                            "unit_price": 2,
                            "free_minutes": 0,
                            "max_charge": 20,
                        },
                        {
                            "name": "night_periodic",
                            "type": "periodic",
                            "time_window": {"start": "20:00", "end": "08:00"},
                            "unit_minutes": 60,
                            "unit_price": 2,
                            "free_minutes": 0,
                            "max_charge": 10,
                        },
                    ],
                }
            ],
        ),
        RuleSeed(
            rule_code="SCN-RULE-VERSION",
            name="场景规则-版本切换",
            city_code="310100",
            lot_codes=["SCN-LOT-E"],
            versions=[
                {
                    "version_no": 1,
                    "effective_from": dt("2026-01-01T00:00:00+08:00"),
                    "effective_to": dt("2026-02-15T00:00:00+08:00"),
                    "priority": 100,
                    "rule_payload": [
                        {
                            "name": "day_periodic_v1",
                            "type": "periodic",
                            "time_window": {"start": "08:00", "end": "20:00"},
                            "unit_minutes": 30,
                            "unit_price": 2,
                            "free_minutes": 0,
                            "max_charge": 20,
                        }
                    ],
                },
                {
                    "version_no": 2,
                    "effective_from": dt("2026-02-15T00:00:00+08:00"),
                    "effective_to": None,
                    "priority": 100,
                    "rule_payload": [
                        {
                            "name": "day_periodic_v2",
                            "type": "periodic",
                            "time_window": {"start": "08:00", "end": "20:00"},
                            "unit_minutes": 30,
                            "unit_price": 3,
                            "free_minutes": 0,
                            "max_charge": 30,
                        }
                    ],
                },
            ],
        ),
    ]


def _scenario_cases() -> list[ScenarioSeed]:
    return [
        ScenarioSeed("SCN-001", ["rule_explain", "fee_verify"], "早上8点到9点停车，费用怎么算？", "310100", "SCN-LOT-A", "沪SCN001", "SCN-001", dt("2026-02-01T08:00:00+08:00"), dt("2026-02-01T09:00:00+08:00"), "SCN-RULE-PERIODIC", Decimal("0.00"), notes="周期计费整除"),
        ScenarioSeed("SCN-002", ["rule_explain", "fee_verify"], "8点到9点05分停车，为什么不是4元？", "310100", "SCN-LOT-A", "沪SCN002", "SCN-002", dt("2026-02-01T08:00:00+08:00"), dt("2026-02-01T09:05:00+08:00"), "SCN-RULE-PERIODIC", Decimal("0.00"), notes="周期计费非整除进位"),
        ScenarioSeed("SCN-003", ["rule_explain", "fee_verify"], "白天停很久怎么封顶？", "310100", "SCN-LOT-A", "沪SCN003", "SCN-003", dt("2026-02-01T08:00:00+08:00"), dt("2026-02-01T23:00:00+08:00"), "SCN-RULE-PERIODIC", Decimal("0.00"), notes="日间封顶"),
        ScenarioSeed("SCN-004", ["rule_explain", "fee_verify"], "跨两天停车封顶怎么算？", "310100", "SCN-LOT-A", "沪SCN004", "SCN-004", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-03T15:10:00+08:00"), "SCN-RULE-PERIODIC", Decimal("0.00"), notes="跨天封顶重置"),
        ScenarioSeed("SCN-005", ["rule_explain", "fee_verify"], "夜间停车免费吗？", "310100", "SCN-LOT-B", "沪SCN005", "SCN-005", dt("2026-02-01T21:00:00+08:00"), dt("2026-02-02T07:00:00+08:00"), "SCN-RULE-DAY-NIGHT", Decimal("0.00"), notes="夜间免费"),
        ScenarioSeed("SCN-006", ["rule_explain", "fee_verify"], "晚上7点到9点停车怎么算？", "310100", "SCN-LOT-B", "沪SCN006", "SCN-006", dt("2026-02-01T19:00:00+08:00"), dt("2026-02-01T21:00:00+08:00"), "SCN-RULE-DAY-NIGHT", Decimal("0.00"), notes="日夜组合"),
        ScenarioSeed("SCN-007", ["rule_explain", "fee_verify"], "阶梯计费在3小时场景下怎么算？", "310100", "SCN-LOT-C", "沪SCN007", "SCN-007", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-01T12:00:00+08:00"), "SCN-RULE-TIERED", Decimal("0.00"), notes="阶梯计费"),
        ScenarioSeed("SCN-008", ["rule_explain", "fee_verify"], "首30分钟免费，29分钟要收多少？", "310100", "SCN-LOT-C", "沪SCN008", "SCN-008", dt("2026-02-01T08:00:00+08:00"), dt("2026-02-01T08:29:00+08:00"), "SCN-RULE-TIERED", Decimal("0.00"), notes="首30分钟免费边界-29分钟"),
        ScenarioSeed("SCN-010", ["rule_explain", "fee_verify"], "首30分钟免费，30分钟要收多少？", "310100", "SCN-LOT-C", "沪SCN010", "SCN-010", dt("2026-02-01T08:00:00+08:00"), dt("2026-02-01T08:30:00+08:00"), "SCN-RULE-TIERED", Decimal("0.00"), notes="首30分钟免费边界-30分钟"),
        ScenarioSeed("SCN-011", ["rule_explain", "fee_verify"], "首30分钟免费，31分钟要收多少？", "310100", "SCN-LOT-C", "沪SCN011", "SCN-011", dt("2026-02-01T08:00:00+08:00"), dt("2026-02-01T08:31:00+08:00"), "SCN-RULE-TIERED", Decimal("0.00"), notes="首30分钟免费边界-31分钟"),
        ScenarioSeed("SCN-012", ["rule_explain", "fee_verify"], "日夜双时段都收费时怎么封顶？", "310100", "SCN-LOT-D", "沪SCN012", "SCN-012", dt("2026-02-01T19:00:00+08:00"), dt("2026-02-02T07:30:00+08:00"), "SCN-RULE-DUAL-CAP", Decimal("0.00"), notes="日夜双时段组合-各自封顶"),
        ScenarioSeed("SCN-013", ["rule_explain", "fee_verify"], "阶梯计费2小时以内怎么算？", "310100", "SCN-LOT-C", "沪SCN013", "SCN-013", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-01T10:30:00+08:00"), "SCN-RULE-TIERED", Decimal("0.00"), notes="阶梯计费-2小时以内"),
        ScenarioSeed("SCN-014", ["rule_explain", "fee_verify"], "阶梯计费超过2小时怎么算？", "310100", "SCN-LOT-C", "沪SCN014", "SCN-014", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-01T11:30:00+08:00"), "SCN-RULE-TIERED", Decimal("0.00"), notes="阶梯计费-超过2小时"),
        ScenarioSeed("SCN-015", ["rule_explain", "fee_verify"], "这个停车场2月14号按哪个规则版本？", "310100", "SCN-LOT-E", "沪SCN015", "SCN-015", dt("2026-02-14T10:00:00+08:00"), dt("2026-02-14T11:00:00+08:00"), "SCN-RULE-VERSION", Decimal("0.00"), notes="规则版本切换-生效前"),
        ScenarioSeed("SCN-016", ["rule_explain", "fee_verify"], "这个停车场2月15号按哪个规则版本？", "310100", "SCN-LOT-E", "沪SCN016", "SCN-016", dt("2026-02-15T10:00:00+08:00"), dt("2026-02-15T11:00:00+08:00"), "SCN-RULE-VERSION", Decimal("0.00"), notes="规则版本切换-生效后"),
        ScenarioSeed("SCN-017", ["rule_explain", "fee_verify"], "同城不同停车场为什么收费不同（A场）？", "310100", "SCN-LOT-A", "沪SCN017", "SCN-017", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-01T11:00:00+08:00"), "SCN-RULE-PERIODIC", Decimal("0.00"), notes="同城不同lot_code差异-A"),
        ScenarioSeed("SCN-018", ["rule_explain", "fee_verify"], "同城不同停车场为什么收费不同（C场）？", "310100", "SCN-LOT-C", "沪SCN018", "SCN-018", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-01T11:00:00+08:00"), "SCN-RULE-TIERED", Decimal("0.00"), notes="同城不同lot_code差异-C"),
        ScenarioSeed("SCN-019", ["fee_verify"], "这个订单金额和模拟金额一致吗？", "310100", "SCN-LOT-A", "沪SCN019", "SCN-019", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-01T10:00:00+08:00"), "SCN-RULE-PERIODIC", Decimal("0.00"), notes="订单金额与模拟结果一致"),
        ScenarioSeed("SCN-020", ["fee_verify"], "这个订单金额是否算错了？", "310100", "SCN-LOT-A", "沪SCN020", "SCN-020", dt("2026-02-01T09:00:00+08:00"), dt("2026-02-01T10:00:00+08:00"), "SCN-RULE-PERIODIC", Decimal("0.00"), force_order_total=Decimal("6.00"), notes="订单金额与模拟结果不一致，需人工复核"),
    ]


def _scenario_009_orders() -> list[dict]:
    return [
        {
            "order_no": "SCN-009-A",
            "plate_no": "沪SCN009",
            "city_code": "310100",
            "lot_code": "SCN-LOT-A",
            "entry_time": dt("2026-02-01T09:00:00+08:00"),
            "exit_time": dt("2026-02-01T10:00:00+08:00"),
            "paid_amount": Decimal("4.00"),
        },
        {
            "order_no": "SCN-009-B",
            "plate_no": "沪SCN009",
            "city_code": "310100",
            "lot_code": "SCN-LOT-A",
            "entry_time": dt("2026-02-02T09:00:00+08:00"),
            "exit_time": dt("2026-02-02T10:00:00+08:00"),
            "paid_amount": Decimal("2.00"),
        },
        {
            "order_no": "SCN-009-C",
            "plate_no": "沪SCN009",
            "city_code": "320100",
            "lot_code": "SCN-LOT-A",
            "entry_time": dt("2026-02-03T09:00:00+08:00"),
            "exit_time": dt("2026-02-03T10:00:00+08:00"),
            "paid_amount": Decimal("0.00"),
        },
    ]


async def _cleanup_old_data(session: AsyncSession) -> None:
    await session.execute(delete(ParkingOrder).where(ParkingOrder.order_no.like("SCN-%")))

    rule_ids = list(
        (
            await session.execute(
                select(BillingRule.id).where(BillingRule.rule_code.like("SCN-RULE-%"))
            )
        ).scalars()
    )
    if rule_ids:
        await session.execute(delete(BillingRuleVersion).where(BillingRuleVersion.rule_id.in_(rule_ids)))
    await session.execute(delete(BillingRule).where(BillingRule.rule_code.like("SCN-RULE-%")))


async def _insert_rules(session: AsyncSession, rules: list[RuleSeed]) -> dict[str, BillingRuleVersion]:
    version_map: dict[str, BillingRuleVersion] = {}

    for rule_seed in rules:
        rule = BillingRule(
            rule_code=rule_seed.rule_code,
            name=rule_seed.name,
            status="enabled",
            scope_type="lot_code",
            scope={
                "scope_type": "lot_code",
                "city_code": rule_seed.city_code,
                "lot_codes": rule_seed.lot_codes,
                "lot_type": None,
            },
        )
        session.add(rule)
        await session.flush()

        for version in rule_seed.versions:
            version_row = BillingRuleVersion(
                rule_id=rule.id,
                version_no=version["version_no"],
                effective_from=version["effective_from"],
                effective_to=version["effective_to"],
                priority=version["priority"],
                rule_payload=version["rule_payload"],
            )
            session.add(version_row)
            await session.flush()
            version_map[f"{rule_seed.rule_code}:v{version_row.version_no}"] = version_row

    return version_map


def _find_active_version(rule: RuleSeed, at_time: datetime) -> dict:
    for version in rule.versions:
        if version["effective_from"] <= at_time and (
            version["effective_to"] is None or at_time < version["effective_to"]
        ):
            return version
    raise ValueError(f"No active version for rule={rule.rule_code} at {at_time}")


async def _insert_scenario_orders(
    session: AsyncSession,
    rules: list[RuleSeed],
    version_map: dict[str, BillingRuleVersion],
) -> list[dict]:
    rule_index = {item.rule_code: item for item in rules}
    dataset_rows: list[dict] = []

    for scenario in _scenario_cases():
        rule_seed = rule_index[scenario.rule_code]
        version = _find_active_version(rule_seed, scenario.entry_time)
        version_key = f"{scenario.rule_code}:v{version['version_no']}"
        version_row = version_map[version_key]

        sim = simulate_fee(version["rule_payload"], scenario.entry_time, scenario.exit_time)
        expected_total = q(sim["total_amount"])
        order_total = scenario.force_order_total if scenario.force_order_total is not None else expected_total
        order_total = q(order_total)
        amount_consistent = order_total == expected_total
        arrears = q(max(Decimal("0.00"), order_total - scenario.paid_amount))
        arrears_status = "HAS_ARREARS" if arrears > 0 else "NONE"

        if scenario.order_no:
            session.add(
                ParkingOrder(
                    order_no=scenario.order_no,
                    plate_no=scenario.plate_no,
                    city_code=scenario.city_code,
                    lot_code=scenario.lot_code,
                    billing_rule_code=scenario.rule_code,
                    billing_rule_version_no=version_row.version_no,
                    entry_time=scenario.entry_time,
                    exit_time=scenario.exit_time,
                    total_amount=order_total,
                    paid_amount=scenario.paid_amount,
                    arrears_amount=arrears,
                    status="UNPAID" if arrears > 0 else "PAID",
                )
            )

        dataset_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "intent_tags": scenario.intent_tags,
                "query": scenario.query,
                "context": {
                    "city_code": scenario.city_code,
                    "lot_code": scenario.lot_code,
                    "plate_no": scenario.plate_no,
                    "order_no": scenario.order_no,
                    "entry_time": scenario.entry_time.isoformat(),
                    "exit_time": scenario.exit_time.isoformat(),
                },
                "expected_tools": [
                    "POST /api/v1/billing-rules/simulate",
                    "GET /api/v1/parking-orders/{order_no}",
                ]
                if "fee_verify" in scenario.intent_tags
                else ["GET /api/v1/arrears-orders"],
                "ground_truth": {
                    "matched_rule_code": scenario.rule_code,
                    "matched_version_no": version_row.version_no,
                    "expected_total_amount": str(expected_total),
                    "order_total_amount": str(order_total),
                    "amount_check_result": "一致" if amount_consistent else "不一致",
                    "amount_check_action": "自动通过" if amount_consistent else "需人工复核",
                    "expected_paid_amount": str(q(scenario.paid_amount)),
                    "expected_arrears_amount": str(arrears),
                    "expected_arrears_status": arrears_status,
                },
                "expected_citations": {
                    "doc_type": ["rule_explain", "policy_doc"],
                    "source_ids": [scenario.rule_code],
                },
                "notes": scenario.notes,
            }
        )

    # SCN-009: same plate with mixed payment status for arrears-check intent
    scn009_truth: list[dict] = []
    rule_seed = rule_index["SCN-RULE-PERIODIC"]
    for item in _scenario_009_orders():
        version = _find_active_version(rule_seed, item["entry_time"])
        version_row = version_map[f"SCN-RULE-PERIODIC:v{version['version_no']}"]
        sim = simulate_fee(version["rule_payload"], item["entry_time"], item["exit_time"])
        total = q(sim["total_amount"])
        arrears = q(max(Decimal("0.00"), total - item["paid_amount"]))

        session.add(
            ParkingOrder(
                order_no=item["order_no"],
                plate_no=item["plate_no"],
                city_code=item["city_code"],
                lot_code=item["lot_code"],
                billing_rule_code="SCN-RULE-PERIODIC",
                billing_rule_version_no=version_row.version_no,
                entry_time=item["entry_time"],
                exit_time=item["exit_time"],
                total_amount=total,
                paid_amount=item["paid_amount"],
                arrears_amount=arrears,
                status="UNPAID" if arrears > 0 else "PAID",
            )
        )
        scn009_truth.append(
            {
                "order_no": item["order_no"],
                "city_code": item["city_code"],
                "total_amount": str(total),
                "paid_amount": str(q(item["paid_amount"])),
                "arrears_amount": str(arrears),
                "arrears_status": "HAS_ARREARS" if arrears > 0 else "NONE",
            }
        )

    dataset_rows.append(
        {
            "scenario_id": "SCN-009",
            "intent_tags": ["arrears_check"],
            "query": "我这个车牌现在是不是有欠费？",
            "context": {
                "city_code": "310100",
                "lot_code": "SCN-LOT-A",
                "plate_no": "沪SCN009",
                "order_no": None,
                "entry_time": None,
                "exit_time": None,
            },
            "expected_tools": ["GET /api/v1/arrears-orders"],
            "ground_truth": {
                "matched_rule_code": "SCN-RULE-PERIODIC",
                "matched_version_no": 1,
                "expected_arrears_orders": [
                    item["order_no"]
                    for item in scn009_truth
                    if item["arrears_status"] == "HAS_ARREARS" and item["city_code"] == "310100"
                ],
                "orders": scn009_truth,
            },
            "expected_citations": {
                "doc_type": ["faq", "sop"],
                "source_ids": ["SCN-FAQ-ARREARS", "SCN-SOP-ARREARS"],
            },
            "notes": "同车牌多订单欠费混合场景",
        }
    )

    return dataset_rows


def _write_jsonl(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def seed_dataset(database_url: str, export_jsonl: Path | None, ensure_schema: bool) -> None:
    engine = create_async_engine(database_url, echo=False, future=True)
    session_maker = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

    try:
        async with session_maker() as session:
            if ensure_schema:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)

            await _cleanup_old_data(session)
            rules = _scenario_rules()
            version_map = await _insert_rules(session, rules)
            dataset_rows = await _insert_scenario_orders(session, rules, version_map)
            await session.commit()

        if export_jsonl is not None:
            _write_jsonl(dataset_rows, export_jsonl)
            print(f"[ok] exported {len(dataset_rows)} scenario rows -> {export_jsonl}")
        print(f"[ok] seeded dataset into {database_url}")
    finally:
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed RAG-000 biz scenario dataset")
    parser.add_argument(
        "--database-url",
        default=(
            os.getenv("BIZ_SCENARIO_DATABASE_URL")
            or os.getenv("BIZ_TEST_DATABASE_URL")
            or "postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_test"
        ),
        help="target biz database URL",
    )
    parser.add_argument(
        "--export-jsonl",
        default="data/rag000/scenarios.jsonl",
        help="output JSONL path for scenario dataset",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="do not export JSONL file",
    )
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help="run Base.metadata.create_all before seeding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_path = None if args.no_export else Path(args.export_jsonl)
    asyncio.run(seed_dataset(args.database_url, export_path, ensure_schema=args.ensure_schema))


if __name__ == "__main__":
    main()
