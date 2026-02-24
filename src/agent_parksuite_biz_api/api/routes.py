from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import Select, and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_parksuite_biz_api.config import settings
from agent_parksuite_biz_api.db.models import BillingRule, BillingRuleVersion, ParkingOrder
from agent_parksuite_biz_api.db.session import get_db_session
from agent_parksuite_biz_api.schemas.billing import (
    BillingRuleResponse,
    BillingRuleUpsertRequest,
    BillingSimulateRequest,
    BillingSimulateResponse,
)
from agent_parksuite_biz_api.schemas.order import ParkingOrderCreateRequest, ParkingOrderResponse
from agent_parksuite_biz_api.services.billing_engine import simulate_fee

router = APIRouter(prefix="/api/v1", tags=["biz-api"])


def _has_time_overlap(
    existing_from: datetime,
    existing_to: datetime | None,
    target_from: datetime,
    target_to: datetime | None,
) -> bool:
    if existing_to is not None and existing_to <= target_from:
        return False
    if target_to is not None and target_to <= existing_from:
        return False
    return True


def _pick_version(versions: list[BillingRuleVersion], at_time: datetime) -> BillingRuleVersion | None:
    matched = [
        item
        for item in versions
        if item.effective_from <= at_time and (item.effective_to is None or at_time < item.effective_to)
    ]
    if not matched:
        return None
    matched.sort(key=lambda item: (item.priority, item.version_no), reverse=True)
    return matched[0]


@router.post(
    "/billing-rules",
    response_model=BillingRuleResponse,
    summary="新增或更新计费规则",
    description="按规则编码新增/更新规则主信息，并追加一个新的规则版本。",
)
async def upsert_billing_rule(
    payload: BillingRuleUpsertRequest,
    db: AsyncSession = Depends(get_db_session),
) -> BillingRule:
    """计费规则配置接口。"""
    logger.info("upsert_billing_rule.request payload={}", payload.model_dump(mode="json"))
    stmt = select(BillingRule).where(BillingRule.rule_code == payload.rule_code)
    existing_rule = (await db.execute(stmt)).scalar_one_or_none()

    if existing_rule:
        existing_rule.name = payload.name
        existing_rule.status = payload.status
        existing_rule.scope_type = payload.scope.scope_type
        existing_rule.scope = payload.scope.model_dump()
        rule = existing_rule
    else:
        rule = BillingRule(
            rule_code=payload.rule_code,
            name=payload.name,
            status=payload.status,
            scope_type=payload.scope.scope_type,
            scope=payload.scope.model_dump(),
        )
        db.add(rule)
        await db.flush()

    versions_stmt = select(BillingRuleVersion).where(BillingRuleVersion.rule_id == rule.id)
    existing_versions = list((await db.execute(versions_stmt)).scalars().all())

    for item in existing_versions:
        if _has_time_overlap(
            item.effective_from,
            item.effective_to,
            payload.version.effective_from,
            payload.version.effective_to,
        ):
            logger.warning(
                "upsert_billing_rule.conflict rule_code={} conflict_version_no={}",
                payload.rule_code,
                item.version_no,
            )
            raise HTTPException(
                status_code=409,
                detail=(
                    "Version time range overlaps with an existing version: "
                    f"v{item.version_no} ({item.effective_from} - {item.effective_to})"
                ),
            )

    next_version_no = (max((item.version_no for item in existing_versions), default=0)) + 1
    new_version = BillingRuleVersion(
        rule_id=rule.id,
        version_no=next_version_no,
        effective_from=payload.version.effective_from,
        effective_to=payload.version.effective_to,
        priority=payload.version.priority,
        rule_payload=payload.version.rule_payload,
    )
    db.add(new_version)

    await db.commit()

    refreshed = (
        await db.execute(
            select(BillingRule).where(BillingRule.id == rule.id)
        )
    ).scalar_one()
    await db.refresh(refreshed, attribute_names=["versions"])
    logger.info(
        "upsert_billing_rule.response rule_code={} version_count={} latest_version_no={}",
        refreshed.rule_code,
        len(refreshed.versions),
        max((item.version_no for item in refreshed.versions), default=None),
    )
    return refreshed


@router.get(
    "/billing-rules",
    response_model=list[BillingRuleResponse],
    summary="查询计费规则列表",
    description="支持按城市编码和停车场编码筛选计费规则。",
)
async def list_billing_rules(
    city_code: str | None = Query(default=None),
    lot_code: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db_session),
) -> list[BillingRule]:
    """计费规则列表查询接口。"""
    logger.info("list_billing_rules.request city_code={} lot_code={}", city_code, lot_code)
    stmt: Select[tuple[BillingRule]] = select(BillingRule)

    filters = []
    if city_code:
        filters.append(BillingRule.scope["city_code"].astext == city_code)
    if lot_code:
        filters.append(BillingRule.scope_type == "lot_code")
        filters.append(BillingRule.scope.contains({"lot_codes": [lot_code]}))

    if filters:
        stmt = stmt.where(and_(*filters))

    rules = list((await db.execute(stmt.order_by(BillingRule.id.desc()))).scalars().all())
    for item in rules:
        await db.refresh(item, attribute_names=["versions"])
    logger.info("list_billing_rules.response count={}", len(rules))
    return rules


@router.get(
    "/billing-rules/{rule_code}",
    response_model=BillingRuleResponse,
    summary="查询计费规则详情",
    description="根据规则编码查询规则详情及其全部版本。",
)
async def get_billing_rule(
    rule_code: str,
    db: AsyncSession = Depends(get_db_session),
) -> BillingRule:
    """计费规则详情查询接口。"""
    logger.info("get_billing_rule.request rule_code={}", rule_code)
    rule = (await db.execute(select(BillingRule).where(BillingRule.rule_code == rule_code))).scalar_one_or_none()
    if not rule:
        logger.warning("get_billing_rule.not_found rule_code={}", rule_code)
        raise HTTPException(status_code=404, detail="Billing rule not found")

    await db.refresh(rule, attribute_names=["versions"])
    logger.info("get_billing_rule.response rule_code={} version_count={}", rule.rule_code, len(rule.versions))
    return rule


@router.post(
    "/billing-rules/simulate",
    response_model=BillingSimulateResponse,
    summary="模拟计费",
    description="按入场时间匹配规则版本并执行计费模拟，返回费用明细。",
)
async def simulate_billing(
    payload: BillingSimulateRequest,
    db: AsyncSession = Depends(get_db_session),
) -> BillingSimulateResponse:
    """计费模拟接口。"""
    logger.info("simulate_billing.request payload={}", payload.model_dump(mode="json"))
    rule = (await db.execute(select(BillingRule).where(BillingRule.rule_code == payload.rule_code))).scalar_one_or_none()
    if not rule:
        logger.warning("simulate_billing.rule_not_found rule_code={}", payload.rule_code)
        raise HTTPException(status_code=404, detail="Billing rule not found")

    await db.refresh(rule, attribute_names=["versions"])
    matched_version = _pick_version(list(rule.versions), payload.entry_time)
    if not matched_version:
        logger.warning(
            "simulate_billing.no_active_version rule_code={} entry_time={}",
            payload.rule_code,
            payload.entry_time.isoformat(),
        )
        raise HTTPException(status_code=404, detail="No active version for entry_time")

    result = simulate_fee(
        matched_version.rule_payload,
        payload.entry_time,
        payload.exit_time,
        business_timezone=settings.business_timezone,
    )
    response = BillingSimulateResponse(**result, matched_version_no=matched_version.version_no)
    logger.info(
        "simulate_billing.response rule_code={} matched_version_no={} total_amount={} segment_count={}",
        payload.rule_code,
        response.matched_version_no,
        str(response.total_amount),
        len(response.breakdown),
    )
    return response


@router.post(
    "/parking-orders",
    response_model=ParkingOrderResponse,
    summary="创建停车订单",
    description="创建停车订单并自动计算欠费金额（应付-已付）。",
)
async def create_parking_order(
    payload: ParkingOrderCreateRequest,
    db: AsyncSession = Depends(get_db_session),
) -> ParkingOrder:
    """停车订单创建接口。"""
    logger.info("create_parking_order.request payload={}", payload.model_dump(mode="json"))
    arrears_amount = max(Decimal("0.00"), payload.total_amount - payload.paid_amount)
    order = ParkingOrder(**payload.model_dump(), arrears_amount=arrears_amount)
    db.add(order)
    await db.commit()
    await db.refresh(order)
    logger.info(
        "create_parking_order.response order_no={} total_amount={} paid_amount={} arrears_amount={} status={}",
        order.order_no,
        str(order.total_amount),
        str(order.paid_amount),
        str(order.arrears_amount),
        order.status,
    )
    return order


@router.get(
    "/parking-orders/{order_no}",
    response_model=ParkingOrderResponse,
    summary="查询停车订单详情",
    description="根据订单号查询停车订单详情。",
)
async def get_parking_order(
    order_no: str,
    db: AsyncSession = Depends(get_db_session),
) -> ParkingOrder:
    """停车订单详情查询接口。"""
    logger.info("get_parking_order.request order_no={}", order_no)
    order = (await db.execute(select(ParkingOrder).where(ParkingOrder.order_no == order_no))).scalar_one_or_none()
    if not order:
        logger.warning("get_parking_order.not_found order_no={}", order_no)
        raise HTTPException(status_code=404, detail="Order not found")
    logger.info(
        "get_parking_order.response order_no={} total_amount={} paid_amount={} arrears_amount={} status={}",
        order.order_no,
        str(order.total_amount),
        str(order.paid_amount),
        str(order.arrears_amount),
        order.status,
    )
    return order


@router.get(
    "/arrears-orders",
    response_model=list[ParkingOrderResponse],
    summary="查询欠费订单列表",
    description="查询欠费金额大于0的订单，支持按车牌号和城市编码筛选。",
)
async def list_arrears_orders(
    plate_no: str | None = Query(default=None),
    city_code: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db_session),
) -> list[ParkingOrder]:
    """欠费订单列表查询接口。"""
    logger.info("list_arrears_orders.request plate_no={} city_code={}", plate_no, city_code)
    stmt: Select[tuple[ParkingOrder]] = select(ParkingOrder).where(ParkingOrder.arrears_amount > 0)

    if plate_no:
        stmt = stmt.where(ParkingOrder.plate_no == plate_no)
    if city_code:
        stmt = stmt.where(ParkingOrder.city_code == city_code)

    rows = list((await db.execute(stmt.order_by(ParkingOrder.id.desc()))).scalars().all())
    logger.info("list_arrears_orders.response count={}", len(rows))
    return rows
