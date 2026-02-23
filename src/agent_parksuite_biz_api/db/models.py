from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agent_parksuite_biz_api.db.base import Base


def _utcnow_utc() -> datetime:
    return datetime.now(UTC)


class BillingRule(Base):
    __tablename__ = "billing_rules"
    __table_args__ = (
        Index("ix_billing_rules_scope_gin", "scope", postgresql_using="gin"),
        {"comment": "计费规则主表"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键ID")
    rule_code: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="规则编码")
    name: Mapped[str] = mapped_column(String(128), comment="规则名称")
    status: Mapped[str] = mapped_column(String(32), default="enabled", index=True, comment="规则状态")
    scope_type: Mapped[str] = mapped_column(String(20), default="lot_code", index=True, comment="作用域类型")
    scope: Mapped[dict] = mapped_column(JSONB, default=dict, comment="作用域配置JSON")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow_utc, comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow_utc, onupdate=_utcnow_utc, comment="更新时间"
    )
    versions: Mapped[list["BillingRuleVersion"]] = relationship(
        back_populates="rule", cascade="all, delete-orphan", order_by="BillingRuleVersion.version_no"
    )


class BillingRuleVersion(Base):
    __tablename__ = "billing_rule_versions"
    __table_args__ = (
        UniqueConstraint("rule_id", "version_no", name="uq_billing_rule_versions_rule_id_version_no"),
        {"comment": "计费规则版本表"},
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键ID")
    rule_id: Mapped[int] = mapped_column(
        ForeignKey("billing_rules.id", ondelete="CASCADE"), index=True, comment="所属规则ID"
    )
    version_no: Mapped[int] = mapped_column(Integer, comment="版本号")
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, comment="生效开始时间")
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="生效结束时间")
    priority: Mapped[int] = mapped_column(Integer, default=100, comment="优先级")
    rule_payload: Mapped[list[dict]] = mapped_column(JSONB, default=list, comment="计费规则内容JSON数组")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow_utc, comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow_utc, onupdate=_utcnow_utc, comment="更新时间"
    )
    rule: Mapped["BillingRule"] = relationship(back_populates="versions")


class ParkingOrder(Base):
    __tablename__ = "parking_orders"
    __table_args__ = ({"comment": "停车订单表"},)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键ID")
    order_no: Mapped[str] = mapped_column(String(64), unique=True, index=True, comment="订单号")
    plate_no: Mapped[str] = mapped_column(String(16), index=True, comment="车牌号")
    city_code: Mapped[str] = mapped_column(String(32), index=True, comment="城市编码")
    lot_code: Mapped[str] = mapped_column(String(64), index=True, comment="停车场编码")
    billing_rule_code: Mapped[str] = mapped_column(String(64), index=True, comment="计费规则编码")
    billing_rule_version_no: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="计费规则版本号")

    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, comment="入场时间")
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, comment="离场时间")

    total_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("0.00"), comment="应付总金额")
    paid_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("0.00"), comment="已支付金额")
    arrears_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), default=Decimal("0.00"), comment="欠费金额")
    status: Mapped[str] = mapped_column(String(32), index=True, default="UNPAID", comment="订单状态")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow_utc, comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow_utc, onupdate=_utcnow_utc, comment="更新时间"
    )
