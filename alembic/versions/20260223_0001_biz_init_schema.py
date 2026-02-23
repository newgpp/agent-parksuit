"""init biz schema

Revision ID: 20260223_0001_biz
Revises:
Create Date: 2026-02-23 20:30:00

"""
from __future__ import annotations

from urllib.parse import urlparse

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260223_0001_biz"
down_revision = None
branch_labels = None
depends_on = None


TARGET_DB = "parksuite_biz"


def _current_db_name() -> str:
    url = context.config.get_main_option("sqlalchemy.url")
    return urlparse(url).path.lstrip("/").split("?", 1)[0]


def _is_target_db() -> bool:
    return _current_db_name() == TARGET_DB


def upgrade() -> None:
    if not _is_target_db():
        return

    op.create_table(
        "billing_rules",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("rule_code", sa.String(length=64), nullable=False, comment="规则编码"),
        sa.Column("name", sa.String(length=128), nullable=False, comment="规则名称"),
        sa.Column("status", sa.String(length=32), nullable=False, comment="规则状态"),
        sa.Column("scope_type", sa.String(length=20), nullable=False, comment="作用域类型"),
        sa.Column("scope", postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment="作用域配置JSON"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        comment="计费规则主表",
    )
    op.create_index(op.f("ix_billing_rules_rule_code"), "billing_rules", ["rule_code"], unique=True)
    op.create_index(op.f("ix_billing_rules_scope_type"), "billing_rules", ["scope_type"], unique=False)
    op.create_index(op.f("ix_billing_rules_status"), "billing_rules", ["status"], unique=False)
    op.create_index("ix_billing_rules_scope_gin", "billing_rules", ["scope"], unique=False, postgresql_using="gin")

    op.create_table(
        "billing_rule_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("rule_id", sa.Integer(), nullable=False, comment="所属规则ID"),
        sa.Column("version_no", sa.Integer(), nullable=False, comment="版本号"),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False, comment="生效开始时间"),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True, comment="生效结束时间"),
        sa.Column("priority", sa.Integer(), nullable=False, comment="优先级"),
        sa.Column("rule_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, comment="计费规则内容JSON数组"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, comment="更新时间"),
        sa.ForeignKeyConstraint(["rule_id"], ["billing_rules.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rule_id", "version_no", name="uq_billing_rule_versions_rule_id_version_no"),
        comment="计费规则版本表",
    )
    op.create_index(op.f("ix_billing_rule_versions_rule_id"), "billing_rule_versions", ["rule_id"], unique=False)
    op.create_index(
        op.f("ix_billing_rule_versions_effective_from"),
        "billing_rule_versions",
        ["effective_from"],
        unique=False,
    )

    op.create_table(
        "parking_orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False, comment="主键ID"),
        sa.Column("order_no", sa.String(length=64), nullable=False, comment="订单号"),
        sa.Column("plate_no", sa.String(length=16), nullable=False, comment="车牌号"),
        sa.Column("city_code", sa.String(length=32), nullable=False, comment="城市编码"),
        sa.Column("lot_code", sa.String(length=64), nullable=False, comment="停车场编码"),
        sa.Column("billing_rule_code", sa.String(length=64), nullable=False, comment="计费规则编码"),
        sa.Column("billing_rule_version_no", sa.Integer(), nullable=True, comment="计费规则版本号"),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False, comment="入场时间"),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True, comment="离场时间"),
        sa.Column("total_amount", sa.Numeric(precision=10, scale=2), nullable=False, comment="应付总金额"),
        sa.Column("paid_amount", sa.Numeric(precision=10, scale=2), nullable=False, comment="已支付金额"),
        sa.Column("arrears_amount", sa.Numeric(precision=10, scale=2), nullable=False, comment="欠费金额"),
        sa.Column("status", sa.String(length=32), nullable=False, comment="订单状态"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, comment="创建时间"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, comment="更新时间"),
        sa.PrimaryKeyConstraint("id"),
        comment="停车订单表",
    )
    op.create_index(op.f("ix_parking_orders_order_no"), "parking_orders", ["order_no"], unique=True)
    op.create_index(op.f("ix_parking_orders_plate_no"), "parking_orders", ["plate_no"], unique=False)
    op.create_index(op.f("ix_parking_orders_city_code"), "parking_orders", ["city_code"], unique=False)
    op.create_index(op.f("ix_parking_orders_lot_code"), "parking_orders", ["lot_code"], unique=False)
    op.create_index(
        op.f("ix_parking_orders_billing_rule_code"),
        "parking_orders",
        ["billing_rule_code"],
        unique=False,
    )
    op.create_index(op.f("ix_parking_orders_entry_time"), "parking_orders", ["entry_time"], unique=False)
    op.create_index(op.f("ix_parking_orders_status"), "parking_orders", ["status"], unique=False)


def downgrade() -> None:
    if not _is_target_db():
        return

    op.drop_index(op.f("ix_parking_orders_status"), table_name="parking_orders")
    op.drop_index(op.f("ix_parking_orders_entry_time"), table_name="parking_orders")
    op.drop_index(op.f("ix_parking_orders_billing_rule_code"), table_name="parking_orders")
    op.drop_index(op.f("ix_parking_orders_lot_code"), table_name="parking_orders")
    op.drop_index(op.f("ix_parking_orders_city_code"), table_name="parking_orders")
    op.drop_index(op.f("ix_parking_orders_plate_no"), table_name="parking_orders")
    op.drop_index(op.f("ix_parking_orders_order_no"), table_name="parking_orders")
    op.drop_table("parking_orders")

    op.drop_index(op.f("ix_billing_rule_versions_effective_from"), table_name="billing_rule_versions")
    op.drop_index(op.f("ix_billing_rule_versions_rule_id"), table_name="billing_rule_versions")
    op.drop_table("billing_rule_versions")

    op.drop_index("ix_billing_rules_scope_gin", table_name="billing_rules")
    op.drop_index(op.f("ix_billing_rules_status"), table_name="billing_rules")
    op.drop_index(op.f("ix_billing_rules_scope_type"), table_name="billing_rules")
    op.drop_index(op.f("ix_billing_rules_rule_code"), table_name="billing_rules")
    op.drop_table("billing_rules")
