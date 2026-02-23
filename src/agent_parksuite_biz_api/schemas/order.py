from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ParkingOrderResponse(BaseModel):
    """停车订单响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="订单主键ID")
    order_no: str = Field(description="订单号")
    plate_no: str = Field(description="车牌号")
    city_code: str = Field(description="城市编码")
    lot_code: str = Field(description="停车场编码")
    billing_rule_code: str = Field(description="计费规则编码")
    billing_rule_version_no: int | None = Field(description="计费规则版本号")
    entry_time: datetime = Field(description="入场时间")
    exit_time: datetime | None = Field(description="离场时间")
    total_amount: Decimal = Field(description="应付总金额")
    paid_amount: Decimal = Field(description="已支付金额")
    arrears_amount: Decimal = Field(description="欠费金额")
    status: str = Field(description="订单状态")


class ParkingOrderCreateRequest(BaseModel):
    """停车订单创建请求。"""

    order_no: str = Field(description="订单号")
    plate_no: str = Field(description="车牌号")
    city_code: str = Field(description="城市编码")
    lot_code: str = Field(description="停车场编码")
    billing_rule_code: str = Field(description="计费规则编码")
    billing_rule_version_no: int | None = Field(default=None, description="计费规则版本号")
    entry_time: datetime = Field(description="入场时间")
    exit_time: datetime | None = Field(default=None, description="离场时间")
    total_amount: Decimal = Field(default=Decimal("0.00"), ge=0, description="应付总金额")
    paid_amount: Decimal = Field(default=Decimal("0.00"), ge=0, description="已支付金额")
    status: str = Field(default="UNPAID", description="订单状态")
