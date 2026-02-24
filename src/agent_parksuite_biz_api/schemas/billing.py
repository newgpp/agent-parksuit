from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class RuleScope(BaseModel):
    """计费规则作用范围。"""

    scope_type: Literal["lot_code"] = Field(default="lot_code", description="作用域类型，当前固定为 lot_code")
    city_code: str = Field(description="城市编码")
    lot_codes: list[str] = Field(min_length=1, description="停车场编码列表，至少一个")
    lot_type: str | None = Field(default=None, description="停车场类型（可选）")

    @field_validator("lot_codes")
    @classmethod
    def validate_lot_codes(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item and item.strip()]
        deduped = list(dict.fromkeys(normalized))
        if not deduped:
            raise ValueError("lot_codes must contain at least one non-empty value")
        return deduped


class BillingRuleVersionCreate(BaseModel):
    """计费规则版本创建请求。"""

    effective_from: datetime = Field(description="版本生效开始时间")
    effective_to: datetime | None = Field(default=None, description="版本生效结束时间，空表示长期生效")
    priority: int = Field(default=100, description="版本优先级，值越大优先级越高")
    rule_payload: list["RuleSegment"] = Field(default_factory=list, description="计费规则内容（JSON 数组）")


class BillingRuleUpsertRequest(BaseModel):
    """新增或更新计费规则请求。"""

    rule_code: str = Field(description="计费规则编码")
    name: str = Field(description="计费规则名称")
    status: Literal["enabled", "disabled"] = Field(default="enabled", description="规则状态")
    scope: RuleScope = Field(description="规则作用范围")
    version: BillingRuleVersionCreate = Field(description="规则版本信息")


class BillingRuleVersionResponse(BaseModel):
    """计费规则版本响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="版本主键ID")
    version_no: int = Field(description="版本号")
    effective_from: datetime = Field(description="生效开始时间")
    effective_to: datetime | None = Field(description="生效结束时间")
    priority: int = Field(description="版本优先级")
    rule_payload: list["RuleSegment"] = Field(description="计费规则内容（JSON 数组）")


class BillingRuleResponse(BaseModel):
    """计费规则响应。"""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="规则主键ID")
    rule_code: str = Field(description="计费规则编码")
    name: str = Field(description="计费规则名称")
    status: str = Field(description="规则状态")
    scope_type: str = Field(description="作用域类型")
    scope: dict = Field(description="作用域配置")
    versions: list[BillingRuleVersionResponse] = Field(description="规则版本列表")


class BillingSimulateRequest(BaseModel):
    """计费模拟请求。"""

    rule_code: str = Field(description="计费规则编码")
    entry_time: datetime = Field(description="入场时间")
    exit_time: datetime = Field(description="离场时间")


class SegmentCharge(BaseModel):
    """分段计费明细。"""

    segment_name: str = Field(description="分段名称")
    segment_type: str = Field(description="分段类型（periodic/tiered/free）")
    minutes: int = Field(description="命中分段分钟数")
    amount: Decimal = Field(description="分段金额")
    free_minutes: int = Field(default=0, description="分段免费分钟数")
    capped: bool = Field(default=False, description="是否触发封顶")


class BillingSimulateResponse(BaseModel):
    """计费模拟响应。"""

    duration_minutes: int = Field(description="停车总时长（分钟）")
    total_amount: Decimal = Field(description="总金额")
    matched_version_no: int = Field(description="命中的规则版本号")
    breakdown: list[SegmentCharge] = Field(description="分段计费明细")


class TimeWindow(BaseModel):
    start: str = Field(description="时间窗口开始，格式 HH:MM")
    end: str = Field(description="时间窗口结束，格式 HH:MM")
    timezone: str = Field(default="Asia/Shanghai", description="时间窗口时区，默认 Asia/Shanghai")

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value


class TierPriceRule(BaseModel):
    start_minute: int = Field(ge=0, description="阶梯起始分钟（包含）")
    end_minute: int | None = Field(default=None, ge=1, description="阶梯结束分钟（不包含），空表示无上限")
    unit_price: Decimal = Field(ge=0, description="该阶梯每计费单元价格")


class BaseRuleSegment(BaseModel):
    name: str = Field(description="分段名称")
    type: str = Field(description="分段类型")
    time_window: TimeWindow | None = Field(default=None, description="命中时间窗口")
    weekdays: list[int] | None = Field(default=None, description="命中星期，ISO weekday 1-7")

    @field_validator("weekdays")
    @classmethod
    def validate_weekdays(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if any(day < 1 or day > 7 for day in value):
            raise ValueError("weekdays must be ISO weekdays in [1, 7]")
        return value


class FreeRuleSegment(BaseRuleSegment):
    type: Literal["free"] = "free"


class PeriodicRuleSegment(BaseRuleSegment):
    type: Literal["periodic"] = "periodic"
    unit_minutes: int = Field(gt=0, description="计费单元分钟")
    unit_price: Decimal = Field(ge=0, description="每计费单元价格")
    free_minutes: int = Field(default=0, ge=0, description="免费分钟数")
    max_charge: Decimal | None = Field(default=None, ge=0, description="封顶金额")


class TieredRuleSegment(BaseRuleSegment):
    type: Literal["tiered"] = "tiered"
    unit_minutes: int = Field(gt=0, description="计费单元分钟")
    tiers: list[TierPriceRule] = Field(min_length=1, description="阶梯价格配置")
    free_minutes: int = Field(default=0, ge=0, description="免费分钟数")
    max_charge: Decimal | None = Field(default=None, ge=0, description="封顶金额")


RuleSegment = Annotated[FreeRuleSegment | PeriodicRuleSegment | TieredRuleSegment, Field(discriminator="type")]
RuleSegmentListAdapter = TypeAdapter(list[RuleSegment])

BillingRuleVersionCreate.model_rebuild()
BillingRuleVersionResponse.model_rebuild()
