from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    rule_payload: list[dict] = Field(default_factory=list, description="计费规则内容（JSON 数组）")


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
    rule_payload: list[dict] = Field(description="计费规则内容（JSON 数组）")


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
