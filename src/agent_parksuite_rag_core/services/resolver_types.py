from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent_parksuite_rag_core.schemas.answer import HybridAnswerRequest

FieldSource = Literal["user", "memory", "inferred"]


@dataclass(frozen=True)
class IntentSlotParseResult:
    """阶段1产物：意图与槽位初步解析结果。"""

    # 解析后（含可能从query抽取槽位）的请求对象
    payload: HybridAnswerRequest
    # 识别到的意图；None表示当前无法确定
    intent: str | None
    # 意图置信度（可选）
    intent_confidence: float | None
    # 槽位来源标记：user/memory/inferred
    field_sources: dict[str, FieldSource]
    # 基于当前意图判断的必填缺失槽位
    missing_required_slots: list[str]
    # 解析出的歧义列表（如订单指代歧义）
    ambiguities: list[str]
    # 本阶段轨迹
    trace: list[str]


@dataclass(frozen=True)
class SlotHydrateResult:
    """阶段2产物：结合会话记忆补槽后的结果。"""

    # 补槽后的请求对象
    payload: HybridAnswerRequest
    # 补槽后字段来源标记
    field_sources: dict[str, FieldSource]
    # 补槽后仍缺失的必填槽位
    missing_required_slots: list[str]
    # 本阶段轨迹
    trace: list[str]
