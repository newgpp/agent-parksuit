from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalQuery:
    eval_id: str
    group: str
    hybrid_request: dict[str, Any] = field(default_factory=dict)
    expected_retrieval: dict[str, Any] = field(default_factory=dict)
    expected_tools: list[str] = field(default_factory=list)
    expected_answer: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalSummary:
    total: int
    retrieval_hit_rate: float
    citation_coverage: float
    empty_retrieval_rate: float
    tool_call_compliance_rate: float
    answer_consistency_rate: float


@dataclass(slots=True)
class EvalSampleResult:
    eval_id: str
    group: str
    intent: str
    retrieval_ok: bool
    citation_ok: bool
    tool_ok: bool
    answer_ok: bool
    retrieval_count: int
    citation_count: int
    expected_tools: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
