from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalQuery:
    eval_id: str
    intent: str
    query: str
    context: dict[str, Any] = field(default_factory=dict)
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
