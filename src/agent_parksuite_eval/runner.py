from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from agent_parksuite_eval.schemas import EvalQuery, EvalSampleResult, EvalSummary


def _load_eval_queries(dataset_path: Path) -> list[EvalQuery]:
    rows: list[EvalQuery] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        rows.append(
            EvalQuery(
                eval_id=str(raw.get("eval_id", "")),
                group=str(raw.get("group", "")),
                hybrid_request=dict(raw.get("hybrid_request", {})),
                expected_retrieval=dict(raw.get("expected_retrieval", {})),
                expected_tools=list(raw.get("expected_tools", [])),
                expected_answer=dict(raw.get("expected_answer", {})),
            )
        )
    return rows


def _resolve_intent(item: EvalQuery) -> str:
    return str(item.hybrid_request.get("intent_hint", "")).strip()


def _build_retrieve_payload(item: EvalQuery) -> dict[str, Any]:
    hybrid = item.hybrid_request
    payload: dict[str, Any] = {
        "query": str(hybrid.get("query", "")),
        "top_k": int(hybrid.get("top_k", 5)),
        "doc_type": hybrid.get("doc_type", "rule_explain"),
        "source_type": hybrid.get("source_type", "biz_derived"),
        "include_inactive": bool(hybrid.get("include_inactive", False)),
    }
    if hybrid.get("city_code"):
        payload["city_code"] = hybrid["city_code"]
    if hybrid.get("lot_code"):
        payload["lot_code"] = hybrid["lot_code"]
    if hybrid.get("at_time"):
        payload["at_time"] = hybrid["at_time"]
    return payload


def _build_hybrid_payload(item: EvalQuery) -> dict[str, Any]:
    return dict(item.hybrid_request)


def _extract_executed_tools(intent: str, business_facts: dict[str, Any]) -> list[str]:
    tools: list[str] = [str(x) for x in business_facts.get("attempted_tools", []) if str(x)]
    if intent == "arrears_check" and ("arrears_count" in business_facts or "orders" in business_facts):
        tools.append("GET /api/v1/arrears-orders")
    if intent == "fee_verify":
        if "order" in business_facts:
            tools.append("GET /api/v1/parking-orders/{order_no}")
        if "simulation" in business_facts:
            tools.append("POST /api/v1/billing-rules/simulate")
    # keep order while removing duplicates
    return list(dict.fromkeys(tools))


def _evaluate_answer_text(expected_answer: dict[str, Any], hybrid_body: dict[str, Any]) -> bool:
    combined_text = " ".join(
        [
            str(hybrid_body.get("conclusion", "")),
            " ".join(str(item) for item in hybrid_body.get("key_points", [])),
        ]
    )

    must_contain = [str(item) for item in expected_answer.get("must_contain", []) if str(item)]
    must_not_contain = [str(item) for item in expected_answer.get("must_not_contain", []) if str(item)]
    amount_check_expected = expected_answer.get("expected_amount_check_result")

    must_contain_ok = True
    if must_contain:
        must_contain_ok = any(token in combined_text for token in must_contain)
    must_not_contain_ok = all(token not in combined_text for token in must_not_contain)

    amount_check_ok = True
    if amount_check_expected is not None:
        facts = hybrid_body.get("business_facts", {})
        amount_check_ok = str(facts.get("amount_check_result", "")) == str(amount_check_expected)

    return must_contain_ok and must_not_contain_ok and amount_check_ok


async def _run_eval_async(
    dataset_path: Path,
    report_dir: Path,
    rag_base_url: str,
    timeout_seconds: float,
) -> int:
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    items = _load_eval_queries(dataset_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_results: list[EvalSampleResult] = []

    async def _post_json(
        client: httpx.AsyncClient, path: str, payload: dict[str, Any]
    ) -> tuple[int | None, dict[str, Any], str | None]:
        try:
            resp = await client.post(path, json=payload)
            status = resp.status_code
            body = resp.json() if status == 200 else {}
            return status, body, None
        except Exception as exc:  # noqa: BLE001
            return None, {}, f"{path}_error={exc.__class__.__name__}"

    async with httpx.AsyncClient(base_url=rag_base_url.rstrip("/"), timeout=timeout_seconds) as client:
        for item in items:
            errors: list[str] = []
            retrieval_count = 0
            citation_count = 0
            retrieval_ok = False
            citation_ok = False
            tool_ok = False
            answer_ok = False
            executed_tools: list[str] = []

            retrieve_status, retrieve_body, retrieve_err = await _post_json(
                client, "/api/v1/retrieve", _build_retrieve_payload(item)
            )
            if retrieve_err:
                errors.append(retrieve_err)
            elif retrieve_status != 200:
                errors.append(f"retrieve_status={retrieve_status}")

            retrieved_items = list(retrieve_body.get("items", [])) if retrieve_body else []
            retrieval_count = len(retrieved_items)
            retrieved_source_ids = {str(row.get("source_id", "")) for row in retrieved_items}
            expected_retrieval = item.expected_retrieval
            min_hit_count = int(expected_retrieval.get("min_hit_count", 0))
            must_include = {str(x) for x in expected_retrieval.get("must_include_source_ids", [])}
            must_exclude = {str(x) for x in expected_retrieval.get("must_exclude_source_ids", [])}
            retrieval_ok = (
                retrieval_count >= min_hit_count
                and must_include.issubset(retrieved_source_ids)
                and retrieved_source_ids.isdisjoint(must_exclude)
            )
            if not retrieval_ok:
                errors.append("retrieval_expectation_failed")

            hybrid_status, hybrid_body, hybrid_err = await _post_json(
                client, "/api/v1/answer/hybrid", _build_hybrid_payload(item)
            )
            if hybrid_err:
                errors.append(hybrid_err)
            elif hybrid_status != 200:
                errors.append(f"hybrid_status={hybrid_status}")

            citations = list(hybrid_body.get("citations", [])) if hybrid_body else []
            citation_count = len(citations)
            citation_source_ids = {str(row.get("source_id", "")) for row in citations}
            citation_ok = (not must_include) or bool(citation_source_ids.intersection(must_include))
            if not citation_ok:
                errors.append("citation_expectation_failed")

            business_facts = dict(hybrid_body.get("business_facts", {})) if hybrid_body else {}
            intent = _resolve_intent(item)
            executed_tools = _extract_executed_tools(intent, business_facts)
            tool_ok = set(item.expected_tools).issubset(set(executed_tools))
            if not tool_ok:
                errors.append("tool_expectation_failed")

            answer_ok = _evaluate_answer_text(item.expected_answer, hybrid_body)
            if not answer_ok:
                errors.append("answer_expectation_failed")

            sample_results.append(
                EvalSampleResult(
                    eval_id=item.eval_id,
                    group=item.group,
                    intent=intent,
                    retrieval_ok=retrieval_ok,
                    citation_ok=citation_ok,
                    tool_ok=tool_ok,
                    answer_ok=answer_ok,
                    retrieval_count=retrieval_count,
                    citation_count=citation_count,
                    expected_tools=item.expected_tools,
                    executed_tools=executed_tools,
                    errors=errors,
                )
            )

    total = len(sample_results)
    retrieval_hit_rate = sum(1 for x in sample_results if x.retrieval_ok) / total if total else 0.0
    citation_coverage = sum(1 for x in sample_results if x.citation_ok) / total if total else 0.0
    empty_retrieval_rate = sum(1 for x in sample_results if x.retrieval_count == 0) / total if total else 0.0
    tool_call_compliance_rate = sum(1 for x in sample_results if x.tool_ok) / total if total else 0.0
    answer_consistency_rate = sum(1 for x in sample_results if x.answer_ok) / total if total else 0.0

    summary = EvalSummary(
        total=total,
        retrieval_hit_rate=round(retrieval_hit_rate, 4),
        citation_coverage=round(citation_coverage, 4),
        empty_retrieval_rate=round(empty_retrieval_rate, 4),
        tool_call_compliance_rate=round(tool_call_compliance_rate, 4),
        answer_consistency_rate=round(answer_consistency_rate, 4),
    )
    (report_dir / "rag006_eval_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failures = [row for row in sample_results if row.errors]
    (report_dir / "rag006_eval_failures.jsonl").write_text(
        "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in failures) + ("\n" if failures else ""),
        encoding="utf-8",
    )
    return 0


def run_eval(
    dataset_path: Path,
    report_dir: Path,
    rag_base_url: str = "http://127.0.0.1:8002",
    timeout_seconds: float = 30.0,
) -> int:
    try:
        return asyncio.run(
            _run_eval_async(
                dataset_path=dataset_path,
                report_dir=report_dir,
                rag_base_url=rag_base_url,
                timeout_seconds=timeout_seconds,
            )
        )
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "rag006_eval_summary.json").write_text(
            json.dumps({"status": "error", "error": exc.__class__.__name__}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (report_dir / "rag006_eval_failures.jsonl").write_text("", encoding="utf-8")
        return 2
