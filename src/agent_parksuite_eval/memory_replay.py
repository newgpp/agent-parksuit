from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class CheckResult:
    ok: bool
    messages: list[str]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _contains_in_trace(trace: list[str], token: str) -> bool:
    return any(token in item for item in trace)


def _evaluate_turn(expect: dict[str, Any], response: dict[str, Any]) -> CheckResult:
    errors: list[str] = []
    intent = str(response.get("intent", ""))
    trace = [str(item) for item in response.get("graph_trace", [])]
    facts = response.get("business_facts", {}) or {}
    attempted_tools = [str(item) for item in facts.get("attempted_tools", [])]
    combined_text = " ".join(
        [
            str(response.get("conclusion", "")),
            " ".join(str(item) for item in response.get("key_points", [])),
            json.dumps(facts, ensure_ascii=False),
            " ".join(trace),
        ]
    )

    must_intent = expect.get("must_intent")
    if must_intent and intent != must_intent:
        errors.append(f"intent mismatch: expected={must_intent}, got={intent}")

    for tool in expect.get("must_call_tools", []):
        if tool not in attempted_tools:
            errors.append(f"missing tool call: {tool}; attempted={attempted_tools}")

    for text in expect.get("must_contain", []):
        if str(text) not in combined_text:
            errors.append(f"missing text: {text}")

    memory_expect = expect.get("memory_expect", {}) or {}
    resolved_order_no = memory_expect.get("resolved_order_no")
    if resolved_order_no and str(facts.get("order_no", "")) != str(resolved_order_no):
        errors.append(
            f"resolved_order_no mismatch: expected={resolved_order_no}, got={facts.get('order_no')}"
        )

    if memory_expect.get("carry_intent_from") and not _contains_in_trace(trace, "memory_hydrate:intent_hint"):
        errors.append("carry_intent_from is no longer supported in memory_hydrate")

    for slot in memory_expect.get("carry_slots", []):
        if not _contains_in_trace(trace, f"memory_hydrate:{slot}"):
            errors.append(f"missing memory carry trace for slot: {slot}")

    reference_resolution = str(memory_expect.get("reference_resolution", ""))
    if "上一单->" in reference_resolution:
        if str(facts.get("error", "")) != "order_reference_needs_clarification":
            errors.append(
                "reference_resolution by memory is removed; expected error=order_reference_needs_clarification"
            )
        if not _contains_in_trace(trace, "memory_hydrate:order_reference_needs_clarification"):
            errors.append("missing memory_hydrate:order_reference_needs_clarification trace")

    if memory_expect.get("needs_disambiguation_when_multiple"):
        if str(facts.get("error", "")) != "order_reference_needs_clarification":
            errors.append(f"expected error=order_reference_needs_clarification, got={facts.get('error')}")
        if not _contains_in_trace(trace, "memory_hydrate:order_reference_needs_clarification"):
            errors.append("missing memory_hydrate:order_reference_needs_clarification trace")

    if expect.get("must_not_memory_carry"):
        if not _contains_in_trace(trace, "memory_hydrate:none"):
            errors.append("expected memory_hydrate:none for isolation check")

    return CheckResult(ok=not errors, messages=errors)


async def _run_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    stop_on_fail: bool,
) -> tuple[int, int]:
    case_id = str(case.get("case_id", ""))
    turns = case.get("turns", []) or []
    passed = 0
    failed = 0
    print(f"\n[case] {case_id} turns={len(turns)}", flush=True)

    for idx, turn in enumerate(turns, start=1):
        turn_id = str(turn.get("turn_id", f"turn-{idx}"))
        payload = turn.get("hybrid_request", {}) or {}
        expect = turn.get("expect", {}) or {}
        try:
            resp = await client.post("/api/v1/answer/hybrid", json=payload)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [fail] {turn_id} request error: {exc}", flush=True)
            if stop_on_fail:
                return passed, failed
            continue

        if resp.status_code != 200:
            failed += 1
            body = resp.text[:500]
            print(f"  [fail] {turn_id} status={resp.status_code} body={body}", flush=True)
            if stop_on_fail:
                return passed, failed
            continue

        result = _evaluate_turn(expect=expect, response=resp.json())
        if result.ok:
            passed += 1
            print(f"  [pass] {turn_id}", flush=True)
            continue

        failed += 1
        print(f"  [fail] {turn_id}", flush=True)
        for message in result.messages:
            print(f"    - {message}", flush=True)
        if stop_on_fail:
            return passed, failed

    return passed, failed


async def _run_memory_replay_async(
    dataset_path: Path,
    base_url: str,
    timeout_seconds: float,
    stop_on_fail: bool,
    max_cases: int,
) -> int:
    if not dataset_path.exists():
        print(f"[error] dataset not found: {dataset_path}")
        return 2

    cases = _load_cases(dataset_path)
    if max_cases:
        cases = cases[:max_cases]
    if not cases:
        print("[error] no cases loaded")
        return 2

    print(
        f"[start] base_url={base_url} dataset={dataset_path} cases={len(cases)} timeout={timeout_seconds}s",
        flush=True,
    )
    timeout = httpx.Timeout(timeout_seconds)
    passed = 0
    failed = 0
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, trust_env=False) as client:
        for case in cases:
            case_passed, case_failed = await _run_case(client=client, case=case, stop_on_fail=stop_on_fail)
            passed += case_passed
            failed += case_failed
            if stop_on_fail and failed:
                break

    total = passed + failed
    print(f"\n[summary] total_turns={total} passed={passed} failed={failed}", flush=True)
    return 0 if failed == 0 else 1


def run_memory_replay(
    dataset_path: Path,
    base_url: str = "http://127.0.0.1:8002",
    timeout_seconds: float = 30,
    stop_on_fail: bool = False,
    max_cases: int = 0,
) -> int:
    return asyncio.run(
        _run_memory_replay_async(
            dataset_path=dataset_path,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            stop_on_fail=stop_on_fail,
            max_cases=max_cases,
        )
    )
