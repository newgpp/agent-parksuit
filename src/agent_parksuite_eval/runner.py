from __future__ import annotations

from pathlib import Path


def run_eval(dataset_path: Path, report_dir: Path) -> int:
    """RAG-006 evaluator entrypoint placeholder.

    Returns process exit code (0 for success).
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    # TODO(RAG-006): implement query replay and metric aggregation.
    (report_dir / "rag006_eval_summary.json").write_text(
        '{"status":"todo","message":"RAG-006 evaluator not implemented yet"}\n',
        encoding="utf-8",
    )
    (report_dir / "rag006_eval_failures.jsonl").write_text("", encoding="utf-8")
    return 0
