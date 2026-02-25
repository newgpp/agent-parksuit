from __future__ import annotations

import argparse
from pathlib import Path

from agent_parksuite_eval.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG-006 offline evaluation baseline")
    parser.add_argument(
        "--dataset-path",
        default="data/rag006/eval_queries.jsonl",
        help="Path to evaluation dataset JSONL",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for evaluation reports",
    )
    parser.add_argument(
        "--rag-base-url",
        default="http://127.0.0.1:8002",
        help="RAG core API base URL",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout per request (seconds)",
    )
    args = parser.parse_args()

    code = run_eval(
        dataset_path=Path(args.dataset_path),
        report_dir=Path(args.report_dir),
        rag_base_url=args.rag_base_url,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"RAG-006 eval done. summary={Path(args.report_dir) / 'rag006_eval_summary.json'}")
    print(f"RAG-006 eval failures={Path(args.report_dir) / 'rag006_eval_failures.jsonl'}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
