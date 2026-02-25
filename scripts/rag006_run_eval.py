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
    args = parser.parse_args()

    raise SystemExit(run_eval(Path(args.dataset_path), Path(args.report_dir)))


if __name__ == "__main__":
    main()
