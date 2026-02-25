from __future__ import annotations

import argparse
from pathlib import Path
from agent_parksuite_eval.memory_replay import run_memory_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay RAG-009 memory acceptance dataset against /answer/hybrid")
    parser.add_argument(
        "--dataset-path",
        default="data/rag009/memory_acceptance_cases.jsonl",
        help="path to memory acceptance jsonl",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8002",
        help="rag-core service base url",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30,
        help="request timeout seconds",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="stop at first failed turn",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="limit case count (0 means all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code = run_memory_replay(
        dataset_path=Path(args.dataset_path),
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        stop_on_fail=args.stop_on_fail,
        max_cases=args.max_cases,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
