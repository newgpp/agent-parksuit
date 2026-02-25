# RAG-009 Short-Term Memory Acceptance Dataset

This dataset is for multi-turn short-term memory acceptance, not for retrieval scoring.

## Purpose
- Validate conversational carry-over in one session.
- Validate deterministic reference resolution (`第一笔` / `上一单` / `这笔`).
- Validate follow-up parameter completion (`order_no` omitted in turn-2).

## Dataset File
- `memory_acceptance_cases.jsonl`

## Per-line Schema
- `case_id`: stable test case id.
- `scenario`: conversation scenario label.
- `session_id`: target session id for all turns in this case.
- `turns`: ordered turn list.

Turn object:
- `turn_id`: stable step id inside case.
- `hybrid_request`: request payload target for `POST /api/v1/answer/hybrid` (includes `session_id`).
- `expect`: acceptance expectations.

## Replay
Run PR-4 replay against a running `rag-core` service:

```bash
python scripts/rag009_replay_memory_acceptance.py \
  --dataset-path data/rag009/memory_acceptance_cases.jsonl \
  --base-url http://127.0.0.1:8002
```

Common options:
- `--stop-on-fail`: stop on first failed turn.
- `--max-cases N`: replay first `N` cases only.

## Notes
- This dataset is prepared in PR-1 and consumed by PR-4 replay/evaluation.
- The replay validates behavior against `expect` rules; failed checks indicate implementation and expectation drift.
