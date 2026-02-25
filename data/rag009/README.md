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
- `hybrid_request`: request payload target for `POST /api/v1/answer/hybrid` (future contract includes `session_id`).
- `expect`: acceptance expectations.

## Notes
- Current project has not enabled short-term memory API contract yet.
- This dataset is prepared first for PR-1, then consumed by PR-2/PR-3 implementation and PR-4 replay tests.
