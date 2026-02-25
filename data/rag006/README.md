# RAG-006 Eval Dataset

Place evaluation samples in `eval_queries.jsonl`.

Expected line format (JSON object):
- `eval_id`
- `group` (`A_direct` / `B_paraphrase` / `C_negative` / `D_hybrid_boundary`)
- `hybrid_request` (direct payload for `POST /api/v1/answer/hybrid` manual replay, intent comes from `intent_hint`)
- `expected_retrieval`
- `expected_tools`
- `expected_answer`

Current baseline dataset:
- total: 60
- intent split: `rule_explain=30`, `arrears_check=12`, `fee_verify=18`
