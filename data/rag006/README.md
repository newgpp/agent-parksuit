# RAG-006 Eval Dataset

Place evaluation samples in `eval_queries.jsonl`.

Expected line format (JSON object):
- `eval_id`
- `intent`
- `query`
- `context`
- `expected_retrieval`
- `expected_tools`
- `expected_answer`
