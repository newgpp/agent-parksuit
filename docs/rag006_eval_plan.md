# RAG-006 Evaluation Plan

## Goal
Build a reproducible offline evaluation baseline for RAG quality before RAG-007/RAG-008 changes.

## Target Size
- Total: 60 evaluation queries
- Split by intent:
  - `rule_explain`: 30
  - `arrears_check`: 12
  - `fee_verify`: 18

## Coverage Matrix
- Retrieval robustness:
  - paraphrase / colloquial query variants
  - metadata filters (`city_code`, `lot_code`, `doc_type`, `at_time`)
  - hard negatives (similar text but wrong lot/time)
  - empty retrieval cases (should explicitly return insufficient evidence)
- Business boundary cases:
  - 29/30/31 minutes
  - non-divisible billing units (e.g. 65 minutes)
  - daytime/nighttime split and cross-midnight
  - multi-day cap reset
  - version switch boundary (`effective_from` exact hit)
  - timezone consistency (`+08:00` vs `+00:00` inputs)
- Hybrid tool orchestration:
  - `rule_explain`: RAG-only explanation path
  - `arrears_check`: arrears tool + optional RAG evidence path
  - `fee_verify`: order detail + simulation + inconsistency decision path
  - error/partial-data cases (`order_no` missing, tool timeout)

## Dataset Structure (per eval sample)
- `eval_id`
- `intent`
- `query`
- `context`:
  - `city_code`, `lot_code`, `plate_no`, `order_no`, `at_time`
- `expected_retrieval`:
  - `min_hit_count`
  - `must_include_source_ids`
  - `must_exclude_source_ids`
- `expected_tools`:
  - required tool calls for hybrid scenarios
- `expected_answer`:
  - `must_contain` keywords/phrases
  - `must_not_contain` hallucination guards
  - `expected_amount_check_result` (for `fee_verify`)

## Metrics
- `retrieval_hit_rate`
  - definition: `must_include_source_ids` all hit / total eval samples
- `citation_coverage`
  - definition: response citations cover at least one expected source / total answer samples
- `empty_retrieval_rate`
  - definition: retrieval returns no evidence / total retrieval samples
- `tool_call_compliance_rate` (hybrid)
  - definition: required tool calls executed / hybrid samples
- `answer_consistency_rate` (hybrid)
  - definition: answer conclusion consistent with tool facts / hybrid samples

## Acceptance Baseline (initial)
- retrieval_hit_rate >= 0.80
- citation_coverage >= 0.85
- empty_retrieval_rate <= 0.10
- tool_call_compliance_rate = 1.00
- answer_consistency_rate >= 0.90

## Suggested Split for 60 Samples
- Group A (20): direct scenario-aligned queries from RAG-000
- Group B (20): semantic paraphrases (wording changes, same intent)
- Group C (10): hard negatives and no-evidence questions
- Group D (10): hybrid failure modes and boundary stress cases

## Execution Outline
1. Generate eval query file (`data/rag006/eval_queries.jsonl`).
2. Run offline evaluator script (to be added in RAG-006):
   - call `/api/v1/retrieve` and `/api/v1/answer/hybrid`
   - collect outputs and tool traces
3. Produce metric report:
   - `reports/rag006_eval_summary.json`
   - `reports/rag006_eval_failures.jsonl`
4. Use report as baseline gate for later PRs.
