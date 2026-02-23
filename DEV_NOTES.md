# DEV Notes

## Modules

### `agent-parksuite-biz-api`
- Tech: FastAPI + SQLAlchemy async + PostgreSQL(JSONB)
- Purpose: parking billing rule configuration/query, parking order query, arrears query, fee simulation
- Entry: `src/agent_parksuite_biz_api/main.py`
- Key files:
  - API: `src/agent_parksuite_biz_api/api/routes.py`
  - Models: `src/agent_parksuite_biz_api/db/models.py`
  - Billing engine: `src/agent_parksuite_biz_api/services/billing_engine.py`
  - Schemas: `src/agent_parksuite_biz_api/schemas/billing.py`, `src/agent_parksuite_biz_api/schemas/order.py`

### `agent-parksuite-rag-core`
- Tech: FastAPI + SQLAlchemy async + PostgreSQL + pgvector
- Purpose: RAG core skeleton (knowledge chunk storage + retrieve API placeholder)
- Entry: `src/agent_parksuite_rag_core/main.py`
- Key files:
  - API: `src/agent_parksuite_rag_core/api/routes.py`
  - Models: `src/agent_parksuite_rag_core/db/models.py`
  - Session/init: `src/agent_parksuite_rag_core/db/session.py`
- Next design scope:
  - Rule explanation for user questions (how fees are calculated)
  - Evidence-backed responses with source/version/effective time
  - Hybrid flow with biz tools for arrears check and fee verification

## Biz API PR List

### BIZ-001: Project bootstrap and dependency baseline
- Add monorepo layout, package config, test config
- Add shared dependencies (FastAPI/Pydantic/LangChain stack + SQLAlchemy + asyncpg + pgvector)
- Files:
  - `pyproject.toml`
  - `README.md`
  - `.env.example`

### BIZ-002: Biz API app skeleton and DB initialization
- Add Biz API startup entry and health endpoint
- Add async session factory
- Files:
  - `src/agent_parksuite_biz_api/main.py`
  - `src/agent_parksuite_biz_api/config.py`
  - `src/agent_parksuite_biz_api/db/base.py`
  - `src/agent_parksuite_biz_api/db/session.py`

### BIZ-003: Billing rule domain model refactor (rule + version)
- Replace single billing rule design with:
  - `billing_rules` (identity/scope/status)
  - `billing_rule_versions` (effective time ranges + payload)
- Add JSONB scope and `lot_codes` support
- Add GIN index for scope query and unique constraint on `(rule_id, version_no)`
- Files:
  - `src/agent_parksuite_biz_api/db/models.py`
  - `src/agent_parksuite_biz_api/schemas/billing.py`
  - `src/agent_parksuite_biz_api/schemas/order.py`

### BIZ-004: Billing rule APIs and versioned simulation
- Implement APIs:
  - `POST /api/v1/billing-rules` (upsert rule + append version)
  - `GET /api/v1/billing-rules`
  - `GET /api/v1/billing-rules/{rule_code}`
  - `POST /api/v1/billing-rules/simulate`
- Add overlap guard for version effective ranges
- Add scope filtering by `city_code` and `lot_code` (JSONB contains)
- Files:
  - `src/agent_parksuite_biz_api/api/routes.py`

### BIZ-005: Parking order and arrears query APIs
- Implement APIs:
  - `POST /api/v1/parking-orders`
  - `GET /api/v1/parking-orders/{order_no}`
  - `GET /api/v1/arrears-orders`
- Add arrears amount calculation on create
- Files:
  - `src/agent_parksuite_biz_api/api/routes.py`
  - `src/agent_parksuite_biz_api/schemas/order.py`
  - `src/agent_parksuite_biz_api/db/models.py`

### BIZ-006: Billing engine enhancement (payload-driven)
- Support `periodic`, `tiered`, `free` segments by `rule_payload`
- Support cross-day time windows (e.g. `20:00-08:00`)
- Support non-divisible unit rounding (ceil charging)
- Change cap logic to daily cap per segment/time_window reset by day
- Files:
  - `src/agent_parksuite_biz_api/services/billing_engine.py`

### BIZ-007: Billing engine test scenarios
- Add/refresh unit tests:
  - periodic with free minutes + cap
  - periodic non-divisible unit rounding
  - periodic across days with daytime cap reset
  - day+night periodic with independent caps
  - tiered across days + night free (including end at 08:29 case)
- Files:
  - `tests/biz_api/test_billing_engine.py`

### BIZ-008: Alembic migration baseline (Doc + Implementation)
- Goal:
  - Introduce versioned DB schema migration workflow
  - Provide initial migration that matches current SQLAlchemy models
- Scope:
  - Add Alembic config and environment wiring
  - Add initial migration to create biz/rag tables and key constraints/indexes
  - Document migration commands and usage
- Deliverables:
  - `alembic.ini`
  - `alembic/env.py`
  - `alembic/script.py.mako`
  - `alembic/versions/<rev>_biz_init_schema.py`
  - `alembic/versions/<rev>_rag_init_schema.py`
  - README updates (migration section)
- Acceptance:
  - `alembic upgrade head` can build schema from empty DB
  - `alembic downgrade base` can roll back initial schema

### BIZ-009: Biz API route integration tests and test-data policy
- Add route-level integration tests for billing and order APIs
- Add test fixtures for dedicated DB, dependency override, and per-test isolation
- Add optional `KEEP_TEST_DATA=1` to preserve test tables/data for debugging
- Use unique test identifiers to avoid collisions when keeping historical test data
- Files:
  - `tests/biz_api/conftest.py`
  - `tests/biz_api/test_routes_billing.py`
  - `tests/biz_api/test_routes_orders.py`
  - `README.md`

## Rag Core PR Plan

### RAG-000: Biz scenario dataset baseline (pre-RAG)
- Build a deterministic business scenario dataset from `biz-api` domain:
  - billing rules (multi-version, multi-lot)
  - parking orders (paid/unpaid/arrears)
  - simulation snapshots for verification cases
- Define scenario IDs and expected facts for each case:
  - expected arrears status
  - expected simulated amount
  - expected rule version hit
- Add export format for downstream RAG generation:
  - JSON/JSONL with scenario metadata and expected outputs
- Acceptance:
  - one command can initialize scenario dataset in test DB
  - scenario facts are reproducible and can be used as eval baseline
- Implementation entry:
  - `scripts/rag000_seed_biz_scenarios.py`

#### RAG-000 Data Dictionary
- `scenario_id`: stable unique ID, e.g. `SCN-001`
- `intent_tags`: one or more tags:
  - `rule_explain`
  - `arrears_check`
  - `fee_verify`
- `query`: natural language user question
- `context`:
  - `city_code`
  - `lot_code`
  - `plate_no`
  - `order_no` (optional)
  - `entry_time`
  - `exit_time`
- `expected_tools`: expected biz-api calls, e.g.
  - `GET /api/v1/arrears-orders`
  - `GET /api/v1/parking-orders/{order_no}`
  - `POST /api/v1/billing-rules/simulate`
- `ground_truth`:
  - `matched_rule_code`
  - `matched_version_no`
  - `expected_total_amount`
  - `order_total_amount`
  - `amount_check_result` (`一致`/`不一致`)
  - `amount_check_action` (`自动通过`/`需人工复核`)
  - `expected_paid_amount`
  - `expected_arrears_amount`
  - `expected_arrears_status` (`NONE`/`HAS_ARREARS`)
- `expected_citations`:
  - `doc_type` (`rule_explain`/`policy_doc`/`faq`/`sop`)
  - `source_ids` (one or more chunk/source IDs)
- `notes`: boundary condition or special check description

#### RAG-000 Scenario Set (20 cases)
- `SCN-001` 周期计费-整除:
  - 08:00-09:00, 30分钟2元, 预期4元
- `SCN-002` 周期计费-非整除进位:
  - 08:00-09:05, 30分钟2元, 预期6元
- `SCN-003` 日间封顶:
  - 08:00-20:00+, 日间封顶20元, 预期20元
- `SCN-004` 跨天日间封顶重置:
  - 连续多天停车, 每天日间封顶独立重置
- `SCN-005` 夜间免费:
  - 20:00-08:00 免费, 仅夜间停车预期0元
- `SCN-006` 日夜组合:
  - 日间收费+夜间免费, 校验分段金额
- `SCN-007` 阶梯计费:
  - 首2小时每半小时2元, 之后每半小时3元
- `SCN-008` 首30分钟免费边界:
  - 29/30/31分钟三个订单对比
- `SCN-009` 欠费判断:
  - 同车牌含已支付、部分支付、未支付订单
- `SCN-010`~`SCN-011`:
  - 首30分钟免费边界补足（30分钟/31分钟）
- `SCN-012`:
  - 日夜双时段组合，日间与夜间各自封顶
- `SCN-013`~`SCN-014`:
  - 阶梯计费2小时内/外差异
- `SCN-015`~`SCN-016`:
  - 规则版本切换（生效前后）
- `SCN-017`~`SCN-018`:
  - 同城不同 `lot_code` 差异
- `SCN-019`:
  - 计费核验一致（订单金额=模拟金额）
- `SCN-020`:
  - 计费核验异常（订单金额!=模拟金额，结论“需人工复核”）

#### RAG-000 Seed Expected Counts
- `billing_rules`: 5
- `billing_rule_versions`: 6
- `parking_orders`: 22
- `scenarios.jsonl`: 20

#### RAG-000 Coverage Matrix
- Time boundaries:
  - `07:59`, `08:00`, `19:59`, `20:00`, `20:01`
- Cross-day boundary:
  - `23:59 -> 00:01`
- Version boundary:
  - `effective_from` hit at exact timestamp
- Cap boundary:
  - equal to cap
  - exceed cap by one billing unit

### RAG-001: RAG data model and storage upgrade
- Extend `rag-core` schema with metadata for retrieval filtering:
  - `doc_type`, `city_code`, `lot_codes`, `effective_from`, `effective_to`, `source`, `version`
- Add indexes for metadata filtering + vector search
- Input source aligned with `RAG-000` outputs (`source_type=biz_derived`)
- Acceptance:
  - can insert and query knowledge chunks with metadata filters

### RAG-002: Ingestion pipeline
- Add ingestion flow: clean text -> chunk -> embedding -> upsert
- Support batch import from JSONL/Markdown sources
- Support derived knowledge generation from biz scenarios:
  - rule explanation text
  - scenario-based FAQ
  - fee calculation walk-through snippets
- Acceptance:
  - imported knowledge is retrievable and traceable by source/version

### RAG-003: Retrieve API (RAG retrieval core)
- Implement `POST /api/v1/retrieve`
- Support filters:
  - `city_code`, `lot_code`, `at_time`, `doc_type`, `top_k`
- Ensure retrieval prefers dataset-aligned chunks for scenario-based queries
- Acceptance:
  - retrieval honors metadata constraints and returns stable top-k results

### RAG-004: Answer API (RAG-only)
- Implement `POST /api/v1/answer` for explanation-style questions
- Output format:
  - conclusion + key points + cited chunks/sources
- Acceptance:
  - responses include usable citation fields and evidence snippets

### RAG-005: Hybrid orchestration (RAG + biz tools)
- Integrate biz tool outputs into answer composition for:
  - arrears check
  - fee verification
- Split final response into:
  - business facts (tool result)
  - rule/policy evidence (RAG result)
- Acceptance:
  - verification answers contain both computed facts and explainable references

### RAG-006: Evaluation baseline and seed dataset
- Build evaluation set (30-50 Q&A) for parking fee consultation
- Add basic offline eval scripts and baseline metrics:
  - retrieval hit rate
  - citation coverage
  - empty retrieval rate
- Acceptance:
  - one command can run baseline evaluation and output metric summary

### RAG-007: Explicit planner layer
- Add a planning step before execution:
  - identify intent type (explanation / arrears check / fee verification / mixed)
  - generate ordered action plan (RAG retrieve, biz-tool calls, final synthesis)
- Enforce structured plan output with plan IDs and step statuses
- Acceptance:
  - each answer request has an auditable execution plan
  - mixed queries trigger both retrieval and tool calls in deterministic order

### RAG-008: Evaluator-optimizer loop
- Add response evaluator after first draft:
  - check evidence sufficiency (citations present and relevant)
  - check tool coverage (required tool calls executed for intent)
  - check consistency between tool facts and generated text
- Add optimizer retry policy:
  - if evaluation fails, auto re-retrieve / re-call missing tools / regenerate once
- Acceptance:
  - failed first-pass answers can be corrected automatically
  - evaluation result is logged with pass/fail reason and retry path

## Open items
- Define and document `rule_payload` schema contract more strictly (JSON Schema / Pydantic typed segments)
