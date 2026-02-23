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

## Biz API PR List

### PR-01: Project bootstrap and dependency baseline
- Add monorepo layout, package config, test config
- Add shared dependencies (FastAPI/Pydantic/LangChain stack + SQLAlchemy + asyncpg + pgvector)
- Files:
  - `pyproject.toml`
  - `README.md`
  - `.env.example`

### PR-02: Biz API app skeleton and DB initialization
- Add Biz API startup entry and health endpoint
- Add async session factory
- Files:
  - `src/agent_parksuite_biz_api/main.py`
  - `src/agent_parksuite_biz_api/config.py`
  - `src/agent_parksuite_biz_api/db/base.py`
  - `src/agent_parksuite_biz_api/db/session.py`

### PR-03: Billing rule domain model refactor (rule + version)
- Replace single billing rule design with:
  - `billing_rules` (identity/scope/status)
  - `billing_rule_versions` (effective time ranges + payload)
- Add JSONB scope and `lot_codes` support
- Add GIN index for scope query and unique constraint on `(rule_id, version_no)`
- Files:
  - `src/agent_parksuite_biz_api/db/models.py`
  - `src/agent_parksuite_biz_api/schemas/billing.py`
  - `src/agent_parksuite_biz_api/schemas/order.py`

### PR-04: Billing rule APIs and versioned simulation
- Implement APIs:
  - `POST /api/v1/billing-rules` (upsert rule + append version)
  - `GET /api/v1/billing-rules`
  - `GET /api/v1/billing-rules/{rule_code}`
  - `POST /api/v1/billing-rules/simulate`
- Add overlap guard for version effective ranges
- Add scope filtering by `city_code` and `lot_code` (JSONB contains)
- Files:
  - `src/agent_parksuite_biz_api/api/routes.py`

### PR-05: Parking order and arrears query APIs
- Implement APIs:
  - `POST /api/v1/parking-orders`
  - `GET /api/v1/parking-orders/{order_no}`
  - `GET /api/v1/arrears-orders`
- Add arrears amount calculation on create
- Files:
  - `src/agent_parksuite_biz_api/api/routes.py`
  - `src/agent_parksuite_biz_api/schemas/order.py`
  - `src/agent_parksuite_biz_api/db/models.py`

### PR-06: Billing engine enhancement (payload-driven)
- Support `periodic`, `tiered`, `free` segments by `rule_payload`
- Support cross-day time windows (e.g. `20:00-08:00`)
- Support non-divisible unit rounding (ceil charging)
- Change cap logic to daily cap per segment/time_window reset by day
- Files:
  - `src/agent_parksuite_biz_api/services/billing_engine.py`

### PR-07: Billing engine test scenarios
- Add/refresh unit tests:
  - periodic with free minutes + cap
  - periodic non-divisible unit rounding
  - periodic across days with daytime cap reset
  - day+night periodic with independent caps
  - tiered across days + night free (including end at 08:29 case)
- Files:
  - `tests/biz_api/test_billing_engine.py`

### PR-08: Alembic migration baseline (Doc + Implementation)
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

### PR-09: Biz API route integration tests and test-data policy
- Add route-level integration tests for billing and order APIs
- Add test fixtures for dedicated DB, dependency override, and per-test isolation
- Add optional `KEEP_TEST_DATA=1` to preserve test tables/data for debugging
- Use unique test identifiers to avoid collisions when keeping historical test data
- Files:
  - `tests/biz_api/conftest.py`
  - `tests/biz_api/test_routes_billing.py`
  - `tests/biz_api/test_routes_orders.py`
  - `README.md`

## Open items
- Define and document `rule_payload` schema contract more strictly (JSON Schema / Pydantic typed segments)
