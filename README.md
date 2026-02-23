# agent-parksuit

Phase 1 monorepo with two FastAPI modules:
- `agent_parksuite_biz_api`: billing rules + parking order APIs
- `agent_parksuite_rag_core`: RAG core skeleton backed by PostgreSQL + pgvector

## Requirements
- Python 3.11+
- PostgreSQL 15+ with `pgvector` extension

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Start PostgreSQL (Docker)
```bash
docker run -d --name parksuite-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=parksuite_biz \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# create rag database
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_rag;"

# enable pgvector extension in rag database
docker exec -it parksuite-pg psql -U postgres -d parksuite_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Environment
Use `.env` (optional):
```env
BIZ_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag
RAG_EMBEDDING_DIM=1536
```

## Run APIs
```bash
uvicorn agent_parksuite_biz_api.main:app --reload --port 8001
uvicorn agent_parksuite_rag_core.main:app --reload --port 8002
```

## DB Migrations (Alembic)
```bash
# Biz DB
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz alembic upgrade head

# Rag DB
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
```

If you see `No module named 'greenlet'`, install dependencies again:
```bash
pip install -e .[dev]
```

## Init Biz Tables
推荐方式（使用 Alembic 初始化 `parksuite_biz`）：
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz alembic upgrade head
```

## Biz API endpoints
- `POST /api/v1/billing-rules` upsert billing rule
- `GET /api/v1/billing-rules` list billing rules
- `GET /api/v1/billing-rules/{rule_code}` billing rule detail
- `POST /api/v1/billing-rules/simulate` simulate fee
- `POST /api/v1/parking-orders` create parking order
- `GET /api/v1/parking-orders/{order_no}` parking order detail
- `GET /api/v1/arrears-orders` list arrears orders

### Billing rule payload example
```json
{
  "rule_code": "SH-PUDONG-A",
  "name": "Pudong A v1",
  "status": "enabled",
  "scope": {
    "scope_type": "lot_code",
    "city_code": "310100",
    "lot_codes": ["LOT-A", "LOT-B"]
  },
  "version": {
    "effective_from": "2026-02-23T00:00:00",
    "effective_to": null,
    "priority": 100,
    "rule_payload": [
      {
        "name": "day_periodic",
        "type": "periodic",
        "time_window": {"start": "08:00", "end": "22:00"},
        "unit_minutes": 30,
        "unit_price": 2,
        "free_minutes": 30,
        "max_charge": 30
      },
      {
        "name": "night_free",
        "type": "free",
        "time_window": {"start": "22:00", "end": "08:00"}
      }
    ]
  }
}
```

## Tests
```bash
pytest
```

Biz API route integration tests need a dedicated test database (default: `parksuite_biz_test`):
```bash
export BIZ_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_test
pytest tests/biz_api/test_routes_billing.py tests/biz_api/test_routes_orders.py
```

To keep test data/tables for debugging:
```bash
export KEEP_TEST_DATA=1
pytest tests/biz_api/test_routes_billing.py tests/biz_api/test_routes_orders.py
```
