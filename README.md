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

# create biz database
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_biz;"

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
RAG_DEEPSEEK_API_KEY=
RAG_DEEPSEEK_BASE_URL=https://api.deepseek.com
RAG_DEEPSEEK_MODEL=deepseek-chat
RAG_BIZ_API_BASE_URL=http://127.0.0.1:8001
RAG_BIZ_API_TIMEOUT_SECONDS=10
```

## Run APIs
```bash
uvicorn agent_parksuite_biz_api.main:app --reload --port 8001
uvicorn agent_parksuite_rag_core.main:app --reload --port 8002
```

## Logging And Trace
- Both modules use `Loguru + contextvars`, default output to console.
- Incoming request headers:
  - `X-Trace-Id` (optional; auto-generated if missing)
- `rag-core` -> `biz-api` httpx calls will propagate `X-Trace-Id` for cross-service tracing.

## DB Migrations (Alembic)
```bash
# Biz DB
BIZ_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz alembic upgrade head

# Rag DB
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
```

If you see `No module named 'greenlet'`, install dependencies again:
```bash
pip install -e .[dev]
```

## Init Biz Tables
推荐方式（使用 Alembic 初始化 `parksuite_biz`）：
```bash
BIZ_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz alembic upgrade head
```

## Init RAG Core Tables
使用 Alembic 初始化 `parksuite_rag`（会创建 `knowledge_sources` / `knowledge_chunks`）：
```bash
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
```

## Init RAG Test Dataset (RAG-000)
建议使用独立种子库 `parksuite_biz_seed`（不要复用 `parksuite_biz_test`）：
```bash
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_biz_seed;"
BIZ_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed alembic upgrade head
```

执行种子脚本（写入 `parksuite_biz_seed`，同时导出 JSONL）：
```bash
python scripts/rag000_seed_biz_scenarios.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed \
  --export-jsonl data/rag000/scenarios.jsonl
```

Optional:
```bash
# no JSONL export
python scripts/rag000_seed_biz_scenarios.py --no-export
```

RAG-000 场景覆盖清单：
- 周期计费基础：可整除 / 不可整除
- 日间封顶：按日重置（跨天后重新计费）
- 夜间免费：跨午夜时段
- 日夜双时段组合：日间封顶 + 夜间封顶
- 阶梯计费：2小时内与2小时外不同单价
- 首30分钟免费边界：29 / 30 / 31 分钟
- 多日停车：中间完整天 + 尾天部分时段
- 规则版本切换：生效时间前后命中不同版本
- 多停车场差异：同城不同 `lot_code` 命中不同规则
- 欠费判断：全额已付 / 部分支付 / 未支付
- 金额核验一致：订单金额 = 模拟金额
- 金额核验不一致：输出 `需人工复核`

初始化完成后的预期数据量（脚本可重复执行，会先清理 `SCN-*` 再重建）：
- `billing_rules`: 5 条
- `billing_rule_versions`: 6 条
- `parking_orders`: 22 条
- `data/rag000/scenarios.jsonl`: 20 行

可用以下 SQL 核对：
```sql
SELECT COUNT(*) FROM billing_rules WHERE rule_code LIKE 'SCN-%';
SELECT COUNT(*) FROM billing_rule_versions brv
JOIN billing_rules br ON br.id = brv.rule_id
WHERE br.rule_code LIKE 'SCN-%';
SELECT COUNT(*) FROM parking_orders WHERE order_no LIKE 'SCN-%';
```

## Biz API endpoints
- `POST /api/v1/billing-rules` upsert billing rule
- `GET /api/v1/billing-rules` list billing rules
- `GET /api/v1/billing-rules/{rule_code}` billing rule detail
- `POST /api/v1/billing-rules/simulate` simulate fee
- `POST /api/v1/parking-orders` create parking order
- `GET /api/v1/parking-orders/{order_no}` parking order detail
- `GET /api/v1/arrears-orders` list arrears orders

## RAG Core endpoints
- `POST /api/v1/knowledge/sources` upsert knowledge source metadata
- `POST /api/v1/knowledge/chunks/batch` batch ingest chunks for a source
- `POST /api/v1/retrieve` retrieve chunks by metadata filters (optional vector ranking)
- `POST /api/v1/answer` generate answer with conclusion/key points/citations (DeepSeek)
- `POST /api/v1/answer/hybrid` hybrid answer (LangGraph + LLM intent routing + biz tools + RAG evidence)

### `/api/v1/answer/hybrid` 逻辑分支图
```mermaid
flowchart TD
    A[POST /api/v1/answer/hybrid] --> B[intent_classifier<br/>LLM classify + rule fallback]
    B -->|rule_explain| C[rule_explain_flow]
    B -->|arrears_check| D[arrears_check_flow]
    B -->|fee_verify| E[fee_verify_flow]

    C --> F[rag_retrieve]
    E --> F
    D --> G[answer_synthesizer]
    F --> G
    G --> H[HybridAnswerResponse]
```

### Retrieval 术语对照（中英）
- `retrieve`：召回（从知识库取回候选内容，不是最终回答）
- `retrieval`：检索/召回过程
- `chunk` / `chunks`：知识分块（文档切分后的最小检索单元）
- `query`：查询文本（用户问题或检索语句）
- `query_embedding`：查询向量（`query` 的向量化表示）
- `embedding`：向量表示（文本语义向量）
- `vector similarity ranking`：向量相似度排序（按向量距离/相似度重排）
- `metadata filters`：元数据过滤（如 `doc_type/city_code/lot_code/at_time`）
- `top_k`：召回条数上限（返回前 K 条）
- `source` / `knowledge source`：知识来源（文档级来源记录）

## Documentation
- [RAG Ingestion](docs/rag_ingestion.md)
- [PR Acceptance Notes](docs/pr_acceptance.md)
- [Testing](docs/testing.md)
