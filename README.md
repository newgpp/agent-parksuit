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

## Init RAG Core Tables
使用 Alembic 初始化 `parksuite_rag`（会创建 `knowledge_sources` / `knowledge_chunks`）：
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
```

## Init RAG Test Dataset (RAG-000)
建议使用独立种子库 `parksuite_biz_seed`（不要复用 `parksuite_biz_test`）：
```bash
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_biz_seed;"
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed alembic upgrade head
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

## RAG-002 Ingestion Pipeline
从 `RAG-000` 场景集生成知识分块并写入 `parksuite_rag`：
```bash
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing
```

### RAG-002 初始化与验收（推荐顺序）
1. （可选）重建 `parksuite_rag`（当迁移状态异常/历史脏数据较多时推荐）
```bash
docker exec -it parksuite-pg psql -U postgres -d postgres -c "DROP DATABASE IF EXISTS parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d parksuite_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

2. 初始化 `parksuite_rag` 表结构
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
```
说明：当前迁移脚本会按“当前连接数据库名”判断执行目标，避免误跳过 `parksuite_rag` 初始化。

3. 生成 `RAG-000` 场景数据（写入 `parksuite_biz_seed` + 导出 JSONL）
```bash
python scripts/rag000_seed_biz_scenarios.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed \
  --export-jsonl data/rag000/scenarios.jsonl
```

4. 执行 `RAG-002` 入库（默认 deterministic embedding）
```bash
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing
```

5. SQL 验收（`parksuite_rag`）
```sql
-- source 总数（预期 40：20 个 scenario * 2 个 doc_type）
SELECT COUNT(*) FROM knowledge_sources WHERE source_id LIKE 'RAG000-%';

-- chunk 总数（应 > 0）
SELECT COUNT(*) FROM knowledge_chunks kc
JOIN knowledge_sources ks ON ks.id = kc.source_pk
WHERE ks.source_id LIKE 'RAG000-%';
```

6. 接口验收（启动 `rag-core` 后）
```bash
curl -X POST "http://127.0.0.1:8002/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "上海A场怎么计费",
    "query_embedding": null,
    "top_k": 5,
    "doc_type": "rule_explain",
    "source_type": "biz_derived",
    "city_code": "310100",
    "lot_code": "SCN-LOT-A",
    "at_time": "2026-02-10T10:00:00+08:00",
    "include_inactive": false
  }'
```
预期：返回 `items` 非空，且结果包含 `source_id/doc_type/content` 字段。

可选：使用 OpenAI embedding
```bash
export OPENAI_API_KEY=your_key
python scripts/rag002_ingest_knowledge.py \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small
```

可选：导入 Markdown 文档
```bash
python scripts/rag002_ingest_knowledge.py \
  --input-type markdown \
  --input-path docs/rag_sources \
  --source-prefix POLICY \
  --replace-existing
```

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

### Integration Tests
通用前置：
```bash
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_biz_test;"
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_rag_test;"
```

Biz API route integration tests need a dedicated test database (default: `parksuite_biz_test`):
```bash
export BIZ_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_test
pytest tests/biz_api/test_routes_billing.py tests/biz_api/test_routes_orders.py
```

RAG Core route integration tests need a dedicated test database (default: `parksuite_rag_test`):
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_rag.py
```

RAG retrieve API (`POST /api/v1/retrieve`) focused integration tests:
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_retrieve.py
```

RAG answer API (`POST /api/v1/answer`) route tests (LLM mocked):
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_answer.py
```
说明：
- 该用例验证 `POST /api/v1/answer` 的路由编排与返回结构（`conclusion/key_points/citations`）。
- 测试中已对 LLM 调用做 mock，不依赖真实 DeepSeek/OpenAI 网络请求。

Semantic-retrieval validation (paraphrase query -> vector recall, requires OpenAI embedding):
```bash
export OPENAI_API_KEY=your_key
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_retrieve_semantic.py
```

RAG hybrid answer API (`POST /api/v1/answer/hybrid`) integration tests (dataset-driven, tools/LLM mocked):
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_hybrid.py
```
说明：
- `test_routes_hybrid.py` 会从 `data/rag000/scenarios.jsonl` 读取场景作为测试输入。
- 用例覆盖 `fee_verify` 与 `arrears_check` 两条分支；请求中带 `intent_hint`，确保分支可稳定复现。
- 测试会 mock 掉 biz-api 调用与 LLM 生成，主要验证 `graph_trace/business_facts/citations` 的路由编排是否正确。
- 若你要先手动刷新场景数据，可执行：
```bash
python scripts/rag000_seed_biz_scenarios.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed \
  --export-jsonl data/rag000/scenarios.jsonl
```

To keep test data/tables for debugging:
```bash
export KEEP_TEST_DATA=1
pytest tests/biz_api/test_routes_billing.py tests/biz_api/test_routes_orders.py
```

### Manual E2E Tests (Real LLM)
完整步骤与示例请求已拆分到：
- [Manual E2E Answer Tests](docs/manual_e2e_answer_tests.md)
