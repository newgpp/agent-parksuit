# agent-parksuit

面向停车计费场景的 Agent 工程项目（FastAPI + LangGraph + PostgreSQL + pgvector），包含业务工具服务、RAG 服务、离线评测模块。

## Highlights
- Hybrid Agent 链路可落地：`RAG + biz tools + LLM`，已支持 `rule_explain / arrears_check / fee_verify` 三类真实场景。
- 工程闭环完整：迁移、种子数据、入库、接口、集成测试、E2E、离线评测（`RAG-006`）。
- 可观测性到位：`X-Trace-Id` 跨服务透传、结构化日志、`graph_trace` 可追踪执行分支。

## System Architecture
- `agent_parksuite_biz_api`:
  - 计费规则配置与版本管理
  - 停车订单与欠费查询
  - 计费模拟
- `agent_parksuite_rag_core`:
  - 知识源/知识分块存储
  - 检索与回答接口
  - Hybrid 编排（LangGraph + biz tools）
- `agent_parksuite_eval`:
  - 评测数据集（RAG-006）
  - 评测回放脚本与报告

## Eval Snapshot (RAG-006)
当前离线评测基线（60 条）：
- `retrieval_hit_rate`: `1.0`
- `citation_coverage`: `1.0`
- `tool_call_compliance_rate`: `1.0`
- `answer_consistency_rate`: `1.0`
- `empty_retrieval_rate`: `0.1`（主要来自负样本）

## Core API Capabilities

### `/api/v1/answer/hybrid` Flow
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

### Biz API
- `POST /api/v1/billing-rules` upsert billing rule
- `GET /api/v1/billing-rules` list billing rules
- `GET /api/v1/billing-rules/{rule_code}` billing rule detail
- `POST /api/v1/billing-rules/simulate` simulate fee
- `POST /api/v1/parking-orders` create parking order
- `GET /api/v1/parking-orders/{order_no}` parking order detail
- `GET /api/v1/arrears-orders` list arrears orders

### RAG Core
- `POST /api/v1/knowledge/sources` upsert knowledge source metadata
- `POST /api/v1/knowledge/chunks/batch` batch ingest chunks for a source
- `POST /api/v1/retrieve` retrieve chunks by metadata filters (optional vector ranking)
- `POST /api/v1/answer` generate answer with conclusion/key points/citations (DeepSeek)
- `POST /api/v1/answer/hybrid` hybrid answer (LangGraph + one-shot intent routing + biz tools + optional RAG evidence)


## Quick Start

### Requirements
- Python 3.11+
- PostgreSQL 15+ with `pgvector` extension

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Start PostgreSQL (Docker)
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

### Environment
Use `.env` (optional):
```env
BIZ_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz
BIZ_LOG_TO_STDOUT=true
BIZ_LOG_TO_FILE=false
BIZ_LOG_DIR=logs
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag
RAG_EMBEDDING_DIM=1536
RAG_DEEPSEEK_API_KEY=
RAG_DEEPSEEK_BASE_URL=https://api.deepseek.com
RAG_DEEPSEEK_MODEL=deepseek-chat
RAG_LOG_TO_STDOUT=true
RAG_LOG_TO_FILE=false
RAG_LOG_DIR=logs
RAG_BIZ_API_BASE_URL=http://127.0.0.1:8001
RAG_BIZ_API_TIMEOUT_SECONDS=10
```

### Run APIs
```bash
uvicorn agent_parksuite_biz_api.main:app --reload --port 8001
uvicorn agent_parksuite_rag_core.main:app --reload --port 8002
```

## Database Migration
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

## Init RAG Test Dataset
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

从 `RAG-000` 场景集生成知识分块并写入 `parksuite_rag`：
```bash
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing
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

## Logging And Trace
- Both modules use `Loguru + contextvars`, default output to console.
- Optional file logging:
  - stdout logging toggle: `BIZ_LOG_TO_STDOUT` / `RAG_LOG_TO_STDOUT` (default `true`)
  - enable with `BIZ_LOG_TO_FILE=true` / `RAG_LOG_TO_FILE=true`
  - default directory: `logs/`
  - filename format: `{service_name}.YYYY-MM-DD.log`
- Incoming request headers:
  - `X-Trace-Id` (optional; auto-generated if missing)
- `rag-core` -> `biz-api` httpx calls will propagate `X-Trace-Id` for cross-service tracing.

## RAG-006 Eval (Phase 2)
Data dependency:
- `parksuite_biz_seed` (seeded by `rag000_seed_biz_scenarios.py`)
- `parksuite_rag` (ingested by `rag002_ingest_knowledge.py`)

Recommended reset + init before eval:
```bash
docker exec -it parksuite-pg psql -U postgres -d postgres -c "DROP DATABASE IF EXISTS parksuite_biz_seed;"
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_biz_seed;"
BIZ_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed alembic upgrade head
python scripts/rag000_seed_biz_scenarios.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed \
  --export-jsonl data/rag000/scenarios.jsonl

docker exec -it parksuite-pg psql -U postgres -d postgres -c "DROP DATABASE IF EXISTS parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d parksuite_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing
```

Run offline evaluation replay (requires running `rag-core` API):
```bash
python scripts/rag006_run_eval.py \
  --dataset-path data/rag006/eval_queries.jsonl \
  --report-dir reports \
  --rag-base-url http://127.0.0.1:8002
```

Outputs:
- `reports/rag006_eval_summary.json`
- `reports/rag006_eval_failures.jsonl`

## AI 术语对照（中英 + 发音）
- `Agent`：智能体；发音：`/ˈeɪdʒənt/`
- `LLM`：大语言模型；发音：`L-L-M`
- `Large Language Model`：大语言模型；发音：`/lɑːdʒ ˈlæŋɡwɪdʒ ˈmɒdl/`
- `Retrieve`：召回；发音：`/rɪˈtriːv/`
- `Retrieval`：检索；发音：`/rɪˈtriːvl/`
- `Augmented`：增强；发音：`/ɔːɡˈmentɪd/`
- `Generation`：生成；发音：`/ˌdʒenəˈreɪʃn/`
- `RAG`：检索增强生成；发音：`R-A-G` 或 `/ræɡ/`
- `Vector`：向量；发音：`/ˈvektə(r)/`
- `Embedding`：嵌入/词向量；发音：`/ɪmˈbedɪŋ/`
- `Database`：数据库；发音：`/ˈdeɪtəbeɪs/`
- `Tool Call`：工具调用；发音：`/tuːl kɔːl/`
- `Prompt`：提示词；发音：`/prɒmpt/`
- `Token`：令牌/词元；发音：`/ˈtəʊkən/`
- `Inference`：推理；发音：`/ˈɪnfərəns/`
- `Fine-tune`：微调；发音：`/ˈfaɪn tjuːn/`
- `Workflow`：工作流；发音：`/ˈwɜːkfləʊ/`
- `Planning`：规划；发音：`/ˈplænɪŋ/`
- `Memory`：记忆；发音：`/ˈmeməri/`
- `Reflection`：反思/自省；发音：`/rɪˈflekʃn/`
- `Chunk` / `Chunks`：知识分块；发音：`/tʃʌŋk/`
- `Query`：查询文本；发音：`/ˈkwɪəri/`
- `Query Embedding`：查询向量；发音：`/ˈkwɪəri ɪmˈbedɪŋ/`
- `Vector Similarity Ranking`：向量相似度排序；发音：`/ˈvektə(r) ˌsɪməˈlærəti ˈræŋkɪŋ/`
- `Metadata Filters`：元数据过滤；发音：`/ˈmetədəɪtə ˈfɪltəz/`
- `Top K`：召回条数上限；发音：`/tɒp keɪ/`
- `Source` / `Knowledge Source`：知识来源；发音：`/sɔːs/`，`/ˈnɒlɪdʒ sɔːs/`

## Documentation
- [RAG Ingestion](docs/rag_ingestion.md)
- [PR Acceptance Notes](docs/pr_acceptance.md)
- [Testing](docs/testing.md)
- [RAG-006 Eval Plan](docs/rag006_eval_plan.md)
