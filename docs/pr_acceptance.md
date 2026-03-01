# PR Acceptance Guide

仅保留 PR 相关验收项，按任务顺序维护。

## RAG-002 入库与检索验收

### 前置条件
```bash
# （可选）重建 parksuite_rag

docker exec -it parksuite-pg psql -U postgres -d postgres -c "DROP DATABASE IF EXISTS parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d parksuite_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 迁移
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
```

### 验收步骤
1. 生成 RAG-000 场景数据
```bash
python scripts/rag000_seed_biz_scenarios.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed \
  --export-jsonl data/rag000/scenarios.jsonl
```

2. 执行 RAG-002 入库
```bash
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing
```

3. SQL 检查
```sql
SELECT COUNT(*) FROM knowledge_sources WHERE source_id LIKE 'RAG000-%';
SELECT COUNT(*) FROM knowledge_chunks kc
JOIN knowledge_sources ks ON ks.id = kc.source_pk
WHERE ks.source_id LIKE 'RAG000-%';
```

4. 检索接口检查
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

### 通过标准
- HTTP 200
- `items` 非空，包含 `source_id/doc_type/content`
- source/chunk 数量符合预期

---

## RAG-010 Step-1 验收：intent_slot_parse（LLM）

目标：仅验证 resolver Step-1 的意图/槽位解析，不进入后续业务链路。

### 前置条件
```bash
export RAG_DEEPSEEK_API_KEY=your_key
export RAG_DEEPSEEK_BASE_URL=https://api.deepseek.com
export RAG_DEEPSEEK_MODEL=deepseek-chat
uvicorn agent_parksuite_rag_core.main:app --reload --port 8002
```

### 调试接口
- `POST /api/v1/debug/intent-slot-parse`

### 用例
```bash
# 1) fee_verify + order_no 提取
curl -X POST "http://127.0.0.1:8002/api/v1/debug/intent-slot-parse" \
  -H "Content-Type: application/json" \
  -d '{"query": "请帮我核验订单 SCN-020 金额是否正确"}'

# 2) arrears_check 缺 plate_no（intent_hint）
curl -X POST "http://127.0.0.1:8002/api/v1/debug/intent-slot-parse" \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我查下有没有欠费", "intent_hint": "arrears_check"}'

# 3) 订单指代歧义
curl -X POST "http://127.0.0.1:8002/api/v1/debug/intent-slot-parse" \
  -H "Content-Type: application/json" \
  -d '{"query": "这笔订单帮我核验下"}'
```

### 通过标准
- HTTP 200
- 返回含 `intent/parsed_payload/trace`
- 能观察到 LLM 输入输出与解析日志：
  - `llm[intent_slot_parse] input ...`
  - `llm[intent_slot_parse] output_preview=...`
  - `llm[intent_slot_parse] output_payload=...`
  - `llm[intent_slot_parse] parse_result=...`

---

## RAG-010 PR-3 验收：react_clarify_gate（短路 + ReAct）

目标：验证 Step-3 澄清决策链路（规则短路 + ReAct），不进入 hybrid 重业务执行。

### 前置条件
- `rag-core` 已启动：`http://127.0.0.1:8002`
- 已配置可用 LLM key
- 多轮场景使用固定 `session_id`

### 调试接口
- `POST /api/v1/debug/clarify-react`

### 用例
```bash
# A. 意图明确 + 缺必填（短路）
curl -X POST "http://127.0.0.1:8002/api/v1/debug/clarify-react" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "RAG010-PR3-DBG-001",
    "query": "这笔订单帮我核验下",
    "intent": "fee_verify",
    "required_slots": ["order_no"]
  }'

# B. 同会话补参后继续
curl -X POST "http://127.0.0.1:8002/api/v1/debug/clarify-react" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "RAG010-PR3-DBG-001",
    "query": "订单号是 SCN-020",
    "intent": "fee_verify",
    "required_slots": ["order_no"]
  }'

# C. 意图不明确触发 ReAct
curl -X POST "http://127.0.0.1:8002/api/v1/debug/clarify-react" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "RAG010-PR3-DBG-003",
    "query": "这单怎么处理",
    "required_slots": []
  }'
```

### 通过标准
- A: `decision=clarify_short_circuit`，`trace` 含 `react_clarify_gate_async:short_circuit:*`
- B: `resolved_slots.order_no=SCN-020` 且 `decision=continue_business`
- C: `decision=clarify_react`（或 `clarify_abort`），`trace` 含 `react_clarify_gate_async:enter_react`
- 进入 ReAct 后单轮最多一次工具调用；若已命中有效工具结果（`hit=true`），应直接收敛为最终 JSON，而不是继续调用第二个工具
- 进入 ReAct 后 `messages` 非空，且同 `session_id` 可连续累积
