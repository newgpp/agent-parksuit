# Acceptance Guide

用于 PR 验收与真实链路联调，包含自动验收（入库与检索）和手工 E2E（真实 LLM + 真实服务调用）。

## PR Status Note
- `RAG-007` is currently marked as `Paused`.
- Current baseline keeps `RAG-005` orchestration with branch-level `graph_trace`, which is sufficient for present acceptance scope.
- `RAG-008` is currently marked as `Paused`.
- `RAG-009` starts with dataset-first delivery (`data/rag009/memory_acceptance_cases.jsonl`) for short-term memory acceptance.

## 1. 数据与入库验收（RAG-002）
1. （可选）重建 `parksuite_rag`（当迁移状态异常或历史脏数据较多时）
```bash
docker exec -it parksuite-pg psql -U postgres -d postgres -c "DROP DATABASE IF EXISTS parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_rag;"
docker exec -it parksuite-pg psql -U postgres -d parksuite_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

2. 初始化 `parksuite_rag` 表结构
```bash
RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag alembic upgrade head
```
说明：迁移脚本会按当前连接数据库名判断执行目标，避免误跳过 `parksuite_rag`。

3. 生成 `RAG-000` 场景数据（写入 `parksuite_biz_seed` 并导出 JSONL）
```bash
python scripts/rag000_seed_biz_scenarios.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_seed \
  --export-jsonl data/rag000/scenarios.jsonl
```

4. 执行 `RAG-002` 入库
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

6. 检索接口验收（`/api/v1/retrieve`）
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
预期：`items` 非空，且包含 `source_id/doc_type/content`。

## 2. 手工 E2E：`/api/v1/answer`（真实 LLM）
### 前置条件
```bash
# 准备 RAG 数据
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing

# 配置 LLM（DeepSeek）
export RAG_DEEPSEEK_API_KEY=your_key
export RAG_DEEPSEEK_BASE_URL=https://api.deepseek.com
export RAG_DEEPSEEK_MODEL=deepseek-chat
```

### 启动服务
```bash
uvicorn agent_parksuite_rag_core.main:app --reload --port 8002
```

### 测试用例
```bash
# 用例1：同城不同停车场差异（只传C场，预期可能提示证据不足）
curl -X POST "http://127.0.0.1:8002/api/v1/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "同城不同停车场为什么收费不同（C场）？",
    "top_k": 5,
    "doc_type": "rule_explain",
    "source_type": "biz_derived",
    "city_code": "310100",
    "lot_code": "SCN-LOT-C",
    "at_time": "2026-02-10T10:00:00+08:00"
  }'

# 用例2：同城不同停车场差异（A场+C场对比）
curl -X POST "http://127.0.0.1:8002/api/v1/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "同城不同停车场为什么收费不同（A场和C场）？请对比说明",
    "top_k": 8,
    "doc_type": "rule_explain",
    "source_type": "biz_derived",
    "city_code": "310100",
    "at_time": "2026-02-10T10:00:00+08:00"
  }'

# 用例3：首30分钟免费边界（31分钟）
curl -X POST "http://127.0.0.1:8002/api/v1/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "首30分钟免费，31分钟要收多少？",
    "top_k": 5,
    "doc_type": "rule_explain",
    "source_type": "biz_derived",
    "city_code": "310100",
    "lot_code": "SCN-LOT-C",
    "at_time": "2026-02-01T08:31:00+08:00"
  }'

# 用例4：规则版本切换（2月15号）
curl -X POST "http://127.0.0.1:8002/api/v1/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "这个停车场2月15号怎么收费？",
    "top_k": 5,
    "doc_type": "rule_explain",
    "source_type": "biz_derived",
    "city_code": "310100",
    "lot_code": "SCN-LOT-E",
    "at_time": "2026-02-15T10:00:00+08:00"
  }'
```

### 验收检查点
- HTTP 200
- 返回包含 `conclusion`、`key_points`、`citations`
- `retrieved_count > 0`
- `citations` 含 `source_id/chunk_id/doc_type/snippet`

## 3. 手工 E2E：`/api/v1/answer/hybrid`（真实 LLM + 真实 biz-api）
### 前置条件
- `biz-api` 已启动（默认 `http://127.0.0.1:8001`）
- `rag-core` 已启动（`http://127.0.0.1:8002`）
- `RAG_BIZ_API_BASE_URL` 指向可访问的 biz-api

### 启动服务
```bash
# 终端1：启动 biz-api
uvicorn agent_parksuite_biz_api.main:app --reload --port 8001

# 终端2：启动 rag-core
uvicorn agent_parksuite_rag_core.main:app --reload --port 8002
```

### 测试用例
```bash
# 用例1：规则解释分支（rule_explain）
curl -X POST "http://127.0.0.1:8002/api/v1/answer/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "这个停车场2月15号怎么收费？",
    "intent_hint": "rule_explain",
    "top_k": 5,
    "doc_type": "rule_explain",
    "source_type": "biz_derived",
    "city_code": "310100",
    "lot_code": "SCN-LOT-E",
    "at_time": "2026-02-15T10:00:00+08:00"
  }'

# 用例2：欠费查询分支（arrears_check）
curl -X POST "http://127.0.0.1:8002/api/v1/answer/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "帮我查下车牌沪SCN001有没有欠费",
    "intent_hint": "arrears_check",
    "top_k": 3,
    "doc_type": "faq",
    "source_type": "biz_derived",
    "city_code": "310100",
    "lot_code": "SCN-LOT-A",
    "plate_no": "沪SCN001"
  }'

# 用例3：历史订单计费核验分支（fee_verify）
curl -X POST "http://127.0.0.1:8002/api/v1/answer/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "这个订单金额为什么不一致，帮我核验下",
    "intent_hint": "fee_verify",
    "top_k": 3,
    "doc_type": "rule_explain",
    "source_type": "biz_derived",
    "city_code": "310100",
    "lot_code": "SCN-LOT-A",
    "order_no": "SCN-020"
  }'
```

### 验收检查点
- HTTP 200
- 返回包含 `intent`、`business_facts`、`conclusion`、`key_points`、`citations`、`graph_trace`
- `graph_trace` 含 `intent_classifier:*` 且命中目标分支（`rule_explain_flow` / `arrears_check_flow` / `fee_verify_flow`）
- `arrears_check` 时 `business_facts.arrears_count` 有值
- `fee_verify` 时 `business_facts.amount_check_result/amount_check_action` 有值

## 4. API 示例
### Billing Rule Payload Example
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
        "time_window": {"start": "08:00", "end": "22:00", "timezone": "Asia/Shanghai"},
        "unit_minutes": 30,
        "unit_price": 2,
        "free_minutes": 30,
        "max_charge": 30
      },
      {
        "name": "night_free",
        "type": "free",
        "time_window": {"start": "22:00", "end": "08:00", "timezone": "Asia/Shanghai"}
      }
    ]
  }
}
```
说明：`time_window.timezone` 可按规则分段单独配置，默认值为 `Asia/Shanghai`。
