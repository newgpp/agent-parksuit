# Manual E2E Tests (Real LLM)

用于真实链路联调（入库数据 + 启动服务 + 连接 DeepSeek + 调 `/api/v1/*`），区别于 `pytest` 自动化测试。

## `/api/v1/answer`（真实 LLM）

### 前置条件
```bash
# 准备 RAG 数据
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing

# 配置 LLM 环境变量（DeepSeek）
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
- 返回字段包含 `conclusion`、`key_points`、`citations`
- `retrieved_count > 0`
- `citations` 中可看到 `source_id/chunk_id/doc_type/snippet`

## `/api/v1/answer/hybrid`（真实 LLM + 真实 biz-api）

### 前置条件
- `biz-api` 已启动（默认 `http://127.0.0.1:8001`）
- `rag-core` 已启动（`http://127.0.0.1:8002`）
- `.env` 或环境变量中 `RAG_BIZ_API_BASE_URL` 指向可访问的 biz-api

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
- 返回字段包含 `intent`、`business_facts`、`conclusion`、`key_points`、`citations`、`graph_trace`
- `graph_trace` 含 `intent_classifier:*` 且命中目标分支（`rule_explain_flow` / `arrears_check_flow` / `fee_verify_flow`）
- `arrears_check` 时 `business_facts.arrears_count` 有值
- `fee_verify` 时 `business_facts.amount_check_result/amount_check_action` 有值
