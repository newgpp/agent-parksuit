# Testing

## Quick Run
```bash
pytest
```

## Integration Tests
通用前置：
```bash
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_biz_test;"
docker exec -it parksuite-pg psql -U postgres -d postgres -c "CREATE DATABASE parksuite_rag_test;"
```

Biz API route integration tests need a dedicated test database (default: `parksuite_biz_test`):
```bash
export BIZ_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz_test
pytest tests/biz_api/test_routes_billing_integration.py tests/biz_api/test_routes_orders_integration.py
```

RAG Core route integration tests need a dedicated test database (default: `parksuite_rag_test`):
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_rag_integration.py
```

RAG retrieve API (`POST /api/v1/retrieve`) focused integration tests:
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_retrieve_integration.py
```

RAG answer API (`POST /api/v1/answer`) route tests (LLM mocked):
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_answer_integration.py
```
说明：
- 该用例验证 `POST /api/v1/answer` 的路由编排与返回结构（`conclusion/key_points/citations`）。
- 测试中已对 LLM 调用做 mock，不依赖真实 DeepSeek/OpenAI 网络请求。

Semantic-retrieval validation (paraphrase query -> vector recall, requires OpenAI embedding):
```bash
export OPENAI_API_KEY=your_key
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_retrieve_semantic_integration.py
```

RAG hybrid answer API (`POST /api/v1/answer/hybrid`) integration tests (dataset-driven, tools/LLM mocked):
```bash
export RAG_TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag_test
pytest tests/rag_core/test_routes_hybrid_integration.py
```
说明：
- `test_routes_hybrid_integration.py` 会从 `data/rag000/scenarios.jsonl` 读取场景作为测试输入。
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
pytest tests/biz_api/test_routes_billing_integration.py tests/biz_api/test_routes_orders_integration.py
```

## Manual E2E Tests (Real LLM)
完整步骤与示例请求见：
- [Acceptance Guide](pr_acceptance.md)
