# RAG Ingestion

## RAG-002 Ingestion Pipeline
从 `RAG-000` 场景集生成知识分块并写入 `parksuite_rag`：
```bash
python scripts/rag002_ingest_knowledge.py \
  --database-url postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag \
  --input-type scenarios_jsonl \
  --input-path data/rag000/scenarios.jsonl \
  --replace-existing
```

RAG-002 初始化与验收详情见：
- [PR Acceptance Notes](pr_acceptance.md)

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
