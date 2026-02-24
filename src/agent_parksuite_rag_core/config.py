from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "agent-parksuite-rag-core"
    app_env: str = "dev"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag"
    embedding_dim: int = 1536
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_log_full_payload: bool = False
    llm_log_max_chars: int = 1000
    biz_api_base_url: str = "http://127.0.0.1:8001"
    biz_api_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")


settings = Settings()
