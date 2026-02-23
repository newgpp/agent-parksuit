from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "agent-parksuite-rag-core"
    app_env: str = "dev"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_rag"
    embedding_dim: int = 1536

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")


settings = Settings()
