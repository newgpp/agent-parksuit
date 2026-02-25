from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "agent-parksuite-biz-api"
    app_env: str = "dev"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/parksuite_biz"
    log_to_stdout: bool = True
    log_to_file: bool = False
    log_dir: str = "logs"

    model_config = SettingsConfigDict(env_prefix="BIZ_", env_file=".env", extra="ignore")


settings = Settings()
