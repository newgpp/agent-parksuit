from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from agent_parksuite_rag_core.api.routes import router as rag_router
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.db.session import init_db


def _setup_app_logging() -> None:
    # Ensure package-level INFO logs are visible regardless of uvicorn logger wiring.
    app_logger = logging.getLogger("agent_parksuite_rag_core")
    app_logger.setLevel(logging.INFO)
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        app_logger.addHandler(handler)
    app_logger.propagate = False


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _setup_app_logging()
    await init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(rag_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
