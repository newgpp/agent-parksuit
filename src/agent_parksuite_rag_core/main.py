from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent_parksuite_common.observability import TraceContextMiddleware, setup_loguru
from agent_parksuite_rag_core.api.routes import router as rag_router
from agent_parksuite_rag_core.config import settings
from agent_parksuite_rag_core.db.session import init_db


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    setup_loguru(settings.app_name)
    await init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(TraceContextMiddleware)
app.include_router(rag_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
