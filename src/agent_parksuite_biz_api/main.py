from fastapi import FastAPI

from agent_parksuite_biz_api.api.routes import router as biz_router
from agent_parksuite_biz_api.config import settings


app = FastAPI(title=settings.app_name)
app.include_router(biz_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
