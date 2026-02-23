from __future__ import annotations

from fastapi import APIRouter

from agent_parksuite_rag_core.schemas.rag import RetrieveRequest, RetrieveResponse

router = APIRouter(prefix="/api/v1", tags=["rag-core"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    # Phase 1 placeholder: return empty set until embedding + retrieval pipeline is wired.
    _ = payload
    return RetrieveResponse(items=[])
