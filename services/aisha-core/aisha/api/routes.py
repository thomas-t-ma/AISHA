from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/v1")


class SessionResponse(BaseModel):
    session_id: str


@router.get("/health")
async def health(request: Request):
    state = request.app.state.aisha
    return {
        "status": "ok",
        "profile": state["profile"].name,
        "provider": state["provider"].name,
        "model": state["provider"].model,
        "persona_version": state["persona"].version,
        "data_dir": str(state["settings"].data_dir),
    }


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: Request):
    session_id = await request.app.state.aisha["store"].create_session()
    return SessionResponse(session_id=session_id)
