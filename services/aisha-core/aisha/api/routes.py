from __future__ import annotations

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/v1")


class SessionResponse(BaseModel):
    session_id: str


class CancelResponse(BaseModel):
    cancelled_turn_id: str | None


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
        "warmup": state.get("warmup_metrics", {}),
    }


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: Request):
    session_id = await request.app.state.aisha["store"].create_session()
    return SessionResponse(session_id=session_id)


@router.post("/sessions/{session_id}/cancel", response_model=CancelResponse)
async def cancel_turn(session_id: str, request: Request, turn_id: str | None = None):
    cancelled = await request.app.state.aisha["orchestrator"].cancel_active_turn(
        session_id,
        turn_id=turn_id,
    )
    return CancelResponse(cancelled_turn_id=cancelled)


@router.get("/sessions/{session_id}/messages")
async def session_messages(
    session_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=500),
):
    return await request.app.state.aisha["store"].session_messages(session_id, limit=limit)


@router.get("/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=2000),
):
    return await request.app.state.aisha["store"].session_events(session_id, limit=limit)


@router.get("/sessions/{session_id}/model-runs")
async def session_model_runs(
    session_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
):
    return await request.app.state.aisha["store"].session_model_runs(session_id, limit=limit)
