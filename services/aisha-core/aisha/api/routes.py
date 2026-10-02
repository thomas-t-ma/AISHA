from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from aisha.embodiment.state import AffectIntent

router = APIRouter(prefix="/v1")

LOCAL_STUDIO_ORIGINS = {
    "http://127.0.0.1:5173", "http://localhost:5173",
    "http://127.0.0.1:8000", "http://localhost:8000",
}


def _check_local_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin not in LOCAL_STUDIO_ORIGINS:
        raise HTTPException(status_code=403, detail="Unrecognized local client origin")


class MemoryWrite(BaseModel):
    text: str = Field(min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def strip_nonblank(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("Memory cannot be blank")
        return clean


class MemoryCreate(MemoryWrite):
    source_session_id: str | None = Field(default=None, max_length=96)



class SessionResponse(BaseModel):
    session_id: str


class SessionSettingsResponse(BaseModel):
    session_id: str
    memory_mode: Literal["normal", "test"]


class SessionSettingsWrite(BaseModel):
    memory_mode: Literal["normal", "test"]


class CancelResponse(BaseModel):
    cancelled_turn_id: str | None


class CameraStateWrite(BaseModel):
    enabled: bool


class EmbodimentAffectWrite(BaseModel):
    affect: AffectIntent
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)


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
        "embodiment": state["orchestrator"].embodiment_status(),
        "perception": {
            **state["perception_hub"].status(),
            "camera": state["camera_controller"].status(),
            "analyzer": state["perception_runtime"].analyzer.status(),
            "analysis_steps": state["perception_runtime"].status()["analysis_steps"],
        },
        "memory_integrity": state.get("memory_integrity_startup", {}),
        "auto_memory": state["orchestrator"].memory_status(),
    }


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: Request):
    session_id = await request.app.state.aisha["store"].create_session()
    return SessionResponse(session_id=session_id)


@router.get("/sessions/{session_id}", response_model=SessionSettingsResponse)
async def get_session_settings(session_id: str, request: Request):
    session = await request.app.state.aisha["store"].session_info(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSettingsResponse(
        session_id=session["session_id"], memory_mode=session["memory_mode"]
    )


@router.patch("/sessions/{session_id}", response_model=SessionSettingsResponse)
async def update_session_settings(
    session_id: str,
    settings: SessionSettingsWrite,
    request: Request,
):
    _check_local_origin(request)
    session = await request.app.state.aisha["store"].set_session_memory_mode(
        session_id, settings.memory_mode
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSettingsResponse(
        session_id=session["session_id"], memory_mode=session["memory_mode"]
    )


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


@router.get("/memories")
async def list_memories(request: Request, limit: int = Query(default=250, ge=1, le=500)):
    return await request.app.state.aisha["store"].list_memories(limit=limit)


@router.post("/memories", status_code=201)
async def create_memory(request: Request, memory: MemoryCreate):
    _check_local_origin(request)
    return await request.app.state.aisha["store"].create_memory(
        memory.text, source_session_id=memory.source_session_id
    )


@router.patch("/memories/{memory_id}")
async def update_memory(request: Request, memory_id: str, memory: MemoryWrite):
    _check_local_origin(request)
    updated = await request.app.state.aisha["store"].update_memory(memory_id, memory.text)
    if updated is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return updated


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(request: Request, memory_id: str):
    _check_local_origin(request)
    deleted = await request.app.state.aisha["store"].delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)


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

@router.get("/embodiment/state")
async def embodiment_state(request: Request):
    return request.app.state.aisha["orchestrator"].embodiment_status()


@router.post("/embodiment/affect")
async def set_embodiment_affect(
    request: Request,
    update: EmbodimentAffectWrite,
):
    _check_local_origin(request)
    return request.app.state.aisha["orchestrator"].set_embodiment_affect(
        update.affect,
        intensity=update.intensity,
    )


@router.get("/perception/status")
async def perception_status(request: Request):
    return request.app.state.aisha["perception_hub"].status()


@router.get("/perception/latest")
async def perception_latest(request: Request):
    frame = request.app.state.aisha["perception_hub"].latest()
    return None if frame is None else frame.model_dump(mode="json")


@router.get("/perception/summary")
async def perception_summary(request: Request):
    return request.app.state.aisha["perception_hub"].summary().model_dump(mode="json")


@router.get("/perception/runtime")
async def perception_runtime_status(request: Request):
    return request.app.state.aisha["perception_runtime"].status()


@router.get("/perception/camera")
async def perception_camera_status(request: Request):
    return request.app.state.aisha["camera_controller"].status()


@router.post("/perception/camera")
async def set_perception_camera(
    request: Request,
    update: CameraStateWrite,
):
    _check_local_origin(request)
    controller = request.app.state.aisha["camera_controller"]
    return controller.enable() if update.enabled else controller.disable()


@router.get("/memory/status")
async def automatic_memory_status(request: Request):
    return request.app.state.aisha["orchestrator"].memory_status()


@router.get("/memory/integrity")
async def memory_integrity(request: Request):
    return await request.app.state.aisha["ledger"].audit_integrity()


@router.get("/memory/episodes")
async def memory_episodes(request: Request, limit: int = Query(default=30, ge=1, le=100)):
    return await request.app.state.aisha["ledger"].list_episodes(limit=limit)


@router.get("/memory/beliefs")
async def memory_beliefs(request: Request, limit: int = Query(default=30, ge=1, le=200)):
    return await request.app.state.aisha["ledger"].list_beliefs(limit=limit)


@router.get("/memory/beliefs/{belief_id}/versions")
async def memory_belief_versions(request: Request, belief_id: str):
    return await request.app.state.aisha["ledger"].versions(belief_id)


@router.delete("/memory/beliefs/{belief_id}", status_code=204)
async def forget_automatic_belief(request: Request, belief_id: str):
    _check_local_origin(request)
    removed = await request.app.state.aisha["ledger"].forget_belief(belief_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Belief not found")
    return Response(status_code=204)
