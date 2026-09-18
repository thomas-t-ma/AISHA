from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from aisha.api.routes import router
from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.providers.registry import build_llm_provider
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    profile = settings.load_profile()
    persona = load_persona(settings.character_dir)
    store = AISHAStore(settings.data_dir / "database" / "aisha.sqlite3")
    await store.initialize()
    provider = build_llm_provider(settings, profile)
    orchestrator = AISHAOrchestrator(store, persona, provider)

    app.state.aisha = {
        "settings": settings,
        "profile": profile,
        "persona": persona,
        "store": store,
        "provider": provider,
        "orchestrator": orchestrator,
    }
    yield


app = FastAPI(title="AISHA Core", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.websocket("/v1/ws/{session_id}")
async def conversation_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    orchestrator = websocket.app.state.aisha["orchestrator"]
    try:
        while True:
            raw = await websocket.receive_text()
            incoming = json.loads(raw)
            if incoming.get("type") != "aisha.user.text":
                await websocket.send_json({"type": "aisha.error", "payload": {"error": "Unsupported event"}})
                continue
            text = str(incoming.get("payload", {}).get("text", "")).strip()
            if not text:
                continue
            async for event in orchestrator.stream_user_turn(session_id, text):
                await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        return
