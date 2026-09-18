from __future__ import annotations

import asyncio
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


app = FastAPI(title="AISHA Core", version="0.2.0", lifespan=lifespan)
app.include_router(router)


@app.websocket("/v1/ws/{session_id}")
async def conversation_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    orchestrator = websocket.app.state.aisha["orchestrator"]
    send_lock = asyncio.Lock()
    turn_tasks: set[asyncio.Task[None]] = set()

    async def send_json_safely(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def run_turn(text: str) -> None:
        async for event in orchestrator.stream_user_turn(session_id, text):
            async with send_lock:
                await websocket.send_text(event.model_dump_json())

    try:
        while True:
            raw = await websocket.receive_text()
            incoming = json.loads(raw)
            event_type = incoming.get("type")

            if event_type == "aisha.user.text":
                text = str(incoming.get("payload", {}).get("text", "")).strip()
                if not text:
                    continue
                task = asyncio.create_task(run_turn(text))
                turn_tasks.add(task)
                task.add_done_callback(turn_tasks.discard)
                continue

            if event_type == "aisha.turn.cancel":
                requested_turn_id = incoming.get("payload", {}).get("turn_id")
                cancelled_turn_id = await orchestrator.cancel_active_turn(
                    session_id,
                    turn_id=requested_turn_id,
                )
                await send_json_safely(
                    {
                        "type": "aisha.control.cancel_ack",
                        "payload": {"cancelled_turn_id": cancelled_turn_id},
                    }
                )
                continue

            await send_json_safely(
                {"type": "aisha.error", "payload": {"error": "Unsupported event"}}
            )
    except WebSocketDisconnect:
        await orchestrator.cancel_active_turn(session_id)
    finally:
        for task in turn_tasks:
            task.cancel()
        if turn_tasks:
            await asyncio.gather(*turn_tasks, return_exceptions=True)
