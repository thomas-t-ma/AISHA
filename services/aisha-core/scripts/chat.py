from __future__ import annotations

import asyncio
import json

import httpx
import websockets

BASE_HTTP = "http://127.0.0.1:8000"
BASE_WS = "ws://127.0.0.1:8000"


async def main() -> None:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_HTTP}/v1/sessions")
        response.raise_for_status()
        session_id = response.json()["session_id"]

    print(f"AISHA session: {session_id}")
    async with websockets.connect(f"{BASE_WS}/v1/ws/{session_id}") as ws:
        while True:
            text = (await asyncio.to_thread(input, "\nYou: ")).strip()
            if text.lower() in {"/quit", "/exit"}:
                break
            await ws.send(json.dumps({"type": "aisha.user.text", "payload": {"text": text}}))
            print("AISHA: ", end="", flush=True)
            while True:
                event = json.loads(await ws.recv())
                if event["type"] == "aisha.assistant.text_delta":
                    print(event["payload"]["text"], end="", flush=True)
                elif event["type"] == "aisha.turn.failed":
                    print(f"\n[turn failed: {event['payload']['error']}]", flush=True)
                    break
                elif event["type"] == "aisha.turn.finished":
                    print()
                    break


if __name__ == "__main__":
    asyncio.run(main())
