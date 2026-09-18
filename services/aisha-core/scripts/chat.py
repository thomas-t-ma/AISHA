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
    print("Commands: /cancel, /events, /runs, /quit")

    async with websockets.connect(f"{BASE_WS}/v1/ws/{session_id}") as ws:
        while True:
            text = (await asyncio.to_thread(input, "\nYou: ")).strip()
            lowered = text.lower()

            if lowered in {"/quit", "/exit"}:
                break

            if lowered == "/events":
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{BASE_HTTP}/v1/sessions/{session_id}/events")
                    response.raise_for_status()
                    for event in response.json():
                        print(f"{event['type']}: {event['payload']}")
                continue

            if lowered == "/runs":
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{BASE_HTTP}/v1/sessions/{session_id}/model-runs"
                    )
                    response.raise_for_status()
                    for run in response.json():
                        print(run)
                continue

            if lowered == "/cancel":
                await ws.send(json.dumps({"type": "aisha.turn.cancel", "payload": {}}))
                acknowledgement = json.loads(await ws.recv())
                print(f"[cancel requested: {acknowledgement['payload']['cancelled_turn_id']}]")
                continue

            await ws.send(json.dumps({"type": "aisha.user.text", "payload": {"text": text}}))
            print("AISHA: ", end="", flush=True)
            while True:
                event = json.loads(await ws.recv())
                if event["type"] == "aisha.assistant.text_delta":
                    print(event["payload"]["text"], end="", flush=True)
                elif event["type"] == "aisha.turn.failed":
                    print(f"\n[turn failed: {event['payload']['error']}]", flush=True)
                    break
                elif event["type"] == "aisha.turn.cancelled":
                    print("\n[turn cancelled]", flush=True)
                    break
                elif event["type"] == "aisha.turn.finished":
                    print(
                        f"\n[ttft={event['payload']['first_token_ms']} ms, "
                        f"total={event['payload']['total_ms']} ms]"
                    )
                    break
                elif event["type"] == "aisha.control.cancel_ack":
                    print(
                        f"\n[cancel requested: "
                        f"{event['payload']['cancelled_turn_id']}]"
                    )


if __name__ == "__main__":
    asyncio.run(main())
