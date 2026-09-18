from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import perf_counter

from aisha.character.loader import PersonaPackage
from aisha.contracts.events import AISHAEvent
from aisha.contracts.turns import Message, TurnContext
from aisha.providers.base import AISHAProviderError
from aisha.storage.database import AISHAStore


class AISHAOrchestrator:
    def __init__(self, store: AISHAStore, persona: PersonaPackage, llm_provider) -> None:
        self.store = store
        self.persona = persona
        self.llm_provider = llm_provider
        self._active_turns: dict[str, tuple[str, asyncio.Event]] = {}
        self._active_turns_lock = asyncio.Lock()

    async def _persisted_event(
        self,
        *,
        session_id: str,
        turn_id: str,
        type_: str,
        payload: dict,
    ) -> AISHAEvent:
        event = AISHAEvent(
            session_id=session_id,
            turn_id=turn_id,
            source="aisha.cognition",
            type=type_,
            payload=payload,
        )
        await self.store.add_event(event)
        return event

    async def _register_turn(self, session_id: str, turn_id: str) -> asyncio.Event:
        cancellation = asyncio.Event()
        async with self._active_turns_lock:
            previous = self._active_turns.get(session_id)
            if previous is not None:
                previous[1].set()
            self._active_turns[session_id] = (turn_id, cancellation)
        return cancellation

    async def _clear_turn(self, session_id: str, turn_id: str) -> None:
        async with self._active_turns_lock:
            active = self._active_turns.get(session_id)
            if active is not None and active[0] == turn_id:
                self._active_turns.pop(session_id, None)

    async def cancel_active_turn(self, session_id: str, turn_id: str | None = None) -> str | None:
        async with self._active_turns_lock:
            active = self._active_turns.get(session_id)
            if active is None:
                return None
            active_turn_id, cancellation = active
            if turn_id is not None and turn_id != active_turn_id:
                return None
            cancellation.set()
            return active_turn_id

    async def stream_user_turn(self, session_id: str, text: str) -> AsyncIterator[AISHAEvent]:
        await self.store.ensure_session(session_id)
        history = await self.store.recent_messages(session_id)

        user_message = Message(role="user", text=text)
        await self.store.add_message(session_id, user_message)

        context = TurnContext(
            session_id=session_id,
            persona_version=self.persona.version,
            system_prompt=self.persona.prompt,
            messages=history,
            user_input=text,
        )
        cancellation = await self._register_turn(session_id, context.turn_id)
        run_id = await self.store.start_model_run(
            session_id=session_id,
            turn_id=context.turn_id,
            provider=self.llm_provider.name,
            model=self.llm_provider.model,
        )

        started = perf_counter()
        first_token_ms: float | None = None
        chunks: list[str] = []

        started_event = await self._persisted_event(
            session_id=session_id,
            turn_id=context.turn_id,
            type_="aisha.turn.started",
            payload={
                "run_id": run_id,
                "provider": self.llm_provider.name,
                "model": self.llm_provider.model,
                "persona_version": self.persona.version,
            },
        )
        yield started_event

        try:
            async for delta in self.llm_provider.stream_turn(context):
                if cancellation.is_set():
                    total_ms = (perf_counter() - started) * 1000
                    await self.store.finish_model_run(
                        run_id,
                        status="cancelled",
                        first_token_ms=first_token_ms,
                        total_ms=total_ms,
                        output_chars=sum(len(chunk) for chunk in chunks),
                    )
                    cancelled_event = await self._persisted_event(
                        session_id=session_id,
                        turn_id=context.turn_id,
                        type_="aisha.turn.cancelled",
                        payload={
                            "run_id": run_id,
                            "status": "cancelled",
                            "output_chars": sum(len(chunk) for chunk in chunks),
                            "total_ms": round(total_ms, 3),
                        },
                    )
                    yield cancelled_event
                    return

                if first_token_ms is None:
                    first_token_ms = (perf_counter() - started) * 1000

                chunks.append(delta)
                delta_event = await self._persisted_event(
                    session_id=session_id,
                    turn_id=context.turn_id,
                    type_="aisha.assistant.text_delta",
                    payload={"run_id": run_id, "text": delta},
                )
                yield delta_event

            if cancellation.is_set():
                total_ms = (perf_counter() - started) * 1000
                await self.store.finish_model_run(
                    run_id,
                    status="cancelled",
                    first_token_ms=first_token_ms,
                    total_ms=total_ms,
                    output_chars=sum(len(chunk) for chunk in chunks),
                )
                cancelled_event = await self._persisted_event(
                    session_id=session_id,
                    turn_id=context.turn_id,
                    type_="aisha.turn.cancelled",
                    payload={
                        "run_id": run_id,
                        "status": "cancelled",
                        "output_chars": sum(len(chunk) for chunk in chunks),
                        "total_ms": round(total_ms, 3),
                    },
                )
                yield cancelled_event
                return
        except AISHAProviderError as exc:
            total_ms = (perf_counter() - started) * 1000
            await self.store.finish_model_run(
                run_id,
                status="failed",
                first_token_ms=first_token_ms,
                total_ms=total_ms,
                output_chars=sum(len(chunk) for chunk in chunks),
                error=str(exc),
            )
            failed_event = await self._persisted_event(
                session_id=session_id,
                turn_id=context.turn_id,
                type_="aisha.turn.failed",
                payload={"run_id": run_id, "error": str(exc), "total_ms": round(total_ms, 3)},
            )
            yield failed_event
            return
        finally:
            await self._clear_turn(session_id, context.turn_id)

        final_text = "".join(chunks).strip()
        assistant_message = Message(role="assistant", text=final_text)
        await self.store.add_message(session_id, assistant_message)

        total_ms = (perf_counter() - started) * 1000
        await self.store.finish_model_run(
            run_id,
            status="completed",
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            output_chars=len(final_text),
        )
        finished_event = await self._persisted_event(
            session_id=session_id,
            turn_id=context.turn_id,
            type_="aisha.turn.finished",
            payload={
                "run_id": run_id,
                "status": "completed",
                "message_id": assistant_message.message_id,
                "first_token_ms": None if first_token_ms is None else round(first_token_ms, 3),
                "total_ms": round(total_ms, 3),
                "output_chars": len(final_text),
            },
        )
        yield finished_event
