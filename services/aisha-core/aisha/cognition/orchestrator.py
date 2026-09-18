from __future__ import annotations

from collections.abc import AsyncIterator

from aisha.character.loader import PersonaPackage
from aisha.contracts.events import AISHAEvent
from aisha.contracts.turns import Message, TurnContext
from aisha.storage.database import AISHAStore


class AISHAOrchestrator:
    def __init__(self, store: AISHAStore, persona: PersonaPackage, llm_provider) -> None:
        self.store = store
        self.persona = persona
        self.llm_provider = llm_provider

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

        yield AISHAEvent(
            session_id=session_id,
            turn_id=context.turn_id,
            source="aisha.cognition",
            type="aisha.turn.started",
            payload={
                "provider": self.llm_provider.name,
                "model": self.llm_provider.model,
                "persona_version": self.persona.version,
            },
        )

        chunks: list[str] = []
        try:
            async for delta in self.llm_provider.stream_turn(context):
                chunks.append(delta)
                yield AISHAEvent(
                    session_id=session_id,
                    turn_id=context.turn_id,
                    source="aisha.cognition",
                    type="aisha.assistant.text_delta",
                    payload={"text": delta},
                )
        except Exception as exc:
            yield AISHAEvent(
                session_id=session_id,
                turn_id=context.turn_id,
                source="aisha.cognition",
                type="aisha.turn.failed",
                payload={"error": str(exc)},
            )
            return

        final_text = "".join(chunks).strip()
        assistant_message = Message(role="assistant", text=final_text)
        await self.store.add_message(session_id, assistant_message)

        yield AISHAEvent(
            session_id=session_id,
            turn_id=context.turn_id,
            source="aisha.cognition",
            type="aisha.turn.finished",
            payload={"status": "completed", "message_id": assistant_message.message_id},
        )
