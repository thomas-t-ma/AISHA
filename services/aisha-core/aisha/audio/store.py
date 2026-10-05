from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Any

from aisha.audio.base import SpeechArtifact


@dataclass(slots=True)
class StoredAudio:
    artifact: SpeechArtifact
    data: bytes


class EphemeralAudioStore:
    """Bounded in-memory audio store with TTL expiry and no persistence."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 120.0,
        max_items: int = 8,
    ) -> None:
        self.ttl_seconds = max(1.0, ttl_seconds)
        self.max_items = max(1, max_items)
        self._items: OrderedDict[str, tuple[StoredAudio, float]] = OrderedDict()

    def _purge_expired(self) -> None:
        now = monotonic()
        expired = [
            utterance_id
            for utterance_id, (_, expires_at) in self._items.items()
            if expires_at <= now
        ]
        for utterance_id in expired:
            self._items.pop(utterance_id, None)

    def put(self, artifact: SpeechArtifact, data: bytes) -> None:
        self._purge_expired()
        self._items[artifact.utterance_id] = (
            StoredAudio(artifact=artifact.model_copy(deep=True), data=bytes(data)),
            monotonic() + self.ttl_seconds,
        )
        self._items.move_to_end(artifact.utterance_id)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def get(self, utterance_id: str) -> StoredAudio | None:
        self._purge_expired()
        item = self._items.get(utterance_id)
        if item is None:
            return None
        stored, _ = item
        return StoredAudio(
            artifact=stored.artifact.model_copy(deep=True),
            data=stored.data,
        )

    def clear(self) -> None:
        self._items.clear()

    def status(self) -> dict[str, Any]:
        self._purge_expired()
        return {
            "artifact_count": len(self._items),
            "ttl_seconds": self.ttl_seconds,
            "max_items": self.max_items,
            "persisted": False,
        }
