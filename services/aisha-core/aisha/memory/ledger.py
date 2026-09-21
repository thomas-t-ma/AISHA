from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from aisha.storage.database import AISHAStore

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS auto_episodes (
    episode_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL UNIQUE,
    user_message_id TEXT NOT NULL,
    assistant_message_id TEXT NOT NULL,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auto_episodes_created
ON auto_episodes(created_at DESC);

CREATE TABLE IF NOT EXISTS auto_beliefs (
    belief_id TEXT PRIMARY KEY,
    topic_key TEXT NOT NULL UNIQUE,
    text TEXT NOT NULL,
    epistemic_status TEXT NOT NULL,
    evidence_status TEXT NOT NULL DEFAULT 'legacy_unchecked',
    source_quote TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_turn_id TEXT NOT NULL,
    source_episode_id TEXT NOT NULL,
    open_question TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_episode_id) REFERENCES auto_episodes(episode_id)
);
CREATE INDEX IF NOT EXISTS idx_auto_beliefs_updated
ON auto_beliefs(updated_at DESC);

CREATE TABLE IF NOT EXISTS auto_belief_versions (
    version_id TEXT PRIMARY KEY,
    belief_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    text TEXT NOT NULL,
    epistemic_status TEXT NOT NULL,
    evidence_status TEXT NOT NULL DEFAULT 'legacy_unchecked',
    source_quote TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_turn_id TEXT NOT NULL,
    source_episode_id TEXT NOT NULL,
    open_question TEXT,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY(belief_id) REFERENCES auto_beliefs(belief_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_belief_versions_id
ON auto_belief_versions(belief_id, revision);
"""

ALLOWED_STATUSES = {"stated", "inferred", "uncertain"}


class ExperienceLedger:
    def __init__(self, store: AISHAStore) -> None:
        self.store = store

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with self.store._connect() as db:
            db.executescript(MEMORY_SCHEMA)
            for table in ("auto_beliefs", "auto_belief_versions"):
                columns = {row["name"] for row in db.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()}
                if "evidence_status" not in columns:
                    db.execute(
                        f"ALTER TABLE {table} ADD COLUMN evidence_status "
                        "TEXT NOT NULL DEFAULT 'legacy_unchecked'"
                    )

    async def record_episode(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message_id: str,
        assistant_message_id: str,
        user_text: str,
        assistant_text: str,
    ) -> dict:
        episode = {
            "episode_id": f"episode_{uuid4().hex}",
            "session_id": session_id,
            "turn_id": turn_id,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "created_at": datetime.now(UTC).isoformat(),
        }

        def work() -> dict:
            with self.store._connect() as db:
                db.execute(
                    """INSERT OR IGNORE INTO auto_episodes (
                    episode_id, session_id, turn_id, user_message_id, assistant_message_id,
                    user_text, assistant_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    tuple(episode.values()),
                )
                row = db.execute(
                    "SELECT * FROM auto_episodes WHERE turn_id = ?", (turn_id,)
                ).fetchone()
                return dict(row)

        return await asyncio.to_thread(work)

    async def list_episodes(self, limit: int = 30) -> list[dict]:
        safe_limit = max(1, min(limit, 100))

        def work() -> list[dict]:
            with self.store._connect() as db:
                rows = db.execute(
                    "SELECT * FROM auto_episodes ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await asyncio.to_thread(work)

    async def list_beliefs(self, limit: int = 30) -> list[dict]:
        safe_limit = max(1, min(limit, 200))

        def work() -> list[dict]:
            with self.store._connect() as db:
                rows = db.execute(
                    "SELECT * FROM auto_beliefs ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                    (safe_limit,),
                ).fetchall()
                return [dict(row) for row in rows]

        return await asyncio.to_thread(work)

    async def versions(self, belief_id: str) -> list[dict]:
        def work() -> list[dict]:
            with self.store._connect() as db:
                rows = db.execute(
                    """SELECT * FROM auto_belief_versions
                    WHERE belief_id = ? ORDER BY revision ASC""", (belief_id,)
                ).fetchall()
                return [dict(row) for row in rows]

        return await asyncio.to_thread(work)

    async def forget_belief(self, belief_id: str) -> bool:
        """Delete this derived belief and its interpretations; original chats remain."""
        def work() -> bool:
            with self.store._connect() as db:
                db.execute(
                    "DELETE FROM auto_belief_versions WHERE belief_id = ?", (belief_id,)
                )
                return bool(db.execute(
                    "DELETE FROM auto_beliefs WHERE belief_id = ?", (belief_id,)
                ).rowcount)

        return await asyncio.to_thread(work)

    async def apply(
        self,
        *,
        episode: dict,
        action: dict,
    ) -> dict | None:
        """Compatibility wrapper for callers that only need the saved belief."""
        saved, _reason = await self.apply_with_reason(episode=episode, action=action)
        return saved

    async def apply_with_reason(
        self,
        *,
        episode: dict,
        action: dict,
        evidence_checked: bool = False,
    ) -> tuple[dict | None, str]:
        """Atomically apply or return a stable, non-sensitive rejection reason."""
        quote = action.get("source_quote", "")
        text = action.get("text", "")
        topic_key = action.get("topic_key", "")
        status = action.get("epistemic_status")
        question = action.get("open_question")
        operation = action.get("action")
        target = action.get("target_belief_id")

        if not isinstance(quote, str) or not quote.strip():
            return None, "empty_source_quote"
        if quote not in episode["user_text"]:
            return None, "quote_not_in_user_message"
        if not isinstance(text, str) or not (1 <= len(text.strip()) <= 300):
            return None, "invalid_memory_text"
        if not isinstance(topic_key, str) or not (1 <= len(topic_key.strip()) <= 80):
            return None, "invalid_topic_key"
        if status not in ALLOWED_STATUSES:
            return None, "invalid_epistemic_status"
        if operation not in {"add", "revise"}:
            return None, "invalid_action"
        if question is not None and (
            not isinstance(question, str) or len(question) > 200
        ):
            return None, "invalid_open_question"

        # A revision must identify an existing belief. "Add" cannot overwrite a
        # belief just because the model guessed the same topic key.
        if operation == "revise" and (not isinstance(target, str) or not target):
            return None, "missing_revision_target"

        now = datetime.now(UTC).isoformat()
        clean_key = topic_key.strip().lower()
        question = question.strip() if question else None
        evidence_status = "verified" if evidence_checked else "legacy_unchecked"

        def work() -> tuple[dict | None, str]:
            with self.store._connect() as db:
                if operation == "add":
                    if db.execute(
                        "SELECT 1 FROM auto_beliefs WHERE topic_key = ?", (clean_key,)
                    ).fetchone():
                        return None, "topic_already_exists_use_revision"
                    belief_id = f"belief_{uuid4().hex}"
                    revision = 1
                    db.execute(
                        """INSERT INTO auto_beliefs (
                        belief_id, topic_key, text, epistemic_status, evidence_status,
                        source_quote, source_session_id, source_turn_id, source_episode_id,
                        open_question, revision, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            belief_id, clean_key, text.strip(), status, evidence_status, quote,
                            episode["session_id"], episode["turn_id"],
                            episode["episode_id"], question, revision, now, now,
                        ),
                    )
                else:
                    row = db.execute(
                        "SELECT belief_id, topic_key, revision FROM auto_beliefs WHERE belief_id = ?",
                        (target,),
                    ).fetchone()
                    if row is None:
                        return None, "revision_target_not_found"
                    if row["topic_key"] != clean_key:
                        return None, "revision_topic_mismatch"
                    belief_id = row["belief_id"]
                    revision = row["revision"] + 1
                    db.execute(
                        """UPDATE auto_beliefs SET
                        text = ?, epistemic_status = ?, evidence_status = ?,
                        source_quote = ?,
                        source_session_id = ?, source_turn_id = ?, source_episode_id = ?,
                        open_question = ?, revision = ?, updated_at = ?
                        WHERE belief_id = ?""",
                        (
                            text.strip(), status, evidence_status, quote, episode["session_id"],
                            episode["turn_id"], episode["episode_id"],
                            question, revision, now, belief_id,
                        ),
                    )

                db.execute(
                    """INSERT INTO auto_belief_versions (
                    version_id, belief_id, revision, text, epistemic_status,
                    evidence_status, source_quote, source_session_id, source_turn_id,
                    source_episode_id, open_question, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"version_{uuid4().hex}", belief_id, revision, text.strip(),
                        status, evidence_status, quote, episode["session_id"], episode["turn_id"],
                        episode["episode_id"], question, now,
                    ),
                )
                return dict(db.execute(
                    "SELECT * FROM auto_beliefs WHERE belief_id = ?", (belief_id,)
                ).fetchone()), "saved"

        return await asyncio.to_thread(work)
