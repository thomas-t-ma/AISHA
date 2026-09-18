from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from aisha.contracts.events import AISHAEvent
from aisha.contracts.turns import Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'committed',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_session_created
ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    first_token_ms REAL,
    total_ms REAL,
    output_chars INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_model_runs_session_started
ON model_runs(session_id, started_at);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_events_session_timestamp
ON events(session_id, timestamp);
"""


class AISHAStore:
    """Small SQLite store using only Python's standard library.

    SQLite calls are moved to worker threads so AISHA's async event loop is not blocked.
    This keeps the first persistence layer portable across macOS/Windows/Linux without
    a platform-specific database dependency.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def work() -> None:
            with self._connect() as db:
                db.executescript(SCHEMA)

        await asyncio.to_thread(work)

    async def create_session(self) -> str:
        session_id = f"session_{uuid4().hex}"

        def work() -> None:
            with self._connect() as db:
                db.execute("INSERT INTO sessions(session_id) VALUES (?)", (session_id,))

        await asyncio.to_thread(work)
        return session_id

    async def ensure_session(self, session_id: str) -> None:
        def work() -> None:
            with self._connect() as db:
                db.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (session_id,))

        await asyncio.to_thread(work)

    async def add_message(self, session_id: str, message: Message, status: str = "committed") -> None:
        def work() -> None:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO messages(message_id, session_id, role, text, status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        message.message_id,
                        session_id,
                        message.role,
                        message.text,
                        status,
                        message.created_at.isoformat(),
                    ),
                )

        await asyncio.to_thread(work)

    async def recent_messages(self, session_id: str, limit: int = 20) -> list[Message]:
        def work() -> list[sqlite3.Row]:
            query = """
            SELECT message_id, role, text, created_at
            FROM messages
            WHERE session_id = ? AND status = 'committed'
            ORDER BY created_at DESC
            LIMIT ?
            """
            with self._connect() as db:
                return list(db.execute(query, (session_id, limit)).fetchall())

        rows = await asyncio.to_thread(work)
        rows.reverse()
        return [
            Message(
                message_id=row["message_id"],
                role=row["role"],
                text=row["text"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def start_model_run(
        self,
        session_id: str,
        turn_id: str,
        provider: str,
        model: str,
    ) -> str:
        run_id = f"run_{uuid4().hex}"

        def work() -> None:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO model_runs(run_id, session_id, turn_id, provider, model, status) "
                    "VALUES (?, ?, ?, ?, ?, 'running')",
                    (run_id, session_id, turn_id, provider, model),
                )

        await asyncio.to_thread(work)
        return run_id

    async def finish_model_run(
        self,
        run_id: str,
        *,
        status: str,
        first_token_ms: float | None,
        total_ms: float,
        output_chars: int,
        error: str | None = None,
    ) -> None:
        def work() -> None:
            with self._connect() as db:
                db.execute(
                    """
                    UPDATE model_runs
                    SET status = ?, first_token_ms = ?, total_ms = ?, output_chars = ?, error = ?
                    WHERE run_id = ?
                    """,
                    (status, first_token_ms, total_ms, output_chars, error, run_id),
                )

        await asyncio.to_thread(work)

    async def add_event(self, event: AISHAEvent) -> None:
        def work() -> None:
            with self._connect() as db:
                db.execute(
                    """
                    INSERT INTO events(
                        event_id, session_id, turn_id, timestamp, source, type, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.session_id,
                        event.turn_id,
                        event.timestamp.isoformat(),
                        event.source,
                        event.type,
                        json.dumps(event.payload, separators=(",", ":")),
                    ),
                )

        await asyncio.to_thread(work)

    async def session_events(self, session_id: str, limit: int = 200) -> list[dict]:
        safe_limit = max(1, min(limit, 2000))

        def work() -> list[sqlite3.Row]:
            with self._connect() as db:
                return list(
                    db.execute(
                        """
                        SELECT event_id, session_id, turn_id, timestamp, source, type, payload_json
                        FROM events
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                        LIMIT ?
                        """,
                        (session_id, safe_limit),
                    ).fetchall()
                )

        rows = await asyncio.to_thread(work)
        return [
            {
                "event_id": row["event_id"],
                "session_id": row["session_id"],
                "turn_id": row["turn_id"],
                "timestamp": row["timestamp"],
                "source": row["source"],
                "type": row["type"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    async def session_model_runs(self, session_id: str, limit: int = 100) -> list[dict]:
        safe_limit = max(1, min(limit, 1000))

        def work() -> list[sqlite3.Row]:
            with self._connect() as db:
                return list(
                    db.execute(
                        """
                        SELECT run_id, session_id, turn_id, provider, model, status, started_at,
                               first_token_ms, total_ms, output_chars, error
                        FROM model_runs
                        WHERE session_id = ?
                        ORDER BY started_at ASC
                        LIMIT ?
                        """,
                        (session_id, safe_limit),
                    ).fetchall()
                )

        rows = await asyncio.to_thread(work)
        return [dict(row) for row in rows]
