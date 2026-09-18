from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from uuid import uuid4

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
