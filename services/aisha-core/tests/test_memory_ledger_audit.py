from __future__ import annotations

import sqlite3

import pytest

from aisha.memory.ledger import ExperienceLedger
from aisha.storage.database import AISHAStore


async def _episode(
    ledger: ExperienceLedger,
    *,
    suffix: str,
    user_text: str,
) -> dict:
    return await ledger.record_episode(
        session_id=f"session_{suffix}",
        turn_id=f"turn_{suffix}",
        user_message_id=f"user_{suffix}",
        assistant_message_id=f"assistant_{suffix}",
        user_text=user_text,
        assistant_text="Acknowledged.",
    )


@pytest.mark.asyncio
async def test_memory_ledger_integrity_audit_passes_clean_revision_history(tmp_path):
    store = AISHAStore(tmp_path / "memory.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()

    first = await _episode(
        ledger,
        suffix="one",
        user_text="I might move to Spain.",
    )
    saved, reason = await ledger.apply_with_reason(
        episode=first,
        action={
            "action": "add",
            "target_belief_id": None,
            "topic_key": "country_relocation",
            "text": "The user may move to Spain.",
            "epistemic_status": "uncertain",
            "source_quote": "I might move to Spain.",
            "open_question": "Will the user move?",
        },
        evidence_checked=True,
    )
    assert saved is not None and reason == "saved"

    second = await _episode(
        ledger,
        suffix="two",
        user_text="I decided not to move to Spain.",
    )
    revised, reason = await ledger.apply_with_reason(
        episode=second,
        action={
            "action": "revise",
            "target_belief_id": saved["belief_id"],
            "topic_key": "country_relocation",
            "text": "The user decided not to move to Spain.",
            "epistemic_status": "stated",
            "source_quote": "I decided not to move to Spain.",
            "open_question": None,
        },
        evidence_checked=True,
    )
    assert revised is not None and reason == "saved"

    audit = await ledger.audit_integrity()
    assert audit == {
        "ok": True,
        "episodes": 2,
        "beliefs": 1,
        "versions": 2,
        "issue_count": 0,
        "issues": [],
    }


@pytest.mark.asyncio
async def test_memory_ledger_integrity_audit_detects_active_history_divergence(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = AISHAStore(path)
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()

    episode = await _episode(
        ledger,
        suffix="one",
        user_text="I prefer quiet keyboards.",
    )
    saved, _ = await ledger.apply_with_reason(
        episode=episode,
        action={
            "action": "add",
            "target_belief_id": None,
            "topic_key": "keyboard_preference",
            "text": "The user prefers quiet keyboards.",
            "epistemic_status": "stated",
            "source_quote": "I prefer quiet keyboards.",
            "open_question": None,
        },
        evidence_checked=True,
    )
    assert saved is not None

    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE auto_beliefs SET text = ? WHERE belief_id = ?",
            ("Corrupted active text", saved["belief_id"]),
        )

    audit = await ledger.audit_integrity()
    codes = {issue["code"] for issue in audit["issues"]}
    assert audit["ok"] is False
    assert "active_belief_differs_from_latest_version" in codes


@pytest.mark.asyncio
async def test_memory_ledger_integrity_audit_detects_bad_version_provenance(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = AISHAStore(path)
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()

    episode = await _episode(
        ledger,
        suffix="one",
        user_text="I enjoy pottery.",
    )
    saved, _ = await ledger.apply_with_reason(
        episode=episode,
        action={
            "action": "add",
            "target_belief_id": None,
            "topic_key": "pottery_hobby",
            "text": "The user enjoys pottery.",
            "epistemic_status": "stated",
            "source_quote": "I enjoy pottery.",
            "open_question": None,
        },
        evidence_checked=True,
    )
    assert saved is not None

    with sqlite3.connect(path) as db:
        db.execute(
            """UPDATE auto_belief_versions
            SET source_episode_id = ?, source_quote = ?
            WHERE belief_id = ? AND revision = 1""",
            ("episode_missing", "fabricated quote", saved["belief_id"]),
        )

    audit = await ledger.audit_integrity()
    codes = {issue["code"] for issue in audit["issues"]}
    assert audit["ok"] is False
    assert "source_episode_missing" in codes
    assert "active_belief_differs_from_latest_version" in codes


@pytest.mark.asyncio
async def test_memory_ledger_integrity_audit_detects_duplicate_revision(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = AISHAStore(path)
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()

    episode = await _episode(
        ledger,
        suffix="one",
        user_text="I like remote work.",
    )
    saved, _ = await ledger.apply_with_reason(
        episode=episode,
        action={
            "action": "add",
            "target_belief_id": None,
            "topic_key": "remote_work",
            "text": "The user likes remote work.",
            "epistemic_status": "stated",
            "source_quote": "I like remote work.",
            "open_question": None,
        },
        evidence_checked=True,
    )
    assert saved is not None

    with sqlite3.connect(path) as db:
        original = db.execute(
            """SELECT * FROM auto_belief_versions
            WHERE belief_id = ? AND revision = 1""",
            (saved["belief_id"],),
        ).fetchone()
        assert original is not None
        db.execute(
            """INSERT INTO auto_belief_versions (
            version_id, belief_id, revision, text, epistemic_status,
            evidence_status, source_quote, source_session_id, source_turn_id,
            source_episode_id, open_question, recorded_at
            ) SELECT ?, belief_id, revision, text, epistemic_status,
            evidence_status, source_quote, source_session_id, source_turn_id,
            source_episode_id, open_question, recorded_at
            FROM auto_belief_versions
            WHERE belief_id = ? AND revision = 1""",
            ("duplicate_version", saved["belief_id"]),
        )

    audit = await ledger.audit_integrity()
    codes = {issue["code"] for issue in audit["issues"]}
    assert audit["ok"] is False
    assert "duplicate_revision" in codes
    assert "non_contiguous_revision_history" in codes
