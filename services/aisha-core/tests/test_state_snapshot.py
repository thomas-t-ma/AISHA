from __future__ import annotations

import json
import zipfile

import pytest

from aisha.memory.ledger import ExperienceLedger
from aisha.storage.database import AISHAStore
from aisha.storage.snapshot import (
    DATABASE_MEMBER,
    MANIFEST_MEMBER,
    SnapshotError,
    create_snapshot_bundle,
    inspect_snapshot_bundle,
    restore_snapshot_bundle,
)


@pytest.mark.asyncio
async def test_state_snapshot_round_trip_preserves_audited_memory(tmp_path):
    source_path = tmp_path / "source.sqlite3"
    source_store = AISHAStore(source_path)
    await source_store.initialize()
    source_ledger = ExperienceLedger(source_store)
    await source_ledger.initialize()

    episode = await source_ledger.record_episode(
        session_id="session_source",
        turn_id="turn_source",
        user_message_id="user_source",
        assistant_message_id="assistant_source",
        user_text="I prefer quiet linear keyboards.",
        assistant_text="Okay.",
    )
    saved, reason = await source_ledger.apply_with_reason(
        episode=episode,
        action={
            "action": "add",
            "target_belief_id": None,
            "topic_key": "keyboard_preference",
            "text": "The user prefers quiet linear keyboards.",
            "epistemic_status": "stated",
            "source_quote": "I prefer quiet linear keyboards.",
            "open_question": None,
        },
        evidence_checked=True,
    )
    assert saved is not None and reason == "saved"

    bundle = tmp_path / "aisha-state.zip"
    created = await create_snapshot_bundle(source_path, bundle)
    assert created["memory_integrity"]["ok"] is True
    assert created["memory_integrity"]["beliefs"] == 1
    assert bundle.exists()

    inspected = await inspect_snapshot_bundle(bundle)
    assert inspected["ok"] is True
    assert inspected["manifest"]["database_sha256"] == created["database_sha256"]

    target_path = tmp_path / "target.sqlite3"
    target_store = AISHAStore(target_path)
    await target_store.initialize()
    await target_store.create_session()

    restored = await restore_snapshot_bundle(bundle, target_path)
    assert restored["ok"] is True
    assert restored["previous_database_backup"] is not None
    assert restored["memory_integrity"]["ok"] is True

    target_ledger = ExperienceLedger(AISHAStore(target_path))
    audit = await target_ledger.audit_integrity()
    assert audit["ok"] is True
    assert audit["episodes"] == 1
    assert audit["beliefs"] == 1
    assert audit["versions"] == 1

    beliefs = await target_ledger.list_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["topic_key"] == "keyboard_preference"
    assert beliefs[0]["text"] == "The user prefers quiet linear keyboards."


@pytest.mark.asyncio
async def test_snapshot_verification_rejects_manifest_checksum_tampering(tmp_path):
    source_path = tmp_path / "source.sqlite3"
    store = AISHAStore(source_path)
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()

    bundle = tmp_path / "original.zip"
    await create_snapshot_bundle(source_path, bundle)

    with zipfile.ZipFile(bundle, "r") as archive:
        database_bytes = archive.read(DATABASE_MEMBER)
        manifest = json.loads(archive.read(MANIFEST_MEMBER).decode("utf-8"))

    manifest["database_sha256"] = "0" * 64
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(DATABASE_MEMBER, database_bytes)
        archive.writestr(MANIFEST_MEMBER, json.dumps(manifest))

    with pytest.raises(SnapshotError, match="checksum mismatch"):
        await inspect_snapshot_bundle(tampered)


@pytest.mark.asyncio
async def test_snapshot_creation_refuses_corrupt_memory_ledger(tmp_path):
    import sqlite3

    source_path = tmp_path / "source.sqlite3"
    store = AISHAStore(source_path)
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()

    episode = await ledger.record_episode(
        session_id="session_source",
        turn_id="turn_source",
        user_message_id="user_source",
        assistant_message_id="assistant_source",
        user_text="I like pottery.",
        assistant_text="Okay.",
    )
    saved, _ = await ledger.apply_with_reason(
        episode=episode,
        action={
            "action": "add",
            "target_belief_id": None,
            "topic_key": "pottery_hobby",
            "text": "The user likes pottery.",
            "epistemic_status": "stated",
            "source_quote": "I like pottery.",
            "open_question": None,
        },
        evidence_checked=True,
    )
    assert saved is not None

    with sqlite3.connect(source_path) as db:
        db.execute(
            "UPDATE auto_belief_versions SET text = ? WHERE belief_id = ?",
            ("Corrupted history", saved["belief_id"]),
        )

    with pytest.raises(SnapshotError, match="fails integrity audit"):
        await create_snapshot_bundle(source_path, tmp_path / "bad.zip")
