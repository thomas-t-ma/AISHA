from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from aisha.memory.ledger import ExperienceLedger
from aisha.storage.database import AISHAStore

SNAPSHOT_FORMAT_VERSION = 1
DATABASE_MEMBER = "aisha.sqlite3"
MANIFEST_MEMBER = "manifest.json"


class SnapshotError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3.Connection's context manager commits/rolls back but does not
    # close the handle. Windows refuses to delete temporary databases while
    # those handles remain open, so snapshot helpers close them explicitly.
    with closing(sqlite3.connect(source)) as source_db, closing(
        sqlite3.connect(destination)
    ) as dest_db:
        source_db.backup(dest_db)
        dest_db.commit()


async def _audit_database(path: Path) -> dict:
    return await ExperienceLedger(AISHAStore(path)).audit_integrity()


async def create_snapshot_bundle(source_database: Path, bundle_path: Path) -> dict:
    """Create an audited, checksummed SQLite snapshot bundle."""
    source_database = source_database.expanduser().resolve()
    bundle_path = bundle_path.expanduser().resolve()
    if not source_database.exists():
        raise SnapshotError(f"Source database not found: {source_database}")

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aisha-snapshot-") as temp_dir:
        snapshot_db = Path(temp_dir) / DATABASE_MEMBER
        await asyncio.to_thread(_sqlite_backup, source_database, snapshot_db)

        audit = await _audit_database(snapshot_db)
        if not audit["ok"]:
            raise SnapshotError(
                "Refusing to snapshot a memory ledger that fails integrity audit "
                f"({audit['issue_count']} issue(s))."
            )

        sha256 = await asyncio.to_thread(_sha256, snapshot_db)
        created_at = datetime.now(UTC).isoformat()
        manifest = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "created_at": created_at,
            "database_member": DATABASE_MEMBER,
            "database_sha256": sha256,
            "database_size_bytes": snapshot_db.stat().st_size,
            "memory_integrity": {
                "ok": True,
                "episodes": audit["episodes"],
                "beliefs": audit["beliefs"],
                "versions": audit["versions"],
                "issue_count": 0,
            },
        }

        def write_bundle() -> None:
            with zipfile.ZipFile(
                bundle_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                archive.write(snapshot_db, DATABASE_MEMBER)
                archive.writestr(
                    MANIFEST_MEMBER,
                    json.dumps(manifest, indent=2, sort_keys=True),
                )

        await asyncio.to_thread(write_bundle)
        return {"bundle": str(bundle_path), **manifest}


async def inspect_snapshot_bundle(bundle_path: Path) -> dict:
    """Verify bundle structure, checksum, and ledger integrity without restoring."""
    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.exists():
        raise SnapshotError(f"Snapshot bundle not found: {bundle_path}")

    with tempfile.TemporaryDirectory(prefix="aisha-inspect-") as temp_dir:
        temp_root = Path(temp_dir)

        def extract_and_read_manifest() -> tuple[dict, Path]:
            with zipfile.ZipFile(bundle_path, mode="r") as archive:
                members = set(archive.namelist())
                required = {DATABASE_MEMBER, MANIFEST_MEMBER}
                if not required <= members:
                    missing = sorted(required - members)
                    raise SnapshotError(
                        f"Snapshot bundle is missing required member(s): {missing}"
                    )
                if members != required:
                    extras = sorted(members - required)
                    raise SnapshotError(
                        f"Snapshot bundle contains unexpected member(s): {extras}"
                    )
                manifest = json.loads(archive.read(MANIFEST_MEMBER).decode("utf-8"))
                archive.extract(DATABASE_MEMBER, temp_root)
            return manifest, temp_root / DATABASE_MEMBER

        manifest, snapshot_db = await asyncio.to_thread(extract_and_read_manifest)
        if manifest.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotError(
                "Unsupported snapshot format version: "
                f"{manifest.get('format_version')!r}"
            )
        if manifest.get("database_member") != DATABASE_MEMBER:
            raise SnapshotError("Snapshot manifest database member does not match.")

        actual_sha256 = await asyncio.to_thread(_sha256, snapshot_db)
        expected_sha256 = manifest.get("database_sha256")
        if actual_sha256 != expected_sha256:
            raise SnapshotError("Snapshot database checksum mismatch.")

        actual_size = snapshot_db.stat().st_size
        if actual_size != manifest.get("database_size_bytes"):
            raise SnapshotError("Snapshot database size does not match manifest.")

        audit = await _audit_database(snapshot_db)
        if not audit["ok"]:
            raise SnapshotError(
                "Snapshot database fails memory integrity audit "
                f"({audit['issue_count']} issue(s))."
            )

        return {
            "ok": True,
            "bundle": str(bundle_path),
            "manifest": manifest,
            "memory_integrity": audit,
        }


async def restore_snapshot_bundle(
    bundle_path: Path,
    target_database: Path,
) -> dict:
    """Verify a bundle, preserve the current DB, then restore via SQLite backup."""
    bundle_path = bundle_path.expanduser().resolve()
    target_database = target_database.expanduser().resolve()
    inspection = await inspect_snapshot_bundle(bundle_path)

    with tempfile.TemporaryDirectory(prefix="aisha-restore-") as temp_dir:
        source_db = Path(temp_dir) / DATABASE_MEMBER

        def extract_database() -> None:
            with zipfile.ZipFile(bundle_path, mode="r") as archive:
                archive.extract(DATABASE_MEMBER, Path(temp_dir))

        await asyncio.to_thread(extract_database)
        target_database.parent.mkdir(parents=True, exist_ok=True)

        previous_backup: Path | None = None
        if target_database.exists():
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            previous_backup = target_database.with_name(
                f"{target_database.stem}.pre-restore-{stamp}{target_database.suffix}"
            )
            await asyncio.to_thread(
                _sqlite_backup,
                target_database,
                previous_backup,
            )

        try:
            await asyncio.to_thread(_sqlite_backup, source_db, target_database)
            restored_audit = await _audit_database(target_database)
            if not restored_audit["ok"]:
                raise SnapshotError(
                    "Restored database failed memory integrity audit "
                    f"({restored_audit['issue_count']} issue(s))."
                )
        except Exception:
            if previous_backup is not None and previous_backup.exists():
                await asyncio.to_thread(
                    _sqlite_backup,
                    previous_backup,
                    target_database,
                )
            raise

        return {
            "ok": True,
            "bundle": str(bundle_path),
            "target_database": str(target_database),
            "previous_database_backup": (
                str(previous_backup) if previous_backup is not None else None
            ),
            "manifest": inspection["manifest"],
            "memory_integrity": restored_audit,
        }
