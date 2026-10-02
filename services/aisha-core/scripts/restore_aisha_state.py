from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aisha.settings import Settings
from aisha.storage.snapshot import restore_snapshot_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore an audited AISHA local-state snapshot. Stop AISHA Core before "
            "running this command."
        )
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Target aisha.sqlite3 path. Defaults to AISHA_DATA_DIR.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = Settings()
    database = (
        args.database.expanduser().resolve()
        if args.database is not None
        else (settings.data_dir / "database" / "aisha.sqlite3").resolve()
    )
    result = await restore_snapshot_bundle(args.bundle, database)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    integrity = result["memory_integrity"]
    print("AISHA state restore: PASS")
    print(f"Target: {result['target_database']}")
    if result["previous_database_backup"]:
        print(f"Previous database backup: {result['previous_database_backup']}")
    print(
        f"Episodes: {integrity['episodes']} · "
        f"Beliefs: {integrity['beliefs']} · "
        f"Versions: {integrity['versions']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
