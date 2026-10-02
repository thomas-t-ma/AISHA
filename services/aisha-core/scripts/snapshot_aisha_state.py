from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from aisha.settings import Settings
from aisha.storage.snapshot import create_snapshot_bundle, inspect_snapshot_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or verify an audited AISHA local-state snapshot bundle."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .zip path. Defaults to the current directory with a UTC timestamp.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Source aisha.sqlite3 path. Defaults to AISHA_DATA_DIR.",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Verify an existing snapshot bundle instead of creating one.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.verify is not None:
        result = await inspect_snapshot_bundle(args.verify)
    else:
        settings = Settings()
        database = (
            args.database.expanduser().resolve()
            if args.database is not None
            else (settings.data_dir / "database" / "aisha.sqlite3").resolve()
        )
        output = args.output
        if output is None:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output = Path.cwd() / f"aisha-state-{stamp}.zip"
        result = await create_snapshot_bundle(database, output)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if args.verify is not None:
        integrity = result["memory_integrity"]
        print("AISHA snapshot verification: PASS")
        print(f"Bundle: {result['bundle']}")
        print(
            f"Episodes: {integrity['episodes']} · "
            f"Beliefs: {integrity['beliefs']} · "
            f"Versions: {integrity['versions']}"
        )
    else:
        integrity = result["memory_integrity"]
        print("AISHA state snapshot created")
        print(f"Bundle: {result['bundle']}")
        print(f"SHA-256: {result['database_sha256']}")
        print(
            f"Episodes: {integrity['episodes']} · "
            f"Beliefs: {integrity['beliefs']} · "
            f"Versions: {integrity['versions']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
