from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aisha.memory.ledger import ExperienceLedger
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit AISHA autobiographical memory integrity without calling any model."
        )
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help=(
            "Path to aisha.sqlite3. Defaults to the database under AISHA_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable audit result.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = Settings()
    database = (
        args.database.expanduser().resolve()
        if args.database is not None
        else (settings.data_dir / "database" / "aisha.sqlite3").resolve()
    )

    if not database.exists():
        raise SystemExit(f"AISHA database not found: {database}")

    ledger = ExperienceLedger(AISHAStore(database))
    result = await ledger.audit_integrity()

    if args.json:
        print(json.dumps({"database": str(database), **result}, indent=2))
    else:
        print("AISHA memory ledger integrity audit")
        print(f"Database: {database}")
        print(
            f"Episodes: {result['episodes']} · "
            f"Beliefs: {result['beliefs']} · "
            f"Versions: {result['versions']}"
        )
        print(
            "Result: "
            + ("PASS" if result["ok"] else f"FAIL ({result['issue_count']} issues)")
        )
        for issue in result["issues"]:
            details = ", ".join(
                f"{key}={value}"
                for key, value in issue.items()
                if key != "code"
            )
            print(
                f"  - {issue['code']}"
                + (f": {details}" if details else "")
            )

    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
