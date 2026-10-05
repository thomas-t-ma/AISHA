from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from aisha.settings import Settings


def dependency_ready() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AISHA's optional local faster-whisper STT backend."
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the configured faster-whisper model into AISHA's local cache.",
    )
    parser.add_argument(
        "--model",
        default="small.en",
        help="faster-whisper model size or Hugging Face model ID.",
    )
    return parser.parse_args()


def resolve_local_model(
    model: str,
    cache_dir: Path,
    *,
    download: bool,
) -> str | None:
    if not dependency_ready():
        return None

    from faster_whisper.utils import download_model

    cache_dir.mkdir(parents=True, exist_ok=True)
    if download:
        download_model(
            model,
            cache_dir=str(cache_dir),
        )

    try:
        return download_model(
            model,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    settings = Settings()
    cache_dir = (settings.data_dir / "models" / "stt").resolve()
    model_path = resolve_local_model(
        args.model,
        cache_dir,
        download=args.download_model,
    )

    print("AISHA local listening")
    print(f"  faster-whisper: {'ready' if dependency_ready() else 'missing'}")
    print(f"  Model: {'ready' if model_path else 'missing'}")
    print(f"  Model name: {args.model}")
    print(f"  Cache: {cache_dir}")
    if model_path:
        print(f"  Snapshot: {model_path}")

    return 0 if dependency_ready() and model_path else 1


if __name__ == "__main__":
    raise SystemExit(main())
