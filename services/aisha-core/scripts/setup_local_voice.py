from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import urllib.request
from pathlib import Path

from aisha.settings import Settings

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/kokoro-v1.0.onnx"
)
KOKORO_MODEL_SHA256 = "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/voices-v1.0.bin"
)
KOKORO_VOICES_SHA256 = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"


def dependency_status() -> dict[str, bool]:
    return {
        "kokoro_onnx": importlib.util.find_spec("kokoro_onnx") is not None,
        "soundfile": importlib.util.find_spec("soundfile") is not None,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    url: str,
    path: Path,
    expected_sha256: str,
    *,
    force: bool = False,
) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        actual = sha256(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for existing {path.name}: {actual}"
            )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    try:
        with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for downloaded {path.name}: {actual}"
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AISHA's optional local Kokoro speech backend."
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download and verify the Kokoro model and voice bundle.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload assets even when verified local copies exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    model_dir = (settings.data_dir / "models" / "tts").resolve()
    model_path = model_dir / "kokoro-v1.0.onnx"
    voices_path = model_dir / "voices-v1.0.bin"

    if args.download_models:
        download_verified(
            KOKORO_MODEL_URL,
            model_path,
            KOKORO_MODEL_SHA256,
            force=args.force,
        )
        download_verified(
            KOKORO_VOICES_URL,
            voices_path,
            KOKORO_VOICES_SHA256,
            force=args.force,
        )

    dependencies = dependency_status()
    model_ready = (
        model_path.exists()
        and sha256(model_path) == KOKORO_MODEL_SHA256
    )
    voices_ready = (
        voices_path.exists()
        and sha256(voices_path) == KOKORO_VOICES_SHA256
    )

    print("AISHA local voice")
    print(f"  kokoro-onnx: {'ready' if dependencies['kokoro_onnx'] else 'missing'}")
    print(f"  soundfile: {'ready' if dependencies['soundfile'] else 'missing'}")
    print(f"  Model: {'verified' if model_ready else 'missing/unverified'}")
    print(f"  Voices: {'verified' if voices_ready else 'missing/unverified'}")
    print(f"  Model path: {model_path}")
    print(f"  Voices path: {voices_path}")

    ready = all(dependencies.values()) and model_ready and voices_ready
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
