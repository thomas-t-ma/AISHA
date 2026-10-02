from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import urllib.request
from pathlib import Path

from aisha.settings import Settings

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def dependency_status() -> dict[str, bool]:
    return {
        "opencv": importlib.util.find_spec("cv2") is not None,
        "mediapipe": importlib.util.find_spec("mediapipe") is not None,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(path: Path, *, force: bool = False) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    try:
        with urllib.request.urlopen(
            FACE_LANDMARKER_MODEL_URL,
            timeout=90,
        ) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AISHA's optional local OpenCV/MediaPipe vision backend."
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the official versioned MediaPipe Face Landmarker model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the model even if it already exists.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override the local model path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else (settings.data_dir / "models" / "face_landmarker.task").resolve()
    )

    dependencies = dependency_status()
    if args.download_model:
        download_model(model_path, force=args.force)

    print("AISHA local vision")
    print(f"  OpenCV: {'ready' if dependencies['opencv'] else 'missing'}")
    print(f"  MediaPipe: {'ready' if dependencies['mediapipe'] else 'missing'}")
    print(f"  Model: {'ready' if model_path.exists() else 'missing'}")
    print(f"  Model path: {model_path}")
    if model_path.exists():
        print(f"  Model SHA-256: {sha256(model_path)}")

    ready = all(dependencies.values()) and model_path.exists()
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
