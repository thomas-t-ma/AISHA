from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import urllib.request
from pathlib import Path

from aisha.perception.local import EphemeralFrameStore, MediaPipeFaceAnalyzer
from aisha.settings import Settings

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
OBJECT_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)

FACE_LANDMARKER_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)
OBJECT_DETECTOR_SHA256 = (
    "0720bf247bd76e6594ea28fa9c6f7c5242be774818997dbbeffc4da460c723bb"
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


def verify_sha256(path: Path, expected_sha256: str) -> None:
    actual = sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected "
            f"{expected_sha256}, got {actual}"
        )


def download_model(
    path: Path,
    *,
    url: str,
    expected_sha256: str,
    force: bool = False,
) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        verify_sha256(path, expected_sha256)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    try:
        with urllib.request.urlopen(
            url,
            timeout=90,
        ) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify_sha256(temporary, expected_sha256)
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
        help="Download the official versioned MediaPipe vision models.",
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
    object_model_path = (
        settings.data_dir / "models" / "efficientdet_lite0.tflite"
    ).resolve()

    dependencies = dependency_status()
    if args.download_model:
        download_model(
            model_path,
            url=FACE_LANDMARKER_MODEL_URL,
            expected_sha256=FACE_LANDMARKER_SHA256,
            force=args.force,
        )
        download_model(
            object_model_path,
            url=OBJECT_DETECTOR_MODEL_URL,
            expected_sha256=OBJECT_DETECTOR_SHA256,
            force=args.force,
        )

    print("AISHA local vision")
    print(f"  OpenCV: {'ready' if dependencies['opencv'] else 'missing'}")
    print(f"  MediaPipe: {'ready' if dependencies['mediapipe'] else 'missing'}")
    print(f"  Face model: {'ready' if model_path.exists() else 'missing'}")
    print(f"  Face model path: {model_path}")
    if model_path.exists():
        face_hash = sha256(model_path)
        print(f"  Face model SHA-256: {face_hash}")
        if face_hash.lower() != FACE_LANDMARKER_SHA256.lower():
            print("  Face model integrity: FAILED")
    print(
        f"  Object model: {'ready' if object_model_path.exists() else 'missing'}"
    )
    print(f"  Object model path: {object_model_path}")
    if object_model_path.exists():
        object_hash = sha256(object_model_path)
        print(f"  Object model SHA-256: {object_hash}")
        if object_hash.lower() != OBJECT_DETECTOR_SHA256.lower():
            print("  Object model integrity: FAILED")

    ready = (
        all(dependencies.values())
        and model_path.exists()
        and object_model_path.exists()
        and sha256(model_path).lower() == FACE_LANDMARKER_SHA256.lower()
        and sha256(object_model_path).lower() == OBJECT_DETECTOR_SHA256.lower()
    )
    if ready:
        analyzer = MediaPipeFaceAnalyzer(
            EphemeralFrameStore(max_frames=1),
            model_path,
            object_model_path=object_model_path,
        )
        try:
            prepared = analyzer.prepare()
            print("  Model load: ready")
            print(
                "  Face task: "
                + ("ready" if prepared["enabled"] else "missing")
            )
            print(
                "  Object task: "
                + (
                    "ready"
                    if prepared["object_detector_enabled"]
                    else "missing"
                )
            )
        except Exception as exc:  # noqa: BLE001 - setup must report backend errors
            print(f"  Model load: failed ({type(exc).__name__}: {exc})")
            ready = False
        finally:
            analyzer.close()

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
