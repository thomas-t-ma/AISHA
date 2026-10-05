from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any
from uuid import uuid4

from aisha.perception.base import BoundingBox, VisionFrame, VisionObservation
from aisha.perception.camera import CameraFrameDescriptor


class EphemeralFrameStore:
    """Small bounded provider-local store for raw frames.

    Entries are transient process memory only. They are never serialized by
    AISHA Core and are removed as soon as an analyzer consumes them.
    """

    def __init__(self, *, max_frames: int = 3) -> None:
        self.max_frames = max(1, max_frames)
        self._frames: OrderedDict[str, Any] = OrderedDict()

    def put(self, frame_ref: str, frame: Any) -> None:
        self._frames[frame_ref] = frame
        self._frames.move_to_end(frame_ref)
        while len(self._frames) > self.max_frames:
            self._frames.popitem(last=False)

    def pop(self, frame_ref: str) -> Any | None:
        return self._frames.pop(frame_ref, None)

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


class OpenCVCameraSource:
    """Optional local camera source using OpenCV VideoCapture."""

    name = "opencv-camera"

    def __init__(
        self,
        frame_store: EphemeralFrameStore,
        *,
        camera_index: int = 0,
        source_id: str = "camera_front",
        cv2_module: Any | None = None,
    ) -> None:
        self.frame_store = frame_store
        self.camera_index = camera_index
        self.source_id = source_id
        self._cv2 = cv2_module
        self._capture: Any | None = None
        self._last_error: str | None = None

    def _dependency_available(self) -> bool:
        return self._cv2 is not None or importlib.util.find_spec("cv2") is not None

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            self._cv2 = importlib.import_module("cv2")
        return self._cv2

    def enable(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return

        self._last_error = None
        try:
            cv2 = self._load_cv2()
            backend = getattr(cv2, "CAP_ANY", 0)
            if os.name == "nt":
                backend = getattr(cv2, "CAP_MSMF", backend)
            capture = cv2.VideoCapture(self.camera_index, backend)
            if not capture.isOpened():
                capture.release()
                self._last_error = (
                    f"Could not open camera index {self.camera_index}"
                )
                return
            self._capture = capture
        except Exception as exc:  # noqa: BLE001 - isolate optional camera backend failures
            self._capture = None
            self._last_error = str(exc)

    def disable(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            finally:
                self.frame_store.clear()
        else:
            self.frame_store.clear()

    async def capture(self) -> CameraFrameDescriptor | None:
        capture = self._capture
        if capture is None or not capture.isOpened():
            return None

        ok, frame = await asyncio.to_thread(capture.read)
        if not ok or frame is None:
            self._last_error = "Camera read failed"
            return None

        shape = getattr(frame, "shape", ())
        height = int(shape[0]) if len(shape) >= 2 else None
        width = int(shape[1]) if len(shape) >= 2 else None
        frame_ref = f"capture_{uuid4().hex}"
        self.frame_store.put(frame_ref, frame)
        self._last_error = None
        return CameraFrameDescriptor(
            frame_ref=frame_ref,
            source_id=self.source_id,
            width=width,
            height=height,
        )

    def status(self) -> dict[str, Any]:
        enabled = self._capture is not None and self._capture.isOpened()
        return {
            "available": self._dependency_available(),
            "enabled": enabled,
            "source": self.name,
            "camera_index": self.camera_index,
            "backend": "msmf" if os.name == "nt" else "auto",
            "raw_frame_scope": "ephemeral-provider-memory",
            "buffered_frames": len(self.frame_store),
            "last_error": self._last_error,
        }


class MediaPipeFaceAnalyzer:
    """Face geometry/action analyzer backed by MediaPipe Face Landmarker."""

    name = "mediapipe-face-landmarker"

    def __init__(
        self,
        frame_store: EphemeralFrameStore,
        model_path: Path,
        *,
        num_faces: int = 2,
        object_model_path: Path | None = None,
        object_score_threshold: float = 0.45,
        object_max_results: int = 8,
        mediapipe_module: Any | None = None,
    ) -> None:
        self.frame_store = frame_store
        self.model_path = model_path.expanduser().resolve()
        self.num_faces = max(1, num_faces)
        self.object_model_path = (
            None
            if object_model_path is None
            else object_model_path.expanduser().resolve()
        )
        self.object_score_threshold = min(1.0, max(0.0, object_score_threshold))
        self.object_max_results = max(1, object_max_results)
        self._mp = mediapipe_module
        self._landmarker: Any | None = None
        self._object_detector: Any | None = None
        self._last_timestamp_ms = 0
        self._last_error: str | None = None

    def _dependency_available(self) -> bool:
        return (
            self._mp is not None
            or importlib.util.find_spec("mediapipe") is not None
        )

    def _load_mediapipe(self) -> Any:
        if self._mp is None:
            self._mp = importlib.import_module("mediapipe")
        return self._mp

    def _ensure_landmarker(self) -> Any:
        if self._landmarker is not None:
            return self._landmarker
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {self.model_path}"
            )

        mp = self._load_mediapipe()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        return self._landmarker

    def _ensure_object_detector(self) -> Any | None:
        if self.object_model_path is None:
            return None
        if self._object_detector is not None:
            return self._object_detector
        if not self.object_model_path.exists():
            return None

        mp = self._load_mediapipe()
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.object_model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            max_results=self.object_max_results,
            score_threshold=self.object_score_threshold,
        )
        self._object_detector = mp.tasks.vision.ObjectDetector.create_from_options(
            options
        )
        return self._object_detector

    @staticmethod
    def _bounding_box(landmarks: list[Any]) -> BoundingBox | None:
        xs = [
            float(point.x)
            for point in landmarks
            if getattr(point, "x", None) is not None
        ]
        ys = [
            float(point.y)
            for point in landmarks
            if getattr(point, "y", None) is not None
        ]
        if not xs or not ys:
            return None

        left = max(0.0, min(1.0, min(xs)))
        top = max(0.0, min(1.0, min(ys)))
        right = max(left, min(1.0, max(xs)))
        bottom = max(top, min(1.0, max(ys)))
        width = max(1e-6, right - left)
        height = max(1e-6, bottom - top)
        return BoundingBox(x=left, y=top, width=width, height=height)

    @staticmethod
    def _matrix_values(matrix: Any) -> list[float] | list[list[float]] | None:
        if matrix is None:
            return None
        if hasattr(matrix, "tolist"):
            return matrix.tolist()
        data = getattr(matrix, "data", None)
        if data is not None:
            return [float(value) for value in data]
        return None

    @staticmethod
    def _frontal_score(
        matrix: list[float] | list[list[float]] | None,
    ) -> float | None:
        """Return camera-axis alignment of the canonical face forward axis.

        The score is geometric, not a probability. It is invariant to a uniform
        scale in MediaPipe's facial transform and deliberately ignores sign so it
        does not depend on camera/canonical Z-axis direction conventions.
        """

        if matrix is None:
            return None

        rows: list[list[float]]
        if matrix and isinstance(matrix[0], list):
            rows = matrix  # type: ignore[assignment]
        else:
            flat = [float(value) for value in matrix]  # type: ignore[arg-type]
            if len(flat) < 12:
                return None
            width = 4 if len(flat) >= 16 else 3
            rows = [
                flat[offset : offset + width]
                for offset in range(0, min(len(flat), width * 3), width)
            ]

        if len(rows) < 3 or any(len(row) < 3 for row in rows[:3]):
            return None

        forward_x = float(rows[0][2])
        forward_y = float(rows[1][2])
        forward_z = float(rows[2][2])
        norm = (forward_x**2 + forward_y**2 + forward_z**2) ** 0.5
        if norm <= 1e-8:
            return None
        return min(1.0, max(0.0, abs(forward_z) / norm))

    @classmethod
    def observations_from_result(cls, result: Any) -> list[VisionObservation]:
        faces = list(getattr(result, "face_landmarks", []) or [])
        matrices = list(
            getattr(result, "facial_transformation_matrixes", []) or []
        )

        observations: list[VisionObservation] = []
        for index, landmarks in enumerate(faces):
            attributes: dict[str, Any] = {
                "face_index": index,
                "confidence_source": "presence_unscored",
            }
            matrix = None
            if index < len(matrices):
                matrix = cls._matrix_values(matrices[index])
            observations.append(
                VisionObservation(
                    kind="face",
                    confidence=0.5,
                    label="face",
                    bounding_box=cls._bounding_box(list(landmarks)),
                    attributes=attributes,
                )
            )

            frontal_score = cls._frontal_score(matrix)
            if frontal_score is not None:
                observations.append(
                    VisionObservation(
                        kind="head_pose",
                        confidence=0.5,
                        label=(
                            "approximately_frontal"
                            if frontal_score >= 0.90
                            else "turned"
                        ),
                        attributes={
                            "face_index": index,
                            "frontal_score": frontal_score,
                            "score_kind": "forward_axis_alignment",
                        },
                    )
                )
        return observations

    @staticmethod
    def object_observations_from_result(
        result: Any,
        *,
        width: int | None,
        height: int | None,
    ) -> list[VisionObservation]:
        if not width or not height:
            return []

        observations: list[VisionObservation] = []
        for detection_index, detection in enumerate(
            list(getattr(result, "detections", []) or [])
        ):
            categories = list(getattr(detection, "categories", []) or [])
            if not categories:
                continue
            category = categories[0]
            label = (
                getattr(category, "category_name", None)
                or getattr(category, "display_name", None)
            )
            score = getattr(category, "score", None)
            box = getattr(detection, "bounding_box", None)
            if not label or score is None or box is None:
                continue

            origin_x = float(getattr(box, "origin_x", 0.0))
            origin_y = float(getattr(box, "origin_y", 0.0))
            box_width = float(getattr(box, "width", 0.0))
            box_height = float(getattr(box, "height", 0.0))
            if box_width <= 0 or box_height <= 0:
                continue

            normalized = BoundingBox(
                x=min(1.0, max(0.0, origin_x / width)),
                y=min(1.0, max(0.0, origin_y / height)),
                width=min(1.0, max(1e-6, box_width / width)),
                height=min(1.0, max(1e-6, box_height / height)),
            )
            observations.append(
                VisionObservation(
                    kind="object",
                    confidence=min(1.0, max(0.0, float(score))),
                    label=str(label).strip().lower(),
                    bounding_box=normalized,
                    attributes={"detection_index": detection_index},
                )
            )
        return observations

    async def analyze(
        self,
        capture: CameraFrameDescriptor,
    ) -> VisionFrame | None:
        frame = self.frame_store.pop(capture.frame_ref)
        if frame is None:
            self._last_error = "Captured frame reference expired"
            return None

        try:
            landmarker = self._ensure_landmarker()
            mp = self._load_mediapipe()
            rgb = frame[:, :, ::-1].copy()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(capture.captured_at.timestamp() * 1000)
            timestamp_ms = max(self._last_timestamp_ms + 1, timestamp_ms)
            self._last_timestamp_ms = timestamp_ms

            result = await asyncio.to_thread(
                landmarker.detect_for_video,
                image,
                timestamp_ms,
            )
            observations = self.observations_from_result(result)

            object_detector = self._ensure_object_detector()
            if object_detector is not None:
                object_result = await asyncio.to_thread(
                    object_detector.detect_for_video,
                    image,
                    timestamp_ms,
                )
                observations.extend(
                    self.object_observations_from_result(
                        object_result,
                        width=capture.width,
                        height=capture.height,
                    )
                )

            self._last_error = None
            return VisionFrame(
                source_id=capture.source_id,
                captured_at=capture.captured_at,
                width=capture.width,
                height=capture.height,
                observations=observations,
                image_ref=None,
            )
        except Exception as exc:  # noqa: BLE001 - isolate optional vision backend failures
            self._last_error = str(exc)
            return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._landmarker is not None,
            "analyzer": self.name,
            "dependency_available": self._dependency_available(),
            "model_path": str(self.model_path),
            "model_available": self.model_path.exists(),
            "object_detection_configured": self.object_model_path is not None,
            "object_model_path": (
                None if self.object_model_path is None else str(self.object_model_path)
            ),
            "object_model_available": (
                False
                if self.object_model_path is None
                else self.object_model_path.exists()
            ),
            "object_detector_enabled": self._object_detector is not None,
            "last_error": self._last_error,
        }

    def close(self) -> None:
        landmarker = self._landmarker
        object_detector = self._object_detector
        self._landmarker = None
        self._object_detector = None
        if landmarker is not None:
            landmarker.close()
        if object_detector is not None:
            object_detector.close()
        self.frame_store.clear()
