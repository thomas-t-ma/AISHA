from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CORE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CORE_DIR.parents[1]


class LLMProfile(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    think: bool | Literal["low", "medium", "high", "max"] | None = None
    keep_alive: str | int | None = None
    preload: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class MemoryProfile(BaseModel):
    semantic_recall: bool = False
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_keep_alive: str | int | None = None
    semantic_threshold: float = 0.72
    semantic_candidate_floor: float = 0.30
    semantic_limit: int = 2
    semantic_relevance_gate: bool = False
    semantic_relevance_model: str | None = None
    semantic_relevance_base_url: str | None = None
    semantic_relevance_keep_alive: str | int | None = None
    semantic_query_instruction: str = (
        "Given a user's current message, retrieve a previously stated personal "
        "memory that is directly relevant and useful for responding. Prefer the "
        "same situation or underlying concern even when phrased differently; "
        "avoid merely topical or generic associations."
    )


class STTProfile(BaseModel):
    provider: Literal["disabled", "local-faster-whisper"] = "disabled"
    model: str = "small.en"
    device: Literal["cpu", "cuda"] = "cpu"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = Field(default=3, ge=1, le=10)
    vad_filter: bool = True
    max_audio_bytes: int = Field(
        default=12 * 1024 * 1024,
        ge=64 * 1024,
        le=128 * 1024 * 1024,
    )


class TTSProfile(BaseModel):
    provider: Literal["disabled", "local-kokoro"] = "disabled"
    voice: str = "af_heart"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    language: str = "en-us"
    model_path: str | None = None
    voices_path: str | None = None
    artifact_ttl_seconds: float = Field(default=120.0, ge=10.0, le=3600.0)


class VisionProfile(BaseModel):
    provider: Literal["disabled", "local-mediapipe"] = "disabled"
    camera_index: int = Field(default=0, ge=0)
    source_id: str = "camera_front"
    poll_interval_seconds: float = Field(default=0.5, ge=0.05, le=10.0)
    num_faces: int = Field(default=2, ge=1, le=8)
    model_path: str | None = None
    object_detection: bool = True
    object_model_path: str | None = None
    object_score_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    object_max_results: int = Field(default=8, ge=1, le=50)


class RuntimeProfile(BaseModel):
    name: str
    compute: dict[str, Any]
    llm: LLMProfile
    memory: MemoryProfile = Field(default_factory=MemoryProfile)
    stt: STTProfile = Field(default_factory=STTProfile)
    tts: TTSProfile = Field(default_factory=TTSProfile)
    vision: VisionProfile = Field(default_factory=VisionProfile)
    notes: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    aisha_profile: str = "mock"
    aisha_data_dir: str = "../../.aisha-data"
    aisha_host: str = "127.0.0.1"
    aisha_port: int = 8000
    aisha_log_level: str = "INFO"
    aisha_auto_memory: bool = True

    aisha_llm_provider: str | None = None
    aisha_llm_model: str | None = None
    aisha_llm_base_url: str | None = None
    aisha_llm_api_key: str | None = None
    aisha_llm_think: bool | Literal["low", "medium", "high", "max"] | None = None
    aisha_llm_keep_alive: str | int | None = None

    aisha_stt_provider: Literal["disabled", "local-faster-whisper"] | None = None
    aisha_stt_model: str | None = None
    aisha_stt_device: Literal["cpu", "cuda"] | None = None
    aisha_stt_compute_type: str | None = None

    aisha_tts_provider: Literal["disabled", "local-kokoro"] | None = None
    aisha_tts_voice: str | None = None
    aisha_tts_model_path: str | None = None
    aisha_tts_voices_path: str | None = None

    aisha_vision_provider: Literal["disabled", "local-mediapipe"] | None = None
    aisha_vision_camera_index: int | None = None
    aisha_vision_model_path: str | None = None
    aisha_vision_object_detection: bool | None = None
    aisha_vision_object_model_path: str | None = None

    @property
    def data_dir(self) -> Path:
        path = Path(self.aisha_data_dir).expanduser()
        if not path.is_absolute():
            path = (CORE_DIR / path).resolve()
        return path

    @property
    def character_dir(self) -> Path:
        return REPO_ROOT / "data" / "aisha-character"

    def load_profile(self) -> RuntimeProfile:
        path = REPO_ROOT / "configs" / "profiles" / f"{self.aisha_profile}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Unknown AISHA profile: {self.aisha_profile} ({path})")
        with path.open("r", encoding="utf-8") as handle:
            profile = RuntimeProfile.model_validate(yaml.safe_load(handle))

        if self.aisha_llm_provider:
            profile.llm.provider = self.aisha_llm_provider
        if self.aisha_llm_model:
            profile.llm.model = self.aisha_llm_model
        if self.aisha_llm_base_url:
            profile.llm.base_url = self.aisha_llm_base_url
        if self.aisha_llm_think is not None:
            profile.llm.think = self.aisha_llm_think
        if self.aisha_llm_keep_alive is not None:
            profile.llm.keep_alive = self.aisha_llm_keep_alive
        if self.aisha_stt_provider is not None:
            profile.stt.provider = self.aisha_stt_provider
        if self.aisha_stt_model is not None:
            profile.stt.model = self.aisha_stt_model
        if self.aisha_stt_device is not None:
            profile.stt.device = self.aisha_stt_device
        if self.aisha_stt_compute_type is not None:
            profile.stt.compute_type = self.aisha_stt_compute_type
        if self.aisha_tts_provider is not None:
            profile.tts.provider = self.aisha_tts_provider
        if self.aisha_tts_voice is not None:
            profile.tts.voice = self.aisha_tts_voice
        if self.aisha_tts_model_path is not None:
            profile.tts.model_path = self.aisha_tts_model_path
        if self.aisha_tts_voices_path is not None:
            profile.tts.voices_path = self.aisha_tts_voices_path
        if self.aisha_vision_provider is not None:
            profile.vision.provider = self.aisha_vision_provider
        if self.aisha_vision_camera_index is not None:
            profile.vision.camera_index = self.aisha_vision_camera_index
        if self.aisha_vision_model_path is not None:
            profile.vision.model_path = self.aisha_vision_model_path
        if self.aisha_vision_object_detection is not None:
            profile.vision.object_detection = self.aisha_vision_object_detection
        if self.aisha_vision_object_model_path is not None:
            profile.vision.object_model_path = self.aisha_vision_object_model_path
        return profile

    def resolve_api_key(self, profile: RuntimeProfile) -> str | None:
        if self.aisha_llm_api_key:
            return self.aisha_llm_api_key
        if profile.llm.api_key_env:
            return os.getenv(profile.llm.api_key_env)
        return None
