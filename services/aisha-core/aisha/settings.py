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


class VisionProfile(BaseModel):
    provider: Literal["disabled", "local-mediapipe"] = "disabled"
    camera_index: int = Field(default=0, ge=0)
    source_id: str = "camera_front"
    poll_interval_seconds: float = Field(default=0.5, ge=0.05, le=10.0)
    num_faces: int = Field(default=2, ge=1, le=8)
    model_path: str | None = None


class RuntimeProfile(BaseModel):
    name: str
    compute: dict[str, Any]
    llm: LLMProfile
    memory: MemoryProfile = Field(default_factory=MemoryProfile)
    stt: dict[str, Any] = Field(default_factory=dict)
    tts: dict[str, Any] = Field(default_factory=dict)
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

    aisha_vision_provider: Literal["disabled", "local-mediapipe"] | None = None
    aisha_vision_camera_index: int | None = None
    aisha_vision_model_path: str | None = None

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
        if self.aisha_vision_provider is not None:
            profile.vision.provider = self.aisha_vision_provider
        if self.aisha_vision_camera_index is not None:
            profile.vision.camera_index = self.aisha_vision_camera_index
        if self.aisha_vision_model_path is not None:
            profile.vision.model_path = self.aisha_vision_model_path
        return profile

    def resolve_api_key(self, profile: RuntimeProfile) -> str | None:
        if self.aisha_llm_api_key:
            return self.aisha_llm_api_key
        if profile.llm.api_key_env:
            return os.getenv(profile.llm.api_key_env)
        return None
