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


class RuntimeProfile(BaseModel):
    name: str
    compute: dict[str, Any]
    llm: LLMProfile
    stt: dict[str, Any] = Field(default_factory=dict)
    tts: dict[str, Any] = Field(default_factory=dict)
    vision: dict[str, Any] = Field(default_factory=dict)
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
        return profile

    def resolve_api_key(self, profile: RuntimeProfile) -> str | None:
        if self.aisha_llm_api_key:
            return self.aisha_llm_api_key
        if profile.llm.api_key_env:
            return os.getenv(profile.llm.api_key_env)
        return None
