from __future__ import annotations

from aisha.providers.mock import MockLLMProvider
from aisha.providers.ollama import OllamaLLMProvider
from aisha.providers.openai_compatible import OpenAICompatibleLLMProvider
from aisha.settings import RuntimeProfile, Settings


def build_llm_provider(settings: Settings, profile: RuntimeProfile):
    cfg = profile.llm
    if cfg.provider == "mock":
        return MockLLMProvider()
    if cfg.provider == "ollama":
        if not cfg.base_url:
            raise ValueError("Ollama provider requires llm.base_url")
        return OllamaLLMProvider(
            model=cfg.model,
            base_url=cfg.base_url,
            think=cfg.think,
            keep_alive=cfg.keep_alive,
            options=cfg.options,
        )
    if cfg.provider == "openai-compatible":
        if not cfg.base_url:
            raise ValueError("OpenAI-compatible provider requires llm.base_url")
        return OpenAICompatibleLLMProvider(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=settings.resolve_api_key(profile),
        )
    raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
