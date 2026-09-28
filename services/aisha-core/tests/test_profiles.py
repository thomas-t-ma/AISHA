from aisha.settings import Settings


def test_profiles_load():
    for name in [
        "mock",
        "mac-apple-silicon",
        "mac-m2max-96gb",
        "nvidia-2070",
        "nvidia-high",
    ]:
        profile = Settings(aisha_profile=name).load_profile()
        assert profile.name == name
        assert profile.llm.provider
        assert profile.llm.model


def test_m2max_profile_preloads_conversation_model():
    profile = Settings(aisha_profile="mac-m2max-96gb").load_profile()

    assert profile.llm.preload is True
    assert profile.llm.think is False
    assert profile.llm.keep_alive == "30m"
    assert profile.memory.semantic_recall is True
    assert profile.memory.embedding_model == "qwen3-embedding:4b"
    assert profile.memory.semantic_threshold == 0.72
    assert profile.memory.semantic_candidate_floor == 0.30
    assert profile.memory.semantic_limit == 2
    assert profile.memory.semantic_relevance_gate is True
    assert profile.memory.semantic_relevance_model is None
    assert profile.memory.semantic_relevance_base_url is None
    assert profile.memory.semantic_relevance_keep_alive is None
