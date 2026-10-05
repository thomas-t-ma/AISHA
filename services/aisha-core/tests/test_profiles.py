from aisha.settings import Settings


def test_profiles_load():
    for name in [
        "mock",
        "mac-apple-silicon",
        "mac-m2max-96gb",
        "nvidia-2070",
        "nvidia-5080",
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



def test_vision_profile_defaults_disabled():
    profile = Settings(aisha_profile="mock").load_profile()

    assert profile.vision.provider == "disabled"
    assert profile.vision.camera_index == 0
    assert profile.vision.source_id == "camera_front"
    assert profile.vision.poll_interval_seconds == 0.5
    assert profile.vision.num_faces == 2
    assert profile.vision.model_path is None


def test_vision_profile_environment_overrides():
    settings = Settings(
        aisha_profile="mock",
        aisha_vision_provider="local-mediapipe",
        aisha_vision_camera_index=2,
        aisha_vision_model_path="models/custom_face_landmarker.task",
    )
    profile = settings.load_profile()

    assert profile.vision.provider == "local-mediapipe"
    assert profile.vision.camera_index == 2
    assert profile.vision.model_path == "models/custom_face_landmarker.task"



def test_tts_profile_defaults_disabled():
    profile = Settings(aisha_profile="mock").load_profile()

    assert profile.tts.provider == "disabled"
    assert profile.tts.voice == "af_heart"
    assert profile.tts.speed == 1.0
    assert profile.tts.language == "en-us"
    assert profile.tts.model_path is None
    assert profile.tts.voices_path is None


def test_tts_profile_environment_overrides():
    settings = Settings(
        aisha_profile="mock",
        aisha_tts_provider="local-kokoro",
        aisha_tts_voice="af_sarah",
        aisha_tts_model_path="models/tts/custom.onnx",
        aisha_tts_voices_path="models/tts/custom.bin",
    )
    profile = settings.load_profile()

    assert profile.tts.provider == "local-kokoro"
    assert profile.tts.voice == "af_sarah"
    assert profile.tts.model_path == "models/tts/custom.onnx"
    assert profile.tts.voices_path == "models/tts/custom.bin"
