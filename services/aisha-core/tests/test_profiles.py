from aisha.settings import Settings


def test_profiles_load():
    for name in ["mock", "mac-apple-silicon", "nvidia-2070", "nvidia-high"]:
        profile = Settings(aisha_profile=name).load_profile()
        assert profile.name == name
        assert profile.llm.provider
        assert profile.llm.model
