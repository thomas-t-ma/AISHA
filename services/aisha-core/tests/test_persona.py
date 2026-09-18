from aisha.character.loader import load_persona
from aisha.settings import Settings


def test_persona_loads():
    settings = Settings(aisha_profile="mock")
    persona = load_persona(settings.character_dir)
    assert persona.version == "0.1.0"
    assert "AISHA" in persona.prompt
    assert "Do not invent memories" in persona.prompt
