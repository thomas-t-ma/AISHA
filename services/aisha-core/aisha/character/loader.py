from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PersonaPackage:
    version: str
    prompt: str


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_persona(character_dir: Path) -> PersonaPackage:
    identity = _load_yaml(character_dir / "identity.yaml")
    style = _load_yaml(character_dir / "speaking-style.yaml")
    boundaries = _load_yaml(character_dir / "boundaries.yaml")

    versions = {identity["version"], style["version"], boundaries["version"]}
    if len(versions) != 1:
        raise ValueError(f"Persona component versions differ: {sorted(versions)}")

    prompt = "\n\n".join(
        [
            f"IDENTITY\nName: {identity['name']}\n{identity['summary']}",
            "PRINCIPLES\n- " + "\n- ".join(identity["principles"]),
            "SPEAKING STYLE\n" + "\n".join(f"{k}: {v}" for k, v in style["style"].items()),
            "BOUNDARIES\n- " + "\n- ".join(boundaries["boundaries"]),
        ]
    )
    return PersonaPackage(version=versions.pop(), prompt=prompt)
