from __future__ import annotations

from aisha.perception.base import PerceptionSummary


class PerceptionPromptPolicy:
    """Whitelist transient observable facts that may reach cognition."""

    def render(self, summary: PerceptionSummary) -> str | None:
        facts: list[str] = []

        if summary.person_present:
            noun = "person" if summary.person_count == 1 else "people"
            facts.append(
                f"{summary.person_count} {noun} currently visible to the local camera."
            )

        if summary.head_approximately_frontal:
            facts.append(
                "At least one visible face is approximately oriented toward the camera."
            )

        if summary.gaze_toward_camera:
            facts.append(
                "A reliable structured gaze observation is labeled toward_camera."
            )

        if not facts:
            return None

        return (
            "\n\nTRANSIENT LOCAL PERCEPTION CONTEXT "
            "(system-generated observation, not user-authored memory):"
            "\n- "
            + "\n- ".join(facts)
            + "\nUse this only when it naturally matters to the current interaction."
            " Do not announce surveillance or describe the camera merely because this"
            " context exists. Do not treat these observations as autobiographical"
            " memory, identity evidence, emotion, intent, or other unobserved traits."
        )
