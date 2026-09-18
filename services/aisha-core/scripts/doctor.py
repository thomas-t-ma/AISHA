from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess

import httpx

from aisha.settings import Settings


def command_version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        text = (result.stdout or result.stderr).strip()
        return text.splitlines()[0] if text else "installed"
    except Exception:
        return "installed (version unavailable)"


async def main() -> None:
    settings = Settings()
    profile = settings.load_profile()
    print("AISHA doctor")
    print(f"OS:       {platform.platform()}")
    print(f"Machine:  {platform.machine()}")
    print(f"Profile:  {profile.name}")
    print(f"Data dir: {settings.data_dir}")
    print(f"Ollama:   {command_version(['ollama', '--version']) or 'not found'}")

    if profile.llm.provider == "ollama" and profile.llm.base_url:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{profile.llm.base_url.rstrip('/')}/api/tags")
                response.raise_for_status()
                models = [m.get("name") for m in response.json().get("models", [])]
                print(f"Runtime:  reachable ({len(models)} local model(s))")
                print(f"Model:    {profile.llm.model} {'✓' if any(str(m).startswith(profile.llm.model) for m in models) else '(not pulled yet)'}")
        except Exception as exc:
            print(f"Runtime:  unavailable ({exc})")


if __name__ == "__main__":
    asyncio.run(main())
