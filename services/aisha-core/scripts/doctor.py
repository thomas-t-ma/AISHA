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
    except (OSError, subprocess.SubprocessError):
        return "installed (version unavailable)"


async def main() -> None:
    settings = Settings()
    profile = settings.load_profile()
    print("AISHA doctor")
    print(f"OS:         {platform.platform()}")
    print(f"Machine:    {platform.machine()}")
    print(f"Profile:    {profile.name}")
    print(f"Data dir:   {settings.data_dir}")
    print(f"Ollama:     {command_version(['ollama', '--version']) or 'not found'}")
    print(f"Think:      {profile.llm.think}")
    print(f"Keep alive: {profile.llm.keep_alive}")
    if profile.llm.options:
        print(f"LLM options:{profile.llm.options}")

    if profile.llm.provider == "ollama" and profile.llm.base_url:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{profile.llm.base_url.rstrip('/')}/api/tags")
                response.raise_for_status()
                models = [m.get("name") for m in response.json().get("models", [])]
                print(f"Runtime:    reachable ({len(models)} local model(s))")
                installed = any(str(model).startswith(profile.llm.model) for model in models)
                print(f"Model:      {profile.llm.model} {'✓' if installed else '(not pulled yet)'}")
        except (httpx.HTTPError, ValueError) as exc:
            print(f"Runtime:    unavailable ({exc})")


if __name__ == "__main__":
    asyncio.run(main())
