# Mac-first, scale-later AISHA plan

## Stage A — now on the Mac

1. Get mock AISHA Core passing locally and in GitHub Actions.
2. Run local Ollama through the provider adapter.
3. Build cancellation, model-run metadata, latency logging, and replay.
4. Build AISHA Studio against the WebSocket event contract.
5. Build memory CRUD and evaluation cases before embedding/vector complexity.
6. Add push-to-talk through an STT adapter; prefer an Apple-Silicon backend on the Mac.
7. Add local TTS through a replaceable adapter.

All of these changes are committed to GitHub.

## Stage B — RTX 2070 migration

Clone the same repository, create `.env`, switch the profile to `nvidia-2070`, install
Ollama/NVIDIA runtime locally, and point the same AISHA Core at it. No conversation,
persona, memory, or UI rewrite is expected.

## Stage C — upgraded home workstation

Clone/pull the repo, switch to `nvidia-high`, and use a stronger local inference server or
larger model. AISHA's compute clients remain the same interface. Increase requirements by
configuration/capability discovery, not branches like `if mac ... else cuda ...` scattered
through cognition code.

## Development rule

If a feature cannot be tested with `AISHA_PROFILE=mock`, separate the hardware-specific
part until the orchestration logic can be. This keeps GitHub CI useful and prevents the
project from becoming tied to whichever machine was used to write it.
