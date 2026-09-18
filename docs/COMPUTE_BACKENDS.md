# AISHA compute backends

AISHA Core is deliberately hardware-agnostic. Hardware acceleration belongs behind
provider/service boundaries, never in the orchestrator.

## What belongs in Git

Commit:
- persona and behavior configuration;
- event/turn/tool/action contracts;
- orchestration and cancellation logic;
- memory policy and retrieval logic;
- provider adapters;
- AISHA Studio source code;
- Unity scripts/configuration (later);
- evaluations and small anonymized fixtures;
- hardware profiles and deployment scripts;
- database migrations;
- tests.

Do not commit:
- model weights;
- Ollama/MLX/CUDA installations;
- local databases;
- raw audio/video;
- generated caches;
- secrets/API keys;
- large checkpoints or Gaussian captures.

## Target progression

### Apple Silicon Mac

Use AISHA Core + Ollama as the first development environment. Ollama is external to
Git; only the adapter and profile are committed. Apple-specific STT can later use an
`mlx-whisper` adapter without changing the orchestrator.

### RTX 2070

Use the same repo and select `nvidia-2070`. The default local model remains modest so
8 GB VRAM is not assumed to host every neural subsystem simultaneously. NVIDIA-specific
STT/TTS implementations remain adapters.

### Future high-VRAM workstation

Select `nvidia-high`, point AISHA at a stronger local inference server, and change the
model name. The rest of AISHA is unchanged. A server may live on localhost or another
machine on the LAN.

### Hosted/higher-budget inference

Use `openai-compatible` or a dedicated hosted-provider adapter. Memory, persona,
conversation history, tools, embodiment, and data remain owned by AISHA.

## Compute topology rule

Every heavy backend is an endpoint, not a dependency of AISHA cognition:

    AISHA Core -> LLM endpoint
               -> STT endpoint/adapter
               -> TTS endpoint/adapter
               -> Vision endpoint/adapter

Initially they can all be localhost. Later they can be distributed across machines.
