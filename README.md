# AISHA

Portable, local-first foundation for a persistent realtime AI character.

This repository is intentionally split between **AISHA logic** (version-controlled) and
**compute runtimes** (machine-local). The same core is meant to run on an Apple Silicon Mac,
an RTX 2070 desktop, or a future high-VRAM workstation.

## Architecture rule

AISHA Core does not import CUDA, Metal, MLX, or Ollama internals. It talks to typed provider
interfaces. Hardware changes should be profile/configuration changes, not rewrites.

## Current vertical slice

- FastAPI AISHA Core
- typed events and turn IDs
- SQLite conversation persistence
- versioned AISHA persona
- mock LLM provider
- local Ollama provider
- generic OpenAI-compatible provider for future local/hosted servers
- hardware/runtime profiles
- CLI chat client
- doctor script
- GitHub Actions tests using the mock backend

## Start on an Apple Silicon Mac

### 1. Clone the repository

```bash
git clone <YOUR_AISHA_REPO_URL>
cd AISHA
```

### 2. Install Python 3.12

If needed:

```bash
brew install python@3.12
```

### 3. Bootstrap AISHA Core

```bash
./infrastructure/scripts/bootstrap_mac.sh
```

### 4. First run without any LLM

Edit `services/aisha-core/.env`:

```dotenv
AISHA_PROFILE=mock
```

Then:

```bash
cd services/aisha-core
source .venv/bin/activate
pytest -q
uvicorn aisha.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd services/aisha-core
source .venv/bin/activate
python scripts/chat.py
```

### 5. Add the free local LLM

Install Ollama separately, then:

```bash
ollama pull qwen3:4b
```

Change `.env`:

```dotenv
AISHA_PROFILE=mac-apple-silicon
```

Run:

```bash
python scripts/doctor.py
uvicorn aisha.main:app --reload --host 127.0.0.1 --port 8000
```

The same `scripts/chat.py` client now talks to the local model.

## Move to the RTX 2070 later

On the desktop, clone/pull the same repo and change only:

```dotenv
AISHA_PROFILE=nvidia-2070
```

Install the machine-local NVIDIA/Ollama runtime and pull the model. The AISHA code and
stored contracts remain the same.

## Scale beyond the RTX 2070

Use `nvidia-high.yaml` as the pattern for a stronger local inference server. AISHA can point
at localhost or a LAN compute node. Large model servers, weights, caches and datasets do not
belong in Git.

See:
- `docs/COMPUTE_BACKENDS.md`
- `docs/MAC_FIRST_PLAN.md`
