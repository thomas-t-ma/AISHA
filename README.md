# AISHA

Portable, local-first foundation for a persistent realtime AI character.

AISHA logic lives in Git; model runtimes, weights, private conversation data, recordings,
and machine-specific state do not. The same core is intended to run on Apple Silicon,
an RTX 2070 desktop, or a future high-VRAM workstation.

## Architecture rule

AISHA Core does not import CUDA, Metal, MLX, or Ollama internals. It talks to typed
provider interfaces. Hardware changes are profile/configuration changes, not cognition
rewrites.

## Current vertical slice

- FastAPI AISHA Core
- typed session / turn / event contracts
- SQLite conversation persistence
- versioned persona package
- mock LLM provider for tests/CI
- local Ollama provider
- generic OpenAI-compatible provider for future local/hosted servers
- runtime profiles
- streaming WebSocket conversation
- CLI chat client
- doctor script

## Mac development machine

Current primary development hardware:

- Apple M2 Max
- 96 GB unified memory
- Ollama
- profile: `mac-m2max-96gb`
- default local model: `qwen3.5:35b-mlx`

The model runtime is external to Git.

### Bootstrap

```bash
./infrastructure/scripts/bootstrap_mac.sh
```

The safe default is the mock backend. Test it first:

```bash
cd services/aisha-core
source .venv/bin/activate
pytest -q
```

Then pull the Mac model:

```bash
ollama pull qwen3.5:35b-mlx
```

Edit `services/aisha-core/.env`:

```dotenv
AISHA_PROFILE=mac-m2max-96gb
```

Verify the machine/runtime:

```bash
python scripts/doctor.py
```

Start AISHA Core:

```bash
uvicorn aisha.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd services/aisha-core
source .venv/bin/activate
python scripts/chat.py
```

## Moving between machines

GitHub is the source of truth for AISHA code.

At the start of a session:

```bash
git pull --rebase
```

At the end of a stable change:

```bash
git add .
git commit -m "Describe the AISHA change"
git push
```

The RTX 2070 machine uses the same repository with `AISHA_PROFILE=nvidia-2070`.
A future high-VRAM workstation can use `nvidia-high` or another checked-in profile.

## What never belongs in Git

- `.env`
- virtual environments
- Python caches
- SQLite runtime databases
- Ollama/model weights
- raw audio/video
- private conversation/memory datasets
- large checkpoints or Gaussian captures

See `docs/COMPUTE_BACKENDS.md` and `docs/MAC_FIRST_PLAN.md`.
