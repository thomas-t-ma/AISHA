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

## AISHA Studio (Milestone 2)

Start Core in one terminal, with `AISHA_PROFILE=mac-m2max-96gb` set in
`services/aisha-core/.env`:

```bash
cd services/aisha-core
source .venv/bin/activate
python -m uvicorn aisha.main:app --reload --host 127.0.0.1 --port 8000
```

Then, from the repository root, start the local React/TypeScript Studio:

```bash
cd apps/aisha-studio
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Use Node.js 20.19+ or 22.12+.
Vite proxies `/v1` (including WebSockets) to local Core, so neither service
needs a public port, cloud account, API key, or permissive CORS.

Studio includes streaming chat, new/resumable local sessions, cancellation,
model/profile and health displays, per-turn TTFT/total latency, model runs
and a developer event inspector. Only committed messages are loaded after
a page refresh. A browser stores the session ID locally; AISHA's SQLite
database remains the source of truth.

This is a development UI; keep both servers on `127.0.0.1`. Authentication,
LAN access, audio and avatar rendering are not included yet.

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

## Explicit cross-session memory (Milestone 4 foundation)

AISHA has a separate local SQLite table for user-approved memories. Studio's
**Memory** panel can add, review, edit, and delete them. Memories are injected as
quoted reference data into new turns, even in a **new session**. A stored chat
transcript is not automatically a remembered fact. The first version deliberately
does **not** extract user facts automatically or call embeddings/vector search:
only memories you explicitly save are eligible for reuse.

The 12 most recently updated memories (up to 2,400 characters combined) can be
supplied to each response. Removing a memory stops it from being supplied to
future turns, but does not rewrite old conversations, database backups, or
answers already generated. The memory table and chat remain in the local,
gitignored `.aisha-data` SQLite database; do not expose this unauthenticated
single-user development server on a public interface.

To try: open Memory in Studio, add a harmless fact such as a preferred nickname,
start **New conversation**, and ask AISHA what you explicitly asked her to
remember. Edit/remove that entry and ask again in another new session.
Voice work remains isolated on `feature/voice-v1` until you have a microphone.

## Autonomous memory v1 (experimental)

Voice is still on a separate branch; this branch builds on the text-only Studio.
When Ollama is configured, Core records a **completed** conversation turn as an
episode and starts a separate local-model reflection after sending the completed
turn event. Reflections are best-effort, limited to two memories per turn; short
greetings should be ignored. They may finish after you see AISHA's answer.

Each learned belief includes a source quote from the *actual user message*,
source session/turn, status (stated, inferred, uncertain), and a revision history.
New evidence can revise a belief; AISHA's own prior output is not evidence.
In the Studio Memory panel, click Refresh to inspect newly formed beliefs and
their source quotes. You can forget a learned belief; original transcripts are
not erased. Manual notes continue working.

Set `AISHA_AUTO_MEMORY=false` in Core's `.env` if you need to disable new
autonomous reflection without deleting existing memories. The mock profile
does not launch a reflection model, so CI remains offline. The first revision
uses recent beliefs (not semantic/vector retrieval), and model classifications
are *fallible*: inspect the source and revisions before trusting conclusions.
Reflections may be interrupted if Core shuts down immediately after a reply.

## Memory integrity gate (experimental)

Automatically proposed beliefs now receive a separate, local-model evidence check
**before** AISHA writes them to SQLite. The checker assesses the entire proposed
claim against the exact quote from the current user message; a partial match
is not enough. The reflector and verifier both use the configured Ollama model
without Ollama's MLX-incompatible structured-output `format` option. Both run
*after* the visible answer; a failed check does not interrupt chat.

A checked belief/version is labeled `verified` in storage and
**Evidence checked (model-assisted)** in Studio. This means a separate LLM
judged the new claim consistent with that quote—not a mathematical guarantee of
truth. Evidence checks may reject good claims as well as bad ones; consult the
existing per-reflection rejection reason.

Historical learned beliefs are **preserved**, not silently rewritten. The schema
labels them `legacy_unchecked`; Studio and the conversation prompt disclose
that their evidence was not evaluated by this new check. In particular, review
the older composite `current_employment` example manually rather than assuming
the new verifier has retroactively fixed it. Existing explicit user-authored
notes and source episodes are untouched.

Revisions require a new source quote for the full *new* wording. Earlier source
quotes remain available in Revision history, rather than silently becoming
proof for arbitrary additional clauses in a replacement belief.
