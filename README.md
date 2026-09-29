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

### Quotation punctuation and atomic follow-ups

The reflector may straighten a typographic apostrophe when quoting a user:
`doesn’t` versus `doesn't`. Core now resolves **only** these one-character
quotation-punctuation differences against the actual user message, then passes
the recovered **original exact quote** into the evidence checker and database.
It does not fuzzy-match words, spelling, case, or changed meaning. Existing
failed reflections are not replayed or backfilled automatically.

The extraction prompt also distinguishes new, narrow facts (e.g. patient
contact at the current job) from wholesale revisions of a legacy composite
biography. The legacy entry itself is preserved, still marked unchecked.

### Bounded recovery from composite revision rejection

If the reflection model proposes a revision that a separate evidence checker
rejects because the current quote does not support the whole replacement claim,
Core attempts **one** atomic reformulation as a **new** topic. It is never an
automatic rewrite of the original belief. The retry may return no candidate,
and its new text, original-source substring, fresh topic key, and evidence
status are checked again before writing.

Studio retains the original rejection alongside a separate recovery outcome
(stored, rejected, no candidate, or failed). A recovered belief is counted as
saved, while the original rejected proposal remains counted as rejected.
This does not backfill previously failed reflections or guarantee that the
local model will always find a useful independent memory.


## Contextual recall v1 (experimental)

AISHA no longer injects every recent learned belief into every turn. Before a
reply, Core now considers up to 50 learned beliefs, excludes every
`legacy_unchecked` entry, and selects at most four **verified** beliefs whose
topic/text/open question shares substantive terms with the current user
message. The selected topics are recorded on the `aisha.turn.started` event as
`memory_recall_topics` for debugging.

This first retrieval pass intentionally favors precision over recall: it is a
small local lexical ranker, not embeddings or another pre-response LLM call.
That keeps first-token latency unchanged and prevents unrelated memories from
being sprayed into the prompt. It can miss semantic relationships expressed
with completely different vocabulary; semantic/vector retrieval is a future
upgrade.

When recalled, verified beliefs are presented as fallible continuity context.
The chat model is told to let genuinely relevant history influence the response
naturally, not to announce memory access, recite stored facts, or force
follow-up questions. Unresolved questions may be revisited only when the
current message already engages that topic.

Manual user-authored memories retain their existing behavior. Legacy learned
beliefs remain visible in Studio and their revision history is unchanged, but
they are no longer active conversational knowledge until separately reconciled.


### Studio recall observability

Studio's event log now hides per-token `aisha.assistant.text_delta` records from
the default event list so lifecycle events such as `turn.started` remain visible.
The selected model-run detail also shows the recalled memory count and topic keys
directly. The underlying delta events are still persisted in SQLite.

The event API now returns the newest requested records (restored to chronological
order) rather than the oldest records in a long session. Studio asks for the
latest 2,000 events, preventing normal streaming traffic from starving recent
turn diagnostics.


### Recall match diagnostics

Each `aisha.turn.started` event now includes `memory_recall_details` for every
retrieved learned belief: the retrieval method, score, and normalized lexical
tokens that caused the match. Studio exposes the same details in Model runs.
This is diagnostic metadata only; it is not added to AISHA's conversation
prompt.

The current lexical retriever cannot infer semantic similarity without shared
normalized tokens. A paraphrase such as "spending my day away from the people
I'm supposed to be helping" is therefore a deliberate zero-recall baseline for
the verified `job_patient_interaction_level` memory unless the actual prompt or
stored belief contains overlapping terms.


### Lexical recall precision

The lexical ranker now ignores low-information conversational overlap such as
`more`, `think`, `something`, and `getting` when deciding whether a belief
is relevant. It also no longer strips `-ing` / `-ed` suffixes naively; that
normalizer had produced artifacts such as `something -> someth` and
`getting -> gett`. Simple plural normalization remains.

Recall diagnostics distinguish the tokens that actually justified selection
from low-information overlap that was ignored. A belief is not retrieved when
its only shared words with the current turn are low-information terms.


## Test conversation mode (experimental)

Studio can mark the current session as a **Test conversation**. This is a
memory sandbox for synthetic prompts and regression tests:

- existing verified memories remain eligible for contextual recall;
- normal chat messages and model runs still persist in that test session;
- autonomous memory episodes are **not** created for test-mode turns;
- post-turn reflection is **not** scheduled, so those turns cannot add or
  revise learned beliefs;
- each turn-start event records `memory_mode: "test"` and
  `memory_learning_enabled: false`;
- Core persists an `aisha.memory.learning_skipped` event with reason
  `test_mode` after the completed turn.

The setting is stored per session and defaults to `normal`. Switching the
same session back to Normal memory re-enables learning for subsequent turns.
The Studio composer displays a persistent test-mode warning while learning is
disabled.

This mode is designed for development prompts that are not necessarily true
about the user. It does not retroactively remove beliefs created by earlier
normal-mode tests; those should be forgotten manually if they were synthetic.


## Hybrid semantic recall v1 (experimental)

The M2 Max profile can now augment high-precision lexical memory recall with a
small local embedding model. Lexical retrieval always runs first. Semantic
retrieval only fills unused recall slots, cannot duplicate an already-selected
lexical belief, and still considers only `verified` learned beliefs.

The M2 Max profile now uses the stronger 4B Qwen3 embedder:

```bash
ollama pull qwen3-embedding:4b
```

Core calls Ollama's current `/api/embed` endpoint with batched inputs. Following
Qwen3-Embedding's retrieval guidance, the **query** is prefixed with a
task-specific instruction while learned-memory documents remain unprefixed.
The instruction asks the model to retrieve a prior personal memory that is
directly useful for the current message, including the same underlying
situation phrased differently, while avoiding merely topical associations.

Belief vectors are cached in-process by belief ID/revision/content; after the
first comparison, subsequent turns normally embed only the new instructed
query. The `0.72` semantic threshold remains for the older semantic-only
compatibility path, but production hybrid recall no longer auto-accepts a
memory based on cosine score alone. The `0.30` floor controls semantic
candidate generation; the final relevance gate decides prompt injection.

Recall diagnostics retain the retrieval method. A semantic match appears as
`method: "semantic"` with its cosine score and no lexical match tokens. Studio
also shows the configured embedding model and any semantic-retrieval error.
For threshold tuning, the turn status records the top semantic candidates even
when they are not selected.

Hybrid recall now separates **candidate generation** from **final selection**:

- the lexical ranker proposes high-overlap candidates;
- the embedding model proposes semantic candidates at or above the `0.30`
  candidate floor;
- duplicate lexical/semantic candidates are merged;
- the union is sent in one batch to the strict relevance gate;
- only memories accepted by that final gate are injected into AISHA's prompt.

Lexical overlap is therefore evidence for candidacy, not automatic permission
to recall. This specifically prevents generic questions such as "How does
medical-school accreditation work?" from recalling a personal medical-school
goal merely because the words overlap.

The relevance gate judges whether the CURRENT message is actually about the
same underlying personal situation, decision, preference, goal, relationship,
problem, or unresolved thread. It is explicitly told to reject broad topical
adjacency such as keyboard -> general computer preferences, healthcare AI ->
every healthcare memory, or a general factual question -> a similarly named
personal goal. It does not re-judge memory truth/evidence. If the gate fails or
returns malformed output, candidates are rejected while chat continues normally.

Low-similarity semantic items still remain cheap because embeddings below
`0.30` never enter the final gate unless they were independently proposed by
the lexical stage.


### Recall latency diagnostics

Each `aisha.turn.started` payload records:

- `memory_retrieval_ms`: total learned-memory retrieval time before generation;
- `lexical_retrieval_ms`: deterministic lexical stage time;
- `semantic_retrieval_ms`: embedding plus any second-stage relevance-gate time.

Studio shows the same values under Memory Recall. This makes pre-generation
memory cost visible separately from the provider's first-token latency. When a
semantic memory is accepted by the second-stage gate, its `reranker_reason`
is also preserved in the per-memory recall diagnostics.

Semantic retrieval fails open: if embedding generation is unavailable, normal
chat and lexical recall continue. A missing embedding model disables semantic
recall for that Core process after the first 404 rather than retrying on every
turn; pull the model and restart Core to re-enable it.

Use **Test conversation** mode when tuning thresholds or trying synthetic
paraphrases. Recall remains active while those test turns are prevented from
creating or revising learned beliefs.


### Synthetic multi-memory benchmark

A standalone benchmark exercises the production lexical + embedding + relevance
gate stack against a synthetic 12-memory bank. It does **not** open AISHA's
SQLite database and does not read or write personal memories.

The suite currently includes direct lexical matches, semantic paraphrases,
same-thread updates/contradictions, a two-memory turn, hard thematic negatives,
and unrelated negatives. It reports exact-case accuracy, micro precision/recall,
negative-control accuracy, final retrieval methods, candidate sources
(lexical/semantic/both), gate decisions, and retrieval latency.

With Ollama running and the profile models already pulled:

```bash
cd services/aisha-core
source .venv/bin/activate
python scripts/benchmark_memory_recall.py \
  --output /tmp/aisha-memory-recall-benchmark.json
```

Run one case while debugging:

```bash
python scripts/benchmark_memory_recall.py --case patient_contact_paraphrase
```


The final relevance-gate model can be benchmarked independently from AISHA's
conversation model:

```bash
python scripts/benchmark_memory_recall.py \
  --reranker-model qwen3.5:4b \
  --output /tmp/aisha-memory-recall-4b.json
```

This override is benchmark-only and does not change the runtime profile. The
production gate continues to fall back to the main conversation model unless
`memory.semantic_relevance_model` is configured explicitly.

The benchmark also reports the gate's Ollama timings (load, prompt evaluation,
generation, prompt tokens, and output tokens). Gate reasons are intentionally
limited to terse diagnostics so the judge spends tokens deciding rather than
writing explanations.


Compare multiple relevance-gate models on the exact same cases:

```bash
python scripts/compare_memory_rerankers.py \
  --suite smoke \
  --model qwen3.5:4b \
  --model qwen3.5:9b-mlx \
  --model qwen3.5:35b-mlx
```


On Apple Silicon, AISHA can also benchmark a dedicated MLX cross-encoder
reranker instead of a generative chat model. This is an optional dependency so
Linux/Core CI remains hardware agnostic.

Install the Mac-only extra:

```bash
cd services/aisha-core
source .venv/bin/activate
pip install -e '.[dev,mac-memory]'
```

Then run the dedicated Qwen3 reranker smoke suite:

```bash
python scripts/benchmark_mlx_memory_reranker.py \
  --suite smoke \
  --model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --threshold 0.50
```

The model is lazy-loaded through `mlx-lm`; the first run may download the
~331 MB model from Hugging Face. The implementation follows the model card's
yes/no-logit scoring recipe and uses a custom AISHA instruction that defines
relevance as the same active personal proposition rather than broad topical
similarity. No explanatory text is generated.

If smoke clears all three cases, run:

```bash
python scripts/benchmark_mlx_memory_reranker.py \
  --suite full \
  --model mlx-community/Qwen3-Reranker-0.6B-4bit \
  --threshold 0.50 \
  --output /tmp/aisha-memory-reranker-mlx.json
```

This path is benchmark-only until it proves quality parity with the 35B
generative gate.

The smoke suite contains the three cases that currently best separate gate
quality: an indirect patient-contact paraphrase, a genuine two-memory work +
volunteering turn, and a general medical-school question that must not trigger
personal continuity. Use `--suite full` after a model clears smoke. The
comparison reports exact accuracy, precision, recall, negative-control
accuracy, median retrieval/gate latency, max gate latency, and concise failure
diagnostics for each model.

The benchmark is intentionally synthetic so threshold/reranker experiments
cannot contaminate AISHA's real autobiographical memory. A failure is not
automatically a reason to loosen thresholds; inspect whether it was a lexical
false positive, embedding miss, candidate-floor miss, or relevance-gate error.
