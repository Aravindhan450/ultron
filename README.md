# Ultron

> Local-first AI assistant — a privacy-respecting, permission-gated AI companion that lives in your terminal.

Ultron is a Python CLI assistant that talks to local LLMs (via [Ollama](https://ollama.com)), uses tools to act on your machine, remembers facts across sessions, and asks for permission before anything risky happens. It runs on your own hardware — no cloud account, no telemetry — and your data stays on your machine unless you explicitly use a remote tool such as web search.

---

## Status

Ultron is under active development. The core is real and usable today:

| Area | Status |
|------|--------|
| Interactive CLI chat (Rich TUI) | ✅ Implemented |
| Simple agent (intent detection + tools) | ✅ Implemented |
| ReAct agent (`--agent react`) | ✅ Implemented |
| Tool library (38 tools) | ✅ Implemented |
| Long-term memory (SQLite) | ✅ Implemented |
| Local LLM via Ollama | ✅ Implemented |
| Security boundary (risk classifier + guardrails) | ✅ Implemented |
| Sandboxed execution | 🚧 Planned |
| Voice, MCP, desktop/web UI | 🚧 Planned (scaffolded but not yet built) |

## Features

- **Local by default** — Ultron talks to Ollama on your machine (`localhost:11434`). Nothing leaves your computer unless you explicitly enable a remote service.
- **Two agent modes** — `simple` (fast, deterministic intent detection with an LLM fallback) and `react` (a Thought → Action → Observation loop for multi-step tool tasks).
- **38 built-in tools** — read/write files, run shell commands (singly or in parallel batches), HTTP requests, a unified retrieval interface (`retrieve` decides between search / page fetch / connectivity check — or combines them), web search, page fetch, SQL queries (SQLite/Postgres), long-term memory, live API-schema learning, a resource monitor, personalized cross-domain memory connections, structured-output enforcement, plan preflight analysis, environmental-state debugging (failure diagnosis + dependency checks against a live environment snapshot), and inter-tool parallel processing (`run_tool_batch` runs several different tools concurrently, each gated through the security boundary, and synthesizes their results into one analysis). See [docs/retrieval.md](docs/retrieval.md), [docs/debugging.md](docs/debugging.md) and [docs/parallel-tools.md](docs/parallel-tools.md).
- **Permission first** — every risky action (shell commands, file writes, non-read-only SQL, state-changing HTTP) shows a confirmation prompt before it executes. Read-only actions run automatically.
- **Layered security** — the security boundary classifies every tool action into a risk tier (`low` → `critical`) and runs guardrails that scan for leaked credentials and PII, block unsafe URLs and path escapes, and flag dangerous shell commands before anything executes.
- **Knowledge-graph memory** — facts persist across sessions in `.ultron_memory.db` as `subject → predicate → object` triples (with a flat-fact fallback for anything that doesn't parse). Ultron can then answer deductive questions deterministically — e.g. *"what is the capital of a country that borders Germany?"* — by walking stored edges in SQL, never by guessing. See [docs/memory-graph.md](docs/memory-graph.md).
- **Multimodal vision** — *"analyze this chart.png"* sends the image to a vision-capable model, which reads diagrams, graphs, and sketches and acts on them (e.g. writes code from a visual spec). Non-vision models get a friendly `ollama pull llava` hint instead of a cryptic failure. See [docs/multimodal.md](docs/multimodal.md).
- **Well-mannered, structured replies** — every system prompt carries shared response-style guidance (polite, lead with the answer, Markdown structure), and replies get a light deterministic polish so local models stay tidy. See [docs/multimodal.md](docs/multimodal.md) and `core/intelligence/prompt_assembly.py`.
- **Real-time API schema inference** — Ultron learns API shapes from every HTTP call and from fetched OpenAPI specs. When an API changes (a field rename, a new required parameter, a type tightening, a moved endpoint), the validation error is parsed automatically: the change is remembered, the next request is corrected without being asked, and the tool output shows exactly what changed. See [docs/api-schema-inference.md](docs/api-schema-inference.md).
- **Personalized learning** — every fact you store is correlated against everything else Ultron remembers: shared topics, keywords, and curated cross-domain concept bridges (e.g. *Renaissance art* ↔ *Medici politics*) surface connections with human-readable reasons, stored as a persistent link map. Ask *"connections for medici"*, *"how is renaissance art related to the medici"*, or *"discover new connections"* — including transitive novel links between indirectly-related facts. All deterministic and grounded in stored text — no hallucination. See [docs/personalized-learning.md](docs/personalized-learning.md).
- **Structured output enforcement** — *"answer as JSON with fields name, age"*, *"in XML with elements title, body"*, or *"as a markdown table"* now carry a real guarantee: the exact schema is injected into the model prompt, then the reply is validated and deterministically repaired before it's shown — trailing commas and single quotes fixed, truncated JSON recovered to its complete prefix, unclosed XML tags balanced, table separators inserted — with `[structured]` notes listing every change, and an explicit non-conformance warning when repair is impossible (nothing is ever fabricated). Try `enforce_schema`, `schema_validate`, or `list_schemas`. See [docs/structured-output.md](docs/structured-output.md).
- **Proactive dependency identification** — multi-step requests (*"read config.py, run pytest, then POST the results"*) are planned, then **preflighted before anything runs**: every step's permission (⚡ auto / 🛡 needs approval / ⛔ blocked), missing required info, data-flow dependencies (write → read → command chains), and heavy-command warnings are listed up front. Plans with blocked steps are never offered; plans needing approval wait for one consent card instead of prompting per step mid-execution; all-auto plans run immediately with the preview shown. `plan_task` now also plans HTTP and SQL steps. Try `preflight_plan`, `analyze_dependencies`, or `list_plan_actions`. See [docs/planning.md](docs/planning.md).
- **Resource constraint awareness** — every command run is measured (wall/CPU time, peak memory) and reported, history feeds future forecasts, and a command whose forecast is heavy or critical is surfaced *before* it runs — silent auto-runs become informed confirmations, and parallel batches warn when they could spike CPU/memory. See [docs/resources.md](docs/resources.md).
- **Transparent** — you see every tool call, every approval, and every decision; everything is written to a rotating log file.
- **Polished, width-adaptive terminal UI** — Rich-based banner, Markdown rendering, action cards, and slash commands, plus a live prompt with a status toolbar (model / agent / status / security mode) that re-flows automatically when the window is resized.

## Requirements

- Python **3.11+**
- [Ollama](https://ollama.com/download) running locally with at least one model pulled:

```bash
ollama pull llama3
```

## Installation

```bash
git clone https://github.com/Aravindhan450/ultron.git
cd ultron
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e .
```

For the Postgres database tool, install the optional extra:

```bash
pip install -e ".[postgres]"
```

Verify the install:

```bash
ultron --version   # → ultron version: 0.1.0
ultron --help
```

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

All settings can also be set via environment variables prefixed with `ULTRON_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ULTRON_MODEL` | `gemini-2.5-flash` | Model name to request from Ollama — set this to a model you've actually pulled (e.g. `llama3`) or switch at runtime with `/model` |
| `ULTRON_SECURITY_MODE` | `interactive` | Security mode for the boundary: `permissive`, `interactive`, or `strict` |
| `ULTRON_WAKE_WORD` | `ultron` | Wake word for the planned voice mode |
| `ULTRON_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ULTRON_DATA_DIR` | `~/.ultron` | Where the log file (`ultron.log`) is written |
| `ULTRON_DATABASE_TYPE` | `sqlite` | Backend for the DB query tool: `sqlite` or `postgres` |
| `ULTRON_DATABASE_URL` | — | e.g. `test_db.sqlite` or a Postgres DSN |

`ULTRON_MEMORY_BACKEND` and `ULTRON_WAKE_WORD` are reserved for upcoming features and not yet enforced. `ULTRON_SECURITY_MODE` drives the security boundary's decisions (see [Security model](#security-model)); the boundary is wired into the agent tool-call flow — every tool call in both the simple and ReAct agents is routed through `boundary.check()` before execution.

> `.env` is gitignored — only `.env.example` is committed.

## Usage

### Start chatting

Make sure Ollama is running first (see [Requirements](#requirements)), then:

```bash
ultron chat
```

Launch with the ReAct agent instead:

```bash
ultron chat --agent react
```

The chat session opens a **responsive Rich UI**: an adaptive banner, Markdown-rendered responses, and a prompt with a live bottom toolbar showing the active model, agent, status, and security mode. The whole interface is terminal-width-aware — narrow the window and the banner degrades to a compact wordmark, the toolbar drops non-essential segments, and panels tighten, so nothing ever wraps or breaks. Resize mid-session and the prompt + toolbar re-flow instantly. Type naturally — *"read config.py"*, *"create notes.txt with hello in it"*, *"what did I tell you about FastAPI?"*. Risky actions render a confirmation card that you approve or reject. Read-only actions execute immediately and their results are shown.

### Slash commands (inside chat)

| Command | Action |
|---------|--------|
| `/help` | Show the command table |
| `/model` | List and switch Ollama models interactively (persists to `.env`) |
| `/agent` | Switch agent type (`simple` or `react`), or `/agent react` to switch directly |
| `/security` | Show the active security mode + per-tier policy; `/security <permissive\|interactive\|strict>` switches modes live (persists to `.env`) |
| `/clear` | Reset conversation history |
| `/reload` | Best-effort hot-reload of core modules (restart if changes don't appear) |
| `/exit`, `/quit` | Leave the chat |
| `Esc` | Cancel the current task while the model is thinking |

### Other commands

```bash
ultron --version         # Show version and exit
ultron run               # Print the active configuration and stay running (Ctrl+C to stop)
ultron logs              # Show the last 50 log lines
ultron logs -n 200       # Show more lines
ultron logs -f           # Follow the log in real time
```

## What you can ask it

| Intent | Example | Behavior |
|--------|---------|----------|
| Read a file | `read config.py` | Confirmation → file contents |
| Write a file | `create notes.txt with hello in it` | Confirmation → written; existing files require an overwrite confirmation |
| Run a command | `run pytest -v` | Confirmation → output, with test results summarized |
| Git | `show diff`, `commit this "fix bug"` | Confirmation → the mapped git command |
| Tests / lint | `run tests`, `check my code` | Confirmation → pytest / ruff, summarized |
| Web search | `search for python 3.12 release date` | Confirmation → top results |
| Fetch a page | `fetch this page https://...` | Confirmation → readable page text |
| HTTP request | `get https://api.example.com/status` | GET runs immediately; POST/PUT/DELETE require confirmation; API schema changes are learned automatically |
| API schema | `learn the api schema for http://localhost:8000` | Fetches + mines the OpenAPI spec; `what apis do you know` lists learned APIs and detected schema changes |
| Resources | `check system resources`, `how heavy is pip install` | System snapshot (CPU/load/memory) or a command's forecast; heavy commands warn before running |
| Database | `query: SELECT * FROM users` | Read-only SQL runs immediately; writes require confirmation |
| Remember | `remember that Paris is the capital of France` | Parseable sentences are stored as knowledge-graph triples; anything else is stored as a flat fact |
| Recall | `what do you know about databases` | Answered from stored facts + graph edges only (no hallucination) |
| Deduce | `what is the capital of a country that borders Germany` | Deterministic multi-hop graph traversal over stored triples |
| Multi-step | `create world.txt, write hello world in it, then read it back` | Broken into steps, planned and executed in order |

## Safety model

The chat flow is permission-gated at the UI layer:

- `run_command`, file writes, non-read-only SQL, and POST/PUT/DELETE HTTP requests all require an explicit confirmation prompt before executing. Parallel command batches (`run_parallel`) gate **every** command in the batch individually — any denial blocks the whole batch, and a batch needing confirmation is approved once, with every command listed. See [docs/parallel-commands.md](docs/parallel-commands.md).
- HTTP is restricted to `https://` and localhost — plain HTTP to external hosts is blocked.
- File tools refuse to touch paths outside the directory you launched Ultron from.
- Read-only operations (reads, SELECTs, GETs, searches) run automatically.
- Tool calls and runtime errors are written to the log; the security boundary also records every allow/confirm/deny verdict to a structured JSON-lines audit trail at `~/.ultron/security_audit.jsonl`.

## Security model

`ultron.security` provides a reusable boundary that classifies and gates every tool action:

| Risk tier | Behavior |
|-----------|----------|
| `low` | Auto-allowed (file reads, web search, memory lookups, GET, SELECT) |
| `medium` | Soft confirm (reserved — no action maps to it yet) |
| `high` | Explicit confirm (writes, state-changing commands/HTTP/SQL) |
| `critical` | Explicit confirm (system paths, destructive commands, `DROP`/`ALTER` SQL) |

Guardrails run before execution and hard-block:

- **Credential leakage** — outgoing content (file writes, memory, command strings) that looks like an API key, token, or private key is denied and redacted.
- **Unsafe URLs** — anything that is not `https://` or localhost.
- **Path escapes** — file targets outside the directory Ultron was launched from (via `ultron.security.file_policy`'s confinement check).
- **Dangerous shell patterns** — `rm -rf /`, `curl | sh`, fork bombs, and friends are escalated to `critical` (the user still has the final say).

The verdict is `allow`, `confirm`, or `deny`, decided by the tier plus `ULTRON_SECURITY_MODE` (`permissive` / `interactive` / `strict`):

```python
from ultron.security import get_boundary

verdict = get_boundary().check("run_command", "rm -rf /")
verdict.decision   # Decision.CONFIRM
verdict.tier       # RiskTier.CRITICAL
```

Every verdict is appended to a JSON-lines audit trail (`~/.ultron/security_audit.jsonl` by default) — one object per line with a UTC timestamp, tier, decision, security mode, reason, and guardrail findings. Records never contain raw content or secret snippets; `tail -f` it to watch the boundary decide in real time:

```bash
tail -f ~/.ultron/security_audit.jsonl | jq '.decision, .action_type'
```

## Project layout

```
src/ultron/
├── main.py            # Typer CLI entry point
├── core/
│   ├── agents/        # BaseAgent + the simple and react agents
│   ├── engine/        # LLM backends (Ollama implemented; cloud/vllm/mlx planned)
│   ├── tools/         # Tool registry, builtin tools, and SQLite memory
│   ├── learning/      # Real-time API schema inference (drift detection + usage prediction)
│   ├── config.py      # Settings (env vars / .env)
│   └── types.py       # ChatMessage, PendingAction, Role
├── ui/                # Rich theme & layout helpers
├── security/          # SecurityBoundary (risk classifier + allow/confirm/deny gate),
│                      #   GuardrailsEngine (secrets/PII/URL/path/command scans),
│                      #   file_policy.py (path confinement, protected paths, glob rules),
│                      #   audit.py (JSON-lines trail of every verdict); sandbox scaffold
├── permissions/       # Permission classifier + interactive confirm flow (reuses security tiers)
├── voice/             # Scaffold: wake word, STT, TTS, VAD
├── mcp/               # Scaffold: MCP client & transports
└── platform/          # Scaffold: macOS / Linux / Windows adapters
```

## Development

Install the dev tooling (pytest, ruff):

```bash
pip install -e ".[dev]"
```

```bash
# Run the test suite
python -m pytest

# Lint
ruff check .

# Add a new tool
#   1. Create a function in src/ultron/core/tools/builtin/
#   2. Register it in src/ultron/core/tools/registry.py
#   3. (Optional) teach the simple agent to detect it with a detect_*/handle_* pair
```

## Roadmap

- Voice pipeline (wake word, STT, TTS, barge-in)
- Sandboxed execution; finer-grained per-tool permission tiers
- MCP client for third-party tools
- Desktop and web UIs
- More agents (orchestrator, codeact, operative, monitor)
- Cloud model backends (vLLM, MLX, OpenAI-compatible APIs)

## Documentation

- [User guide](docs/user.md)
- [Agents guide](docs/agents.md)
- [Knowledge-graph memory design](docs/memory-graph.md)
- [API schema inference design](docs/api-schema-inference.md)
- [Resource constraint awareness](docs/resources.md)
