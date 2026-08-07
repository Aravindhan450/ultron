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
| Tool library (10 tools) | ✅ Implemented |
| Long-term memory (SQLite) | ✅ Implemented |
| Local LLM via Ollama | ✅ Implemented |
| Security boundary (risk classifier + guardrails) | ✅ Implemented |
| Sandboxed execution | 🚧 Planned |
| Voice, MCP, desktop/web UI | 🚧 Planned (scaffolded but not yet built) |

## Features

- **Local by default** — Ultron talks to Ollama on your machine (`localhost:11434`). Nothing leaves your computer unless you explicitly enable a remote service.
- **Two agent modes** — `simple` (fast, deterministic intent detection with an LLM fallback) and `react` (a Thought → Action → Observation loop for multi-step tool tasks).
- **10 built-in tools** — read/write files, run shell commands, HTTP requests, web search, page fetch, SQL queries (SQLite/Postgres), and long-term memory.
- **Permission first** — every risky action (shell commands, file writes, non-read-only SQL, state-changing HTTP) shows a confirmation prompt before it executes. Read-only actions run automatically.
- **Layered security** — the security boundary classifies every tool action into a risk tier (`low` → `critical`) and runs guardrails that scan for leaked credentials and PII, block unsafe URLs and path escapes, and flag dangerous shell commands before anything executes.
- **Long-term memory** — facts persist across sessions in `.ultron_memory.db` (created in the directory you launch Ultron from).
- **Transparent** — you see every tool call, every approval, and every decision; everything is written to a rotating log file.
- **Polished terminal UI** — Rich-based banner, status bar, Markdown rendering, action cards, and slash commands.

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

The chat session opens a Rich TUI: banner, status bar, and a prompt. Type naturally — *"read config.py"*, *"create notes.txt with hello in it"*, *"what did I tell you about FastAPI?"*. Risky actions render a confirmation card that you approve or reject. Read-only actions execute immediately and their results are shown.

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
| HTTP request | `get https://api.example.com/status` | GET runs immediately; POST/PUT/DELETE require confirmation |
| Database | `query: SELECT * FROM users` | Read-only SQL runs immediately; writes require confirmation |
| Remember | `remember that my favorite color is blue` | Stored in long-term memory |
| Recall | `what do you know about databases` | Answered from stored facts only (no hallucination) |
| Multi-step | `create world.txt, write hello world in it, then read it back` | Broken into steps, planned and executed in order |

## Safety model

The chat flow is permission-gated at the UI layer:

- `run_command`, file writes, non-read-only SQL, and POST/PUT/DELETE HTTP requests all require an explicit confirmation prompt before executing.
- HTTP is restricted to `https://` and localhost — plain HTTP to external hosts is blocked.
- File tools refuse to touch paths outside the directory you launched Ultron from.
- Read-only operations (reads, SELECTs, GETs, searches) run automatically.
- Tool calls and runtime errors are written to the log; the security boundary also records every allow/confirm/deny verdict.

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
- **Path escapes** — file targets outside the directory Ultron was launched from.
- **Dangerous shell patterns** — `rm -rf /`, `curl | sh`, fork bombs, and friends are escalated to `critical` (the user still has the final say).

The verdict is `allow`, `confirm`, or `deny`, decided by the tier plus `ULTRON_SECURITY_MODE` (`permissive` / `interactive` / `strict`):

```python
from ultron.security import get_boundary

verdict = get_boundary().check("run_command", "rm -rf /")
verdict.decision   # Decision.CONFIRM
verdict.tier       # RiskTier.CRITICAL
```

## Project layout

```
src/ultron/
├── main.py            # Typer CLI entry point
├── core/
│   ├── agents/        # BaseAgent + the simple and react agents
│   ├── engine/        # LLM backends (Ollama implemented; cloud/vllm/mlx planned)
│   ├── tools/         # Tool registry, builtin tools, and SQLite memory
│   ├── config.py      # Settings (env vars / .env)
│   └── types.py       # ChatMessage, PendingAction, Role
├── ui/                # Rich theme & layout helpers
├── security/          # SecurityBoundary (risk classifier + allow/confirm/deny gate),
│                      #   GuardrailsEngine (secrets/PII/URL/path/command scans); sandbox scaffold
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
