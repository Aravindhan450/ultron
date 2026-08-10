# ULTRON — Complete Project Context & AI Handoff Document

> **Purpose of this file:** This document is the single entry point for any AI (or
> human) receiving this package. Read this first, then `README.md`, then the
> per-topic specs in `docs/`, then the source. It explains **what Ultron is**,
> **why it exists**, **how it is built**, **what it can do**, and **how every
> piece wires together end to end** — plus the author's intent for where this
> project is going.

---

## 1. What Ultron Is

**Ultron is a local-first, permission-gated, tool-using AI assistant that lives
in your terminal.** It is a Python CLI (`ultron chat`) that:

- talks to **local LLMs** (Ollama by default — nothing leaves your machine unless
  you explicitly use a remote tool like web search),
- **uses 38 tools** to act on your machine (files, shell, HTTP, SQL, web, memory),
- **remembers facts across sessions** as a knowledge graph (`subject → predicate
  → object` triples) and can answer deductive questions deterministically,
- **asks permission before anything risky** runs (a risk classifier + guardrails
  gate every tool call),
- renders a **polished, width-adaptive terminal UI** (Claude Code-inspired) that
  re-flows perfectly on window resize.

The project name and aesthetic (ASCII "ULTRON" logo, molten-flame palette
`#E73F1E → #FB6C00 → #F9B637 → #FFDD9C`) evoke a "JARVIS-style" personal
assistant: capable, proactive, transparent, and privacy-first.

---

## 2. The Intent (what the author is building toward)

The author's stated direction, accumulated across the development sessions:

1. **Production-grade quality.** Every feature must be "production grade and work
   perfectly" — each subsystem ships with unit tests, lint-clean code, and a
   stress/audit pass.
2. **A self-improving assistant**, not a chatbot. The feature roadmap is a series
   of "intelligence upgrades", nearly all of which are **implemented**:
   - **Knowledge-graph memory with deductive reasoning** (triples → multi-hop SQL
     traversal, e.g. "capital of a country that borders Germany").
   - **Parallel execution** — multiple commands (`run_parallel`) and multiple
     *different* tools (`run_tool_batch`) run concurrently and synthesize results.
   - **Unified retrieval** — one `retrieve` tool decides between web search, page
     fetch, and connectivity check (or combines them) instead of guessing.
   - **Real-time API schema inference** — learns API shapes from calls/specs and
     auto-corrects when an API drifts (renamed fields, new required params, etc.).
   - **Resource constraint awareness** — every command is measured (wall/CPU/peak
     memory), forecast against history, and heavy commands warn **before** running.
   - **Personalized learning** — stored facts are correlated across domains via a
     persistent connection map with human-readable reasons and novel transitive
     links.
   - **Structured output enforcement** — JSON / XML / Markdown schemas are
     validated and deterministically repaired (never fabricated) before display.
   - **Proactive dependency identification** — multi-step plans are preflighted
     before anything runs: permissions, missing info, and data-flow dependencies
     are listed up front; blocked plans are never offered.
   - **Code debugging with environmental state** — failures are diagnosed against
     a live environment snapshot (OS, Python, declared deps vs. installed).
   - **Multimodal input** — vision-capable models analyze uploaded images/charts.
   - **Well-mannered, structured replies** — shared response-style guidance is
     injected into every system prompt and replies are deterministically polished.
3. **UI evolution** — the CLI was deliberately redesigned to a minimal, premium,
   Claude Code-like aesthetic: borderless conversation flow, boxed responses with
   an `ULTRON` header, muted meta text, a single flame-orange accent, and a
   resize-reflow system that re-renders the whole transcript at the live width.
4. **Roadmap (still planned)** — voice pipeline (wake word/STT/TTS/VAD/barge-in),
   MCP client, sandboxed execution, desktop/web UIs, more agent types
   (orchestrator, codeact, operative, monitor), and more model backends (vLLM,
   MLX, llama.cpp, cloud). These are scaffolded but not yet built.

---

## 3. How to Read This Package (for an AI)

| Order | File(s) | What you get |
|---|---|---|
| 1 | `PROJECT_CONTEXT.md` (this file) | Vision, intent, architecture, end-to-end flow |
| 2 | `README.md` | User-facing feature summary, install, usage, security model |
| 3 | `AGENTS.md` | Contributor / code-generation guidelines (conventions, commands, structure) |
| 4 | `docs/*.md` | Per-feature **specs** (14 documents) |
| 5 | `src/ultron/**` | The implementation (see §5 module map) |
| 6 | `tests/` + `_stress_audit.py` + `_reflow_e2e.py` | How behavior is verified |
| 7 | `pyproject.toml`, `.env.example`, `configs/` | Build, config, and model/tool policy |

---

## 4. Project Structure (live)

```
ultron/
├── pyproject.toml            # hatchling build; deps; [project.scripts] ultron = ultron.main:app
├── README.md                 # user-facing docs
├── .env.example              # config template (ULTRON_* vars)
├── configs/
│   ├── models.yaml           # ollama / mlx / llama_cpp runtime config
│   └── security.yaml         # risk tiers + per-mode policies (permissive/interactive/strict)
├── docs/                     # 14 spec documents (memory-graph, api-schema, planning, …)
├── scripts/bootstrap.sh
├── src/ultron/
│   ├── main.py               # Typer CLI: run / chat (--agent) / logs; slash commands; async_chat loop
│   ├── core/
│   │   ├── agents/           # base.py, simple.py, react.py (implemented);
│   │   │                     #   orchestrator.py, codeact.py, operative.py, monitor.py (scaffolds);
│   │   │                     #   security.py (shared SecurityBoundary accessor)
│   │   ├── engine/           # ollama.py (implemented); llama_cpp/vllm/mlx/cloud.py (scaffolds); base.py
│   │   ├── tools/            # registry.py (38 tools + schema gen); builtin/ (file_reader, file_writer,
│   │   │                     #   command_runner, database, http_client, web_search, retrieval);
│   │   │                     #   memory/ (sqlite, faiss, hybrid, graph); resource_monitor.py; paths.py
│   │   ├── intelligence/     # prompt_assembly.py, structured_output.py, parallel_tools.py,
│   │   │                     #   planning.py, debug_context.py, hardware_aware.py, model_catalog.py
│   │   ├── learning/         # api_schema.py, associations.py (implemented);
│   │   │                     #   traces.py, spec_search.py, router_policies.py (scaffolds)
│   │   ├── config.py         # Settings (pydantic-settings, ULTRON_* env / .env)
│   │   ├── types.py          # ChatMessage, Role, PendingAction, truncate_history
│   │   └── state.py          # CLIState (model, cwd, version, status)
│   ├── ui/
│   │   ├── theme.py          # Palette + UI render helpers (banner/logo, response panel, help table, …)
│   │   ├── session.py        # prompt_toolkit ChatSession + width-adaptive bottom toolbar
│   │   └── responsive.py     # ResizeReflow — re-renders the whole transcript on resize
│   ├── security/             # boundary.py (risk classifier), guardrails.py, audit.py (JSON-lines),
│   │   │                     #   file_policy.py, models.py, scanners/ (secret.py, pii.py),
│   │   │                     #   sandbox/ (container/wasm/agent — scaffolds)
│   ├── permissions/          # classifier.py, confirm.py, tiers.py, audit.py (reuses security tiers)
│   ├── voice/  mcp/  platform/  ui/web  ui/desktop  ui/api  # scaffolds (planned features)
│   └── __init__.py           # __version__
├── tests/                    # 24 pytest files covering every subsystem
├── _stress_audit.py          # 29-check stress harness (CI + manual)
├── _reflow_e2e.py            # pty end-to-end resize-reflow harness (CI)
├── AnalysisReport.md         # latest stress-test & audit report
└── .github/workflows/ci.yml  # CI: test, stress, reflow-e2e jobs
```

> Note: `src/ultron/ui/layout.py` and the old Rich **Live/Layout** TUI were
> **removed** in the cleanup — the chat uses simple sequential printing, and a
> resize watcher (`ui/responsive.py`) replays recorded prints at the live width.

---

## 5. Architecture & End-to-End Integration

### 5.1 Request flow (chat)

```
User types "read config.py"
   │
   ▼
ultron chat (main.py async_chat)
   │  CLIState created; ChatSession (prompt_toolkit) with live bottom toolbar
   │  ResizeReflow starts (records every console.print; re-renders on resize)
   ▼
prompt_async() → input echoed as "❯ read config.py"
   │
   ├─ slash command? ── /help /model /agent /security /memory /clear /reload /exit
   │
   ▼
agent.run(input, history)
   │
   ├─ SimpleAgent: deterministic detect_*() intents → tool call (LLM fallback)
   └─ ReActAgent: LLM Thought→Action→Observation loop (JSON tool calls, max_iterations cap)
   │
   ▼
Every tool call passes SecurityBoundary.check(tool, args)
   │  risk classifier → tier (low/medium/high/critical)
   │  guardrails → secrets / PII / unsafe URL / path escape / dangerous shell
   │  verdict: ALLOW → auto-run | CONFIRM → PendingAction → questionary prompt
   │           DENY → hard-blocked (never offered)
   │  every verdict → security/audit.py JSON-lines trail
   ▼
Tool executes via core/tools/registry.py (38 tools)
   │  result rendered: tool execution chip (✻ read_file …) or boxed response panel
   ▼
Response: UI.render_response → ╭─ ULTRON ─╮ boxed Markdown reply
   │  history appended; every print recorded for resize reflow
   ▼
Loop back to prompt
```

### 5.2 Agent model

- **`BaseAgent`** (`core/agents/base.py`) defines `run()`. `SimpleAgent` and
  `ReActAgent` implement it; both are selectable via `ultron chat --agent <type>`
  and the `/agent` slash command (validated against `SUPPORTED_AGENTS` in
  `core/agents/__init__.py`).
- **Safety contract (both agents):** read-only/low-risk tool calls execute
  directly; **state-modifying actions never execute silently** — they surface as
  `PendingAction` so the CLI shows a confirmation card first.
- **ReAct** builds its system prompt at call time from the live tool-registry
  schema, parses fenced JSON tool calls, and can emit a single `run_tool_batch`
  for parallel read-only lookups (each member still gated by the boundary).

### 5.3 Memory & learning

- **Graph memory** (`core/tools/memory/graph.py`) — `add_memory` parses sentences
  into triples stored in SQLite (`.ultron_memory.db`), with flat-fact fallback.
  `query_triples`/`search_triples` support deterministic multi-hop deduction.
- **API schema learning** (`core/learning/api_schema.py`) — records request/
  response shapes, parses 400/422 validation errors for drift signals, and
  auto-corrects high-confidence renames on the next call.
- **Associations** (`core/learning/associations.py`) — cross-domain fact
  correlation with a curated concept-bridge map and a persistent connections DB.
- Scaffolds: `traces.py`, `spec_search.py`, `router_policies.py`, `faiss.py`,
  `hybrid.py`.

### 5.4 Security model

`ultron.security.SecurityBoundary` (mode from `ULTRON_SECURITY_MODE`:
`permissive` / `interactive` / `strict`) classifies every action into a risk
tier and returns `allow` / `confirm` / `deny`. Guardrails hard-block leaked
credentials, PII, non-https URLs, path escapes (via `file_policy.py`
confinement), and dangerous shell patterns. Every verdict is written to
`~/.ultron/security_audit.jsonl`.

### 5.5 UI / rendering

- `ui/theme.py` — the single UI surface: molten-flame palette
  (`EMBER #E73F1E`, `ORANGE #FB6C00` (primary accent), `AMBER #F9B637`,
  `GOLD #FFDD9C`), adaptive ASCII-logo banner (side/stack/text modes), boxed
  responses (`╭─ ULTRON ─╮` with `#FB6C00` border), help table, action cards.
- `ui/session.py` — prompt_toolkit `ChatSession`, width-adaptive bottom toolbar
  (status dot / model / agent / cwd / security chip / hints), input-line clearing
  so the transcript echo is never duplicated.
- `ui/responsive.py` — `ResizeReflow` records every print and, on window resize,
  re-renders the whole conversation at the live width (guarded to only run while
  the chat app is active; ignores the thinking spinner).

### 5.6 Configuration

`core/config.py` (pydantic-settings) reads `ULTRON_*` env vars / `.env`:
`ULTRON_MODEL`, `ULTRON_SECURITY_MODE`, `ULTRON_LOG_LEVEL`, `ULTRON_DATA_DIR`,
`ULTRON_DATABASE_TYPE`/`ULTRON_DATABASE_URL`, plus reserved
`ULTRON_WAKE_WORD`/`ULTRON_MEMORY_BACKEND`.

---

## 6. Tool Inventory (38 registered tools)

| Tool | Source | Purpose |
|---|---|---|
| `read_file` | builtin/file_reader.py | Read a file (confined to project dir) |
| `write_file` | builtin/file_writer.py | Create/overwrite files (confirm-gated) |
| `run_command` | builtin/command_runner.py | Run one shell command (confirm-gated; pytest/ruff outputs summarized) |
| `run_parallel` | builtin/command_runner.py | Run many commands concurrently; every command individually gated |
| `make_http_request` | builtin/http_client.py | GET auto; POST/PUT/DELETE confirm-gated; https/localhost only |
| `retrieve` | builtin/retrieval.py | Unified retrieval: decides search / page fetch / connectivity (or combines) |
| `check_connectivity` | builtin/retrieval.py | Connectivity probe for a URL |
| `search_web` | builtin/web_search.py | Web search (ddgs) |
| `fetch_page_text` | builtin/web_search.py | Fetch + readability-extract a page |
| `run_query` | builtin/database.py | SQLite/Postgres; read-only SELECT auto, writes confirm-gated |
| `learn_api_schema` / `api_usage_hint` / `get_api_knowledge` / `forget_api` | learning/api_schema.py | Learn, predict, inspect, forget API shapes |
| `check_resources` / `resource_forecast` | core/tools/resource_monitor.py | System snapshot; command cost forecast |
| `add_memory` | memory/graph.py `store_memory_text` | Store fact (triples when parseable, else flat) |
| `add_triple` / `query_triples` / `search_triples` / `get_all_triples` | memory/graph.py | Knowledge-graph write / deductive query |
| `get_all_memories` / `search_memories` | memory/sqlite.py | Flat memory store |
| `memory_connections` / `related_facts` / `discover_connections` / `explain_relation` | learning/associations.py | Personalized cross-domain memory links |
| `enforce_schema` / `schema_validate` / `list_schemas` | intelligence/structured_output.py | Structured JSON/XML/Markdown enforcement |
| `preflight_plan` / `analyze_dependencies` / `list_plan_actions` | intelligence/planning.py | Proactive plan preflight + dependency analysis |
| `get_debug_context` / `diagnose_failure` / `check_dependency` | intelligence/debug_context.py | Environmental-state debugging |
| `run_tool_batch` / `synthesize_analysis` | intelligence/parallel_tools.py | Run several different tools concurrently, gated per member, synthesize results |

`core/tools/registry.py` also exposes `get_tool(name)` and `get_tools_schema()`
(dynamic JSON Schema list used by the ReAct agent's system prompt).

---

## 7. What This AI Can Do (capability summary)

- **File operations** — read / create / overwrite files with confirmation.
- **Shell** — run single or parallel command batches; pytest & ruff outputs are
  auto-summarized; heavy commands warn before running.
- **Web** — search, fetch readable page text, connectivity checks, unified
  `retrieve`.
- **HTTP APIs** — requests with safety restrictions; **learns API schemas and
  auto-corrects on drift**.
- **Databases** — SQLite/Postgres queries (writes confirm-gated).
- **Memory** — explicit facts, deductive graph queries, personalized
  cross-domain connections, and discovery of novel links.
- **Planning** — multi-step task preflight with per-step permissions and
  data-flow dependencies.
- **Structured answers** — enforced JSON / XML / Markdown schemas.
- **Debugging** — failure diagnosis against a live environment snapshot.
- **Resources** — system monitoring and command cost forecasting.
- **Parallel intelligence** — concurrent multi-tool synthesis.
- **Multimodal** — vision-model image analysis (if a vision model is loaded).
- **Polished CLI** — slash commands (`/help /model /agent /security /memory
  /clear /reload /exit`), Esc to cancel, resize-proof responsive UI.

---

## 8. Testing, Lint & CI

- **Tests:** `python -m pytest` → 600+ tests across 24 files in `tests/`
  (agents, security, memory, learning, intelligence, tools, UI, slash commands).
  Run without a live Ollama (engines are mocked/faked).
- **Lint:** `ruff check .` (clean).
- **Stress harness:** `python _stress_audit.py` — 29 checks across all recent
  features; last report: `AnalysisReport.md`.
- **Resize e2e:** `python _reflow_e2e.py` — spawns the real CLI in a pty and
  proves the transcript re-renders exactly once per resize.
- **CI (GitHub Actions, `.github/workflows/ci.yml`):** jobs `test`, `stress`,
  `reflow-e2e` on Python 3.12 via `pip install -e ".[dev]"`.

---

## 9. Development Conventions

- **Add a tool:** 1) function in `core/tools/builtin/`, 2) register in
  `core/tools/registry.py`, 3) optionally a `detect_*`/`handle_*` pair in
  `simple.py` so the simple agent can recognize the intent.
- **Add an agent:** subclass `BaseAgent`, implement `run()`, add to
  `SUPPORTED_AGENTS` + `get_agent()` factory.
- **UI:** all rendering goes through `ui/theme.py`; width-adaptivity is handled
  by `ui/responsive.py`; new printed content is automatically recorded for
  reflow — do not bypass `console.print`.
- **Security:** any new state-modifying tool must route through
  `boundary.check()` and return a `PendingAction` when the verdict is `confirm`.
- **Quality bar:** production-grade — every feature ships with tests, is
  ruff-clean, and passes the stress harness.

---

*Generated as the Ultron project handoff package. Repo: github.com/Aravindhan450/ultron.*
