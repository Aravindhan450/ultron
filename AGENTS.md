# Repository Guidelines

Ultron is a local-first, permission-gated AI assistant CLI (Python 3.11+, Typer + Rich + prompt_toolkit). It talks to local LLMs via Ollama, uses 38 registered tools to act on the machine, remembers facts as a knowledge graph, and routes every tool call through a security boundary (risk classifier + guardrails). See `PROJECT_CONTEXT.md` for the full vision and end-to-end architecture.

## Project Structure & Module Organization

- `src/ultron/main.py` — Typer CLI (`run` / `chat --agent` / `logs`) and the `async_chat` loop: prompt → echo → slash command or `agent.run()` → render.
- `src/ultron/core/agents/` — `BaseAgent` subclasses. `simple.py` (deterministic `detect_*` intents + LLM fallback) and `react.py` (LLM Thought→Action→Observation loop) are implemented; `orchestrator/codeact/operative/monitor.py` are scaffolds. `SUPPORTED_AGENTS` in `agents/__init__.py` is the single source of truth for `--agent` and `/agent`.
- `src/ultron/core/tools/` — `definitions.py` is the **single canonical source of tool metadata** (`TOOL_DEFINITIONS`: name, capabilities, domain, description, read_only, risk, confirmation, aliases, routing classification). `registry.py` derives the tool registry (58 tools) and JSON Schema from it. Add a tool: function in `tools/builtin/`, one `_define(...)` entry in `definitions.py` — registry, routing, ReAct redirects, security tiers, and classifier labels follow automatically. Do NOT add tool names to any other list; consumers derive from canonical queries (`readonly_tool_names()`, `code_intel_tool_names()`, `preferred_tool_for()`, `canonical_action_name()`, …). Memory lives in `tools/memory/` (graph triples + sqlite flat facts).
- `src/ultron/core/intelligence/` + `learning/` — non-tool capabilities wired as registered tools: structured output, parallel tool batches, plan preflight, env-state debugging, resource monitor, API-schema learning, cross-domain associations.
- `src/ultron/security/` — `boundary.py` decides allow/confirm/deny per risk tier + `ULTRON_SECURITY_MODE`; `guardrails.py` hard-blocks secrets/PII/unsafe URLs/path escapes; `audit.py` writes JSONL verdicts. Every state-modifying tool must route through `boundary.check()` and return a `PendingAction` on confirm.
- `src/ultron/ui/` — all rendering (`theme.py`), the prompt_toolkit session (`session.py`), and `responsive.py` (ResizeReflow re-renders the recorded transcript at the live width on resize). New output must go through `console.print` so it is recorded for reflow.
- Stubs under `voice/`, `mcp/`, `platform/`, `ui/web`, `ui/desktop`, `ui/api`, and `core/engine/{llama_cpp,vllm,mlx,cloud}.py` are planned, not implemented.

## Build, Test, and Development Commands

- Install: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- Run: `ultron chat` (or `.venv/bin/python -m ultron.main chat`)
- Test: `.venv/bin/python -m pytest -q`
- Single test: `.venv/bin/python -m pytest tests/test_ui_session.py::test_x -q`
- Lint: `.venv/bin/ruff check .`
- Stress harness: `.venv/bin/python _stress_audit.py`
- Resize e2e: `.venv/bin/python _reflow_e2e.py`

## Coding Style & Naming Conventions

- ruff (0.16.x, target py311) is the only linter; `ruff check .` must stay clean.
- Imports: stdlib → third-party → `ultron.*`, absolute paths only.
- Tool functions map 1:1 to registry names (snake_case). Security action names are canonicalized bidirectionally (both `web_search` and `search_web` gate identically) — keep names symmetric.
- UI colors come only from `ultron.ui.theme` constants (EMBER/ORANGE/AMBER/GOLD/TEXT/MUTED/FAINT/GREEN/RED/YELLOW/BLUE) — no raw rich color markup elsewhere.

## Testing Guidelines

- pytest; engines must be mocked/faked — never require a live Ollama.
- Every feature ships with tests; CI runs the suite on Python 3.12.
- Slash-command logic is tested against the real `handle_slash_command` path.
- UI behavior is unit-tested at several terminal widths; resize regressions are caught by `_reflow_e2e.py` (pty harness) in the `reflow-e2e` CI job.

## Commit & Pull Request Guidelines

- Commit messages are short, lowercase, imperative phrases describing the change (e.g. "added web search and fetch tool", "linter and compiler diagnostics added").
- Keep the tree lint- and test-clean before committing; CI runs `test` + `stress` + `reflow-e2e` on every push.
