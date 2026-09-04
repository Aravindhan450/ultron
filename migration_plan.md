# PROJECT ULTRON — OLLAMA → LLAMA.CPP MIGRATION MASTER DIRECTIVE

You are modifying the existing Project Ultron repository.

Your sole objective is:

> Completely remove Ollama as a runtime/backend dependency from Ultron and make llama.cpp/llama-server the local LLM backend, while preserving all existing Ultron functionality, architecture, security guarantees, agent behavior, CLI behavior, tests, and capability-routing logic.

## NON-NEGOTIABLE RULES

1. DO NOT redesign Ultron.

2. DO NOT rewrite unrelated modules.

3. DO NOT remove existing capabilities merely because they are unrelated to the migration.

4. DO NOT weaken or bypass the SecurityBoundary.

5. DO NOT change Intent → Capability → Tool routing architecture.

6. DO NOT change SimpleAgent/ReActAgent behavior unless required for backend compatibility.

7. DO NOT introduce cloud inference.

8. DO NOT introduce `llama-cpp-python` unless explicitly instructed.

9. The intended inference architecture is:

   Ultron Python process
   ↓
   LlamaCppEngine
   ↓
   llama-server HTTP API
   ↓
   GGUF model
   ↓
   Metal/GPU

10. llama-server is the inference runtime. Ultron should communicate with it over HTTP.

11. Preserve the existing BaseEngine abstraction.

12. Do not make agents directly aware of llama.cpp.

13. Do not make tools directly aware of llama.cpp.

14. Do not hard-code machine-specific absolute model paths into Git.

15. Do not silently change existing behavior to make tests pass.

16. Never delete Ollama until the replacement has been implemented and validated.

17. Never modify tests merely to accommodate an incorrect implementation.

18. Existing passing tests are regression contracts.

19. Every implementation phase must be validated before moving to the next phase.

20. If a requirement is ambiguous, STOP and report the ambiguity rather than inventing architecture.

## REQUIRED VALIDATION MODEL

Every functional change requires TWO validation layers:

### Layer 1 — Automated validation

Use:

* targeted unit tests
* integration tests
* regression tests
* ruff
* relevant existing harnesses

### Layer 2 — Model-in-the-loop validation

The actual Ultron CLI must communicate with the actual llama.cpp/llama-server runtime and execute small real user tasks.

Do not consider the migration complete until both layers pass.

## CURRENT ARCHITECTURAL CONTRACT

The backend abstraction currently centers around:

BaseEngine:

* generate(...)
* stream(...)

Agents consume BaseEngine rather than directly depending on Ollama.

Preserve this abstraction.

## FINAL SUCCESS CONDITION

At the end of the migration:

* LlamaCppEngine is the production local LLM backend.
* Ollama is completely removed.
* No production code imports Ollama.
* No configuration depends on Ollama.
* No documentation instructs users to install/use Ollama.
* No tests require Ollama.
* No CLI path calls Ollama endpoints.
* No references remain to:

  * Ollama
  * localhost:11434
  * /api/chat
  * /api/tags
  * /api/show
  * `ollama pull`
  * other Ollama-specific APIs

A repository-wide search must confirm zero remaining Ollama runtime references.

## IMPORTANT

Work in small phases.

At the end of each phase:

1. summarize files changed,
2. summarize architectural impact,
3. run the required validation,
4. report exact test results,
5. report any remaining risks,
6. DO NOT begin the next phase automatically unless explicitly instructed.

The goal is controlled migration, not maximum code change.
