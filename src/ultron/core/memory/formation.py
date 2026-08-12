"""ultron.core.memory.formation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic memory formation (FIX #6 integration).

Nothing here is left to the LLM: facts are promoted to project memory only
when they come from a structured, verifiable source:

- **Workspace facts** — what the repository itself declares (language,
  package manager, build system, test framework, source/test directories).
  Source: REPOSITORY_INSPECTION / DIRECT_OBSERVATION.
- **Intelligence facts** — ``symbol:X -> file`` pairs produced by the code
  index (definition lookups the executor actually performed). Source:
  CODE_INTELLIGENCE / DIRECT_OBSERVATION.
- **Reconciliation** — the code index is the authority. When a stored
  symbol fact points at a file that no longer matches the current
  definition, the new location supersedes it (history kept); when the
  symbol no longer exists in the index, the fact is marked stale.

The ``ProjectMemoryStore.store`` guard (secret scanning) and its idempotent
re-store semantics apply to every fact written here, so formation can safely
run on every reasoning turn without churning history or persisting
credentials. Trivial observations (file reads, command stdout, timings) are
never promoted — there is no formation path for them at all.
"""

from __future__ import annotations

from ultron.core.memory.models import MemoryConfidence, MemorySource


def remember_workspace_facts(store, workspace) -> int:
    """Stores DIRECT_OBSERVATION project facts from workspace detection.

    Returns the number of facts stored (0 when the store or workspace is
    unavailable). Idempotent: re-storing identical facts is a no-op.
    """
    if store is None or workspace is None:
        return 0

    stored = 0

    def _maybe(name: str, content: str) -> None:
        nonlocal stored
        if not content:
            return
        if store.store(
            name,
            content,
            source=MemorySource.REPOSITORY_INSPECTION,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
        ):
            stored += 1

    project_type = str(getattr(workspace, "project_type", "") or "")
    if project_type and project_type != "unknown":
        _maybe("project_type", f"project type is {project_type}")

    for language in (getattr(workspace, "languages", None) or [])[:6]:
        _maybe(f"language:{language}", f"project uses {language}")

    package_manager = getattr(workspace, "package_manager", None)
    _maybe("package_manager", f"package manager is {package_manager}" if package_manager else "")

    build_system = getattr(workspace, "build_system", None)
    _maybe("build_system", f"build system is {build_system}" if build_system else "")

    test_framework = getattr(workspace, "test_framework", None)
    _maybe("test_framework", f"test framework is {test_framework}" if test_framework else "")

    test_dirs = list(getattr(workspace, "test_dirs", None) or [])[:6]
    if test_dirs:
        _maybe("test_dirs", "tests live in " + ", ".join(test_dirs))

    source_dirs = list(getattr(workspace, "source_dirs", None) or [])[:6]
    if source_dirs:
        _maybe("source_dirs", "source lives in " + ", ".join(source_dirs))

    return stored


def remember_intelligence_facts(store, bridge, limit: int = 12) -> int:
    """Promotes ``symbol:X -> file`` facts from the code-intelligence log.

    Only symbols the executor actually resolved (recorded symbol-layer
    queries with hits) become facts — never guesses. Returns the number of
    facts stored.
    """
    if store is None or bridge is None:
        return 0
    stored = 0
    for name, file in bridge.symbol_facts(limit):
        if store.store(
            f"symbol:{name}",
            f"defined in {file}",
            source=MemorySource.CODE_INTELLIGENCE,
            confidence=MemoryConfidence.DIRECT_OBSERVATION,
            metadata={"file": file},
        ):
            stored += 1
    return stored


def reconcile_project_memory(store, bridge) -> int:
    """Reconciles stored symbol facts against the current code index.

    The index is the authority (FIX #6 rule: current repository wins):

    - a fact whose file no longer matches the current definition is
      superseded by the new location (history preserved);
    - a fact whose symbol no longer exists in the index is marked stale,
      never silently deleted.

    Returns the number of records updated.
    """
    if store is None or bridge is None:
        return 0
    updated = 0
    for record in store.recall(limit=200):
        name = record.name or ""
        if not name.startswith("symbol:"):
            continue
        symbol = name[len("symbol:") :]
        files = bridge.definition_files(symbol)
        current_file = files[0] if files else None
        old_file = (record.metadata or {}).get("file")

        if current_file and old_file and current_file != old_file:
            stored = store.store(
                name,
                f"defined in {current_file}",
                source=MemorySource.CODE_INTELLIGENCE,
                confidence=MemoryConfidence.DIRECT_OBSERVATION,
                metadata={"file": current_file},
            )
            updated += 1 if stored else 0
        elif current_file is None and old_file is not None:
            if store.invalidate(name):
                updated += 1
    return updated
