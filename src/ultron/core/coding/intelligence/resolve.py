"""ultron.core.coding.intelligence.resolve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbol query resolution — turns a user-typed symbol phrase into a
VERIFIED / INFERRED / UNKNOWN, evidence-grounded answer.

The observed failures this layer fixes:

- ``taskstate`` / ``codingexecutor`` returned *No references/definition*
  because the index queried the exact (case-sensitive) identifier.
- ``find where the supervisor is defined`` produced a speculative
  *"src/ultron/core/supervisor.py is likely ..."* inference, because the
  query fell through to the LLM and the LLM guessed from a filename.

Design rules:

- :func:`normalize_symbol_phrase` generates candidate identifier spellings
  from any phrase (``task state`` -> ``TaskState``/``task_state``/...).
- The cascade is always **exact index -> case-insensitive index ->
  normalized identifiers -> lexical source search -> references ->
  semantic** — never the reverse (no semantic jump for exact-symbol
  questions).
- A result is only ``VERIFIED`` when the symbol index or a real source
  definition line (``class X`` / ``def X`` / ...) proves it. Everything
  else is ``INFERRED`` or ``UNKNOWN`` — the layer never emits a file path
  it has not actually seen in the index or source.

The registered tools and the :class:`CodeIntelligenceBridge` both use this
module, so the ReAct loop and the deterministic NLP route behave the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ultron.core.coding.intelligence.search import search_code
from ultron.core.coding.intelligence.semantic import SemanticHit
from ultron.core.coding.intelligence.symbols import (
    Symbol,
    SymbolReference,
)

# Definition-looking source lines accepted as VERIFIED lexical evidence.
_DEFINITION_LINE_RE = re.compile(
    r"\b(?:class|def|interface|struct|trait|enum|type|func|fn|function|"
    r"const|let|var|mixin)\s+([\w.]+)\b"
)

_MAX_LEXICAL = 8  # fallback search cap (bounded, deterministic)


@dataclass
class ResolvedLookup:
    """One symbol-query resolution result (status + evidence)."""

    query: str
    status: str = "UNKNOWN"  # VERIFIED | INFERRED | UNKNOWN
    strategy: str = "unknown"  # exact|case_insensitive|normalized|lexical|references|semantic
    matched_name: str | None = None  # the identifier actually found
    definitions: list[Symbol] = field(default_factory=list)
    references: list[SymbolReference] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    lexical_hits: list[str] = field(default_factory=list)  # raw search lines
    semantic_hits: list[SemanticHit] = field(default_factory=list)


def _split_words(phrase: str) -> list[str]:
    """Splits a phrase into identifier words (separators + camelCase)."""
    parts = re.split(r"[\s_\-/]+", phrase)
    words: list[str] = []
    for part in parts:
        if not part:
            continue
        sub = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
        words.extend(sub if sub else [part])
    return [w for w in words if w]


def normalize_symbol_phrase(phrase: str) -> list[str]:
    """Returns candidate identifier spellings for *phrase* (most likely first).

    Examples::

        "task state"    -> ["task state", "TaskState", "task_state", ...]
        "taskstate"     -> ["taskstate", "Taskstate", "TASKSTATE", ...]
        "codingexecutor"-> ["codingexecutor", "Codingexecutor", ...]
        "the supervisor"-> ["supervisor", "Supervisor", ...]

    Leading articles (``the/a/an``) are stripped — ``the supervisor`` is the
    symbol ``supervisor``, never a literal identifier.
    """
    phrase = (phrase or "").strip().strip("\"'`").strip("?.!,;:").strip()
    if not phrase:
        return []
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE).strip()
    words = _split_words(phrase)
    if not words:
        return []

    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(phrase)                                   # as typed
    add(" ".join(words))                          # spaced
    add("".join(words))                           # concatenated
    add("_".join(words))                          # snake (original case)
    add("_".join(w.lower() for w in words))       # snake_case
    add("".join(w.capitalize() for w in words))   # PascalCase
    add(words[0].lower() + "".join(w.capitalize() for w in words[1:]))  # camelCase
    add("".join(w.lower() for w in words))        # lowercase concat
    add(" ".join(w.capitalize() for w in words))  # Title Case spaced
    add(phrase.lower())
    add(phrase.upper())
    return candidates


def _skip(candidate: str) -> bool:
    """Filters junk candidates (articles/verbs/stopwords, too-short)."""
    if len(candidate) < 2:
        return True
    return candidate.lower() in {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in",
        "at", "for", "it", "that", "this", "find", "show", "where", "how",
        "what", "why", "which", "who", "do", "does", "did",
    }


def _lexical_definition_lines(ci, candidate: str) -> list[str]:
    """Source lines that *prove* a definition of *candidate* (regex, bounded)."""
    pattern = (
        r"\b(?:class|def|interface|struct|trait|enum|type|func|fn|function|"
        r"const|let|var|mixin)\s+" + re.escape(candidate) + r"\b"
    )
    out = search_code(pattern, path=str(ci.root), regex=True, max_results=_MAX_LEXICAL)
    if not out or out.startswith(("No matches", "Error:")):
        return []
    return out.splitlines()


def _lexical_occurrence_lines(ci, candidate: str) -> list[str]:
    """Word-boundary occurrences of *candidate* in the source (bounded)."""
    out = search_code(
        r"\b" + re.escape(candidate) + r"\b",
        path=str(ci.root),
        regex=True,
        max_results=_MAX_LEXICAL,
    )
    if not out or out.startswith(("No matches", "Error:")):
        return []
    return out.splitlines()


# ---------------------------------------------------------------------------
# Resolution cascades (one per operation)
# ---------------------------------------------------------------------------


def _candidates_for(phrase: str) -> list[str]:
    """Raw phrase first (never filtered — single-char symbols are valid),
    then derived spellings with junk candidates removed."""
    phrase = (phrase or "").strip()
    derived = [c for c in normalize_symbol_phrase(phrase) if c != phrase]
    return [phrase] + [c for c in derived if not _skip(c)]


def resolve_definition(ci, phrase: str) -> ResolvedLookup:
    """Finds a VERIFIED definition, or explains why none could be verified."""
    phrase = (phrase or "").strip()
    result = ResolvedLookup(query=phrase)
    if not phrase:
        return result

    candidates = _candidates_for(phrase)

    # 1-3: exact -> case-insensitive -> normalized identifier lookups.
    for candidate in candidates:
        definitions = ci.find_definition(candidate)
        if definitions:
            result.status = "VERIFIED"
            result.strategy = "exact" if candidate == phrase else "normalized"
            result.matched_name = definitions[0].name
            result.definitions = definitions
            return result
    for candidate in candidates:
        definitions = ci.find_definition(candidate, case_insensitive=True)
        if definitions:
            result.status = "VERIFIED"
            result.strategy = "case_insensitive"
            result.matched_name = definitions[0].name
            result.definitions = definitions
            return result

    # 4: lexical source fallback — a real definition line is verified evidence.
    for candidate in candidates:
        hits = _lexical_definition_lines(ci, candidate)
        if hits:
            result.status = "VERIFIED"
            result.strategy = "lexical"
            result.matched_name = candidate
            result.lexical_hits = hits
            return result

    # 5: references suggest the symbol exists (but no verified definition).
    reference_lookup = resolve_references(ci, phrase)
    if reference_lookup.status == "VERIFIED" and reference_lookup.references:
        result.status = "INFERRED"
        result.strategy = "references"
        result.matched_name = reference_lookup.matched_name
        result.references = reference_lookup.references[:8]
        return result

    # 6: semantic fallback for conceptual queries.
    semantic_hits = ci.search_semantically(phrase, top_k=3)
    if semantic_hits:
        result.status = "INFERRED"
        result.strategy = "semantic"
        result.semantic_hits = semantic_hits
        return result

    return result  # UNKNOWN — no speculative claims


def resolve_references(ci, phrase: str) -> ResolvedLookup:
    """Finds usage sites of *phrase* (exact -> CI -> normalized -> lexical)."""
    phrase = (phrase or "").strip()
    result = ResolvedLookup(query=phrase)
    if not phrase:
        return result

    candidates = _candidates_for(phrase)

    for candidate in candidates:
        references = ci.find_references(candidate)
        if references:
            result.status = "VERIFIED"
            result.strategy = "exact" if candidate == phrase else "normalized"
            result.matched_name = references[0].name
            result.references = references
            return result
    for candidate in candidates:
        references = ci.find_references(candidate, case_insensitive=True)
        if references:
            result.status = "VERIFIED"
            result.strategy = "case_insensitive"
            result.matched_name = references[0].name
            result.references = references
            return result

    # Lexical fallback: word-boundary occurrences are real source usages.
    for candidate in candidates:
        hits = _lexical_occurrence_lines(ci, candidate)
        if hits:
            result.status = "INFERRED"
            result.strategy = "lexical"
            result.matched_name = candidate
            result.lexical_hits = hits
            return result

    semantic_hits = ci.search_semantically(phrase, top_k=3)
    if semantic_hits:
        result.status = "INFERRED"
        result.strategy = "semantic"
        result.semantic_hits = semantic_hits
        return result
    return result


def resolve_symbol(ci, phrase: str) -> ResolvedLookup:
    """Finds every symbol named *phrase* (any kind)."""
    phrase = (phrase or "").strip()
    result = ResolvedLookup(query=phrase)
    if not phrase:
        return result

    candidates = _candidates_for(phrase)

    for candidate in candidates:
        symbols = ci.find_symbol(candidate)
        if symbols:
            result.status = "VERIFIED"
            result.strategy = "exact" if candidate == phrase else "normalized"
            result.matched_name = symbols[0].name
            result.symbols = symbols
            return result
    for candidate in candidates:
        symbols = ci.find_symbol(candidate, case_insensitive=True)
        if symbols:
            result.status = "VERIFIED"
            result.strategy = "case_insensitive"
            result.matched_name = symbols[0].name
            result.symbols = symbols
            return result

    for candidate in candidates:
        hits = _lexical_occurrence_lines(ci, candidate)
        if hits:
            result.status = "INFERRED"
            result.strategy = "lexical"
            result.matched_name = candidate
            result.lexical_hits = hits
            return result
    return result


# ---------------------------------------------------------------------------
# Tool-friendly formatting (shared by tools.py and the bridge)
# ---------------------------------------------------------------------------


def format_definition_result(result: ResolvedLookup) -> str:
    """Formats a definition resolution as an evidence-grounded string."""
    query = result.query
    if result.status == "VERIFIED" and result.definitions:
        head = f"Definitions of '{result.matched_name}':"
        if result.strategy == "case_insensitive":
            head = f"Definitions of '{result.matched_name}' (query: '{query}', case-insensitive):"
        elif result.strategy in ("normalized", "lexical"):
            head = f"Definitions of '{result.matched_name}' (query: '{query}'):"
        lines = [head]
        for symbol in result.definitions[:10]:
            lines.append(f"  - {symbol.to_prompt_line()}")
        return "\n".join(lines)

    if result.status == "VERIFIED" and result.lexical_hits:
        lines = [
            "Definitions of '" + str(result.matched_name) + "' found via "
            "source search (query: '" + query + "'):"
        ]
        lines.extend(f"  - {hit}" for hit in result.lexical_hits[:8])
        return "\n".join(lines)

    if result.status == "INFERRED" and result.references:
        lines = [
            "No definition found for '" + query + "' in the index; found "
            "references suggesting the symbol exists (definition not verified):"
        ]
        lines.extend(f"  - {ref.to_prompt_line()}" for ref in result.references[:8])
        return "\n".join(lines)

    if result.status == "INFERRED" and result.semantic_hits:
        lines = [
            "No definition found for '" + query + "' in the index; "
            "closest semantic matches:"
        ]
        lines.extend(f"  - {hit.to_prompt_line()}" for hit in result.semantic_hits[:5])
        return "\n".join(lines)

    if result.status == "VERIFIED":
        return f"No definition found for '{query}' in the index."
    return f"No definition found for '{query}' in the repository."


def format_reference_result(result: ResolvedLookup) -> str:
    """Formats a references resolution as an evidence-grounded string."""
    query = result.query
    if result.status == "VERIFIED" and result.references:
        head = f"References to '{result.matched_name}' ({len(result.references)} found, showing up to 20):"
        if result.strategy in ("case_insensitive", "normalized"):
            head = (
                f"References to '{result.matched_name}' (query: '{query}', "
                f"{len(result.references)} found, showing up to 20):"
            )
        lines = [head]
        for ref in result.references[:20]:
            lines.append(f"  - {ref.to_prompt_line()}")
        return "\n".join(lines)

    if result.status == "INFERRED" and result.lexical_hits:
        lines = [
            "References to '" + query + "' (lexical source occurrences, not "
            "symbol-index verified — may include definition lines):"
        ]
        lines.extend(f"  - {hit}" for hit in result.lexical_hits[:8])
        return "\n".join(lines)

    if result.status == "INFERRED" and result.semantic_hits:
        lines = [f"No references found for '{query}'; closest semantic matches:"]
        lines.extend(f"  - {hit.to_prompt_line()}" for hit in result.semantic_hits[:5])
        return "\n".join(lines)

    return f"No references found for '{query}'."


def format_symbol_result(result: ResolvedLookup) -> str:
    """Formats a symbol-search resolution as an evidence-grounded string."""
    query = result.query
    if result.status == "VERIFIED" and result.symbols:
        head = f"Symbols named '{result.matched_name}':"
        if result.strategy in ("case_insensitive", "normalized"):
            head = f"Symbols named '{result.matched_name}' (query: '{query}'):"
        lines = [head]
        for symbol in result.symbols[:20]:
            lines.append(f"  - {symbol.to_prompt_line()}")
        return "\n".join(lines)

    if result.status == "INFERRED" and result.lexical_hits:
        lines = [
            "No symbol named '" + query + "' in the index; lexical source "
            "occurrences:"
        ]
        lines.extend(f"  - {hit}" for hit in result.lexical_hits[:8])
        return "\n".join(lines)

    return f"No symbol named '{query}' found in the index."


# ---------------------------------------------------------------------------
# Repository investigation ("how does X work" / "where is X implemented")
# ---------------------------------------------------------------------------


def _file_priority(file_path: str) -> int:
    """Ranks a repo-relative file for implementation questions.

    src/ beats tests/ beats docs/ beats root scripts — incidental matches in
    live-check scripts or docs never dominate an implementation answer.
    """
    p = (file_path or "").replace("\\", "/")
    if p.startswith("src/"):
        return 0
    if p.startswith("tests/"):
        return 1
    if p.startswith("docs/"):
        return 2
    if p.startswith("scripts/"):
        return 3
    return 4


def _test_reference_lines(ci, name: str) -> list[str]:
    """Test files referencing *name* (bounded, evidence-based)."""
    tests_dir = str(ci.root) + "/tests"
    out = search_code(
        r"\b" + re.escape(name) + r"\b",
        path=tests_dir,
        regex=True,
        max_results=_MAX_LEXICAL,
    )
    if not out or out.startswith(("No matches", "Error:")):
        return []
    # search_code returns paths relative to the tests dir; prefix them so
    # the evidence reads as repo-relative (tests/test_auth.py:…).
    prefixed = []
    for line in out.splitlines():
        if line.startswith("tests/"):
            prefixed.append(line)
        else:
            prefixed.append("tests/" + line)
    return prefixed


@dataclass
class InvestigationResult:
    """Synthesized repository investigation (definition + relationships)."""

    query: str
    status: str = "UNKNOWN"  # VERIFIED | INFERRED | UNKNOWN
    primary_name: str | None = None
    primary_kind: str | None = None
    primary_file: str | None = None
    primary_lines: list[str] = field(default_factory=list)
    supporting: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def resolve_investigation(ci, phrase: str) -> InvestigationResult:
    """Synthesizes \"how does X work\" / \"where is X implemented\" answers.

    Strategy:

    1. Definition cascade on the phrase (exact -> CI -> normalized ->
       lexical) — a verified definition is the primary implementation.
    2. Supporting components: imports + dependents of the defining file.
    3. Relevant tests: test files referencing the symbol.
    4. If no definition exists (conceptual subject like \"command
       execution\"): ranked semantic evidence, src-first.

    Everything is evidence-grounded; UNKNOWN is returned instead of a
    filename-convention guess.
    """
    phrase = (phrase or "").strip()
    result = InvestigationResult(query=phrase)
    if not phrase:
        return result

    definition = resolve_definition(ci, phrase)
    if definition.status == "VERIFIED":
        if definition.definitions:
            symbol = definition.definitions[0]
            result.status = "VERIFIED"
            result.primary_name = symbol.name
            result.primary_kind = symbol.kind.value
            result.primary_file = symbol.location.file
            result.primary_lines = [symbol.to_prompt_line()]
            result.evidence = [symbol.to_prompt_line()]
            # Supporting components: imports + dependents of the defining file.
            rel = symbol.location.file.lstrip("./")
            try:
                imports = ci.get_imports(rel)
            except Exception:  # noqa: BLE001 — index edge queries are best-effort
                imports = []
            for edge in imports[:5]:
                result.supporting.append(f"imports {edge.to_prompt_line()}")
            try:
                dependents = ci.get_dependents(rel)
            except Exception:  # noqa: BLE001
                dependents = []
            for dep in dependents[:5]:
                result.supporting.append(f"used by {dep}")
            result.tests = _test_reference_lines(ci, symbol.name)[:5]
            return result
        if definition.lexical_hits:
            result.status = "VERIFIED"
            result.primary_lines = definition.lexical_hits[:4]
            result.evidence = definition.lexical_hits[:4]
            return result

    # No verified definition: rank ALL evidence src-first.  Lexical source
    # hits ("command execution" -> src/ultron/core/coding/command.py) are
    # merged with the semantic hits the definition cascade already collected
    # (plus a fresh top-k search), and files under src/ outrank tests/docs/
    # scripts so incidental matches never dominate implementation answers.
    lexical = _lexical_occurrence_lines(ci, phrase)
    hits = list(definition.semantic_hits) + list(ci.search_semantically(phrase, top_k=10))
    if lexical or hits:
        result.status = "INFERRED"
        # file -> (best lexical line, best semantic hit)
        grouped: dict[str, tuple[str | None, SemanticHit | None]] = {}
        for line in lexical:
            file_path = line.split(":", 1)[0]
            grouped.setdefault(file_path, (None, None))
            cur_lex, _ = grouped[file_path]
            if cur_lex is None:
                grouped[file_path] = (line, grouped[file_path][1])
        for hit in hits:
            file_path = hit.chunk.file
            lex, cur_sem = grouped.get(file_path, (None, None))
            if cur_sem is None:
                grouped[file_path] = (lex, hit)
        ranked = sorted(
            grouped,
            key=lambda f: (_file_priority(f), -(grouped[f][0] is not None)),
        )
        if ranked:
            result.primary_file = ranked[0]
        for file_path in ranked[:6]:
            lex_line, sem_hit = grouped[file_path]
            evidence_line = lex_line or (sem_hit.to_prompt_line() if sem_hit else file_path)
            result.primary_lines.append(
                f"{evidence_line} (semantic match, verify in source)"
            )
        result.evidence = [
            (grouped[f][0] or (grouped[f][1].to_prompt_line() if grouped[f][1] else f))
            for f in ranked[:6]
        ]
        return result

    return result  # UNKNOWN — no speculative claims


def format_investigation_result(result: InvestigationResult) -> str:
    """Formats an investigation into primary / supporting / tests / summary."""
    query = result.query
    lines = [f"Repository investigation: '{query}'", ""]
    if result.status == "UNKNOWN":
        lines.append(
            f"No repository evidence found for '{query}' in the current "
            "repository."
        )
        return "\n".join(lines)

    if result.primary_lines:
        lines.append("Primary implementation:")
        lines.extend(f"  - {line}" for line in result.primary_lines[:4])
        lines.append("")
    if result.supporting:
        lines.append("Supporting components:")
        lines.extend(f"  - {line}" for line in result.supporting[:8])
        lines.append("")
    if result.tests:
        lines.append("Relevant tests:")
        lines.extend(f"  - {line}" for line in result.tests[:5])
        lines.append("")

    summary: list[str] = []
    if result.primary_name:
        where = result.primary_file or "the repository"
        summary.append(
            f"{result.primary_name} ({result.primary_kind or 'symbol'}) is the "
            f"primary implementation, at {where}."
        )
    elif result.primary_file:
        summary.append(
            f"Best evidence points to {result.primary_file} as the primary "
            "implementation (semantic match — verify in source)."
        )
    if result.supporting:
        summary.append(
            f"It relates to {len(result.supporting)} supporting "
            "component(s) (imports/dependents)."
        )
    if result.tests:
        summary.append(f"{len(result.tests)} relevant test reference(s) found.")
    if summary:
        lines.append("Summary: " + " ".join(summary))
        lines.append("")
    lines.append(f"Evidence status: {result.status}")
    return "\n".join(lines)
