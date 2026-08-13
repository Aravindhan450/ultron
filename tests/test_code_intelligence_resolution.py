"""
Code Intelligence query-resolution tests (Fix #8.5b).

Covers the observed failures and the mandatory test matrix:

- case-insensitive symbol lookup (taskstate -> TaskState)
- multi-word symbol normalization (coding executor -> CodingExecutor)
- article stripping (the supervisor -> Supervisor)
- definition vs reference distinction
- lexical fallback when the symbol index misses
- evidence-grounded responses: never speculative file paths
- VERIFIED / INFERRED / UNKNOWN semantics
- semantic routing for conceptual queries
- the Supervisor test (no "likely" filename guesses)
- the negative case (no speculative claims)

All filesystem tests use temporary repositories; the real Ultron repository
is never modified.
"""

import pytest

from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.intelligence.index import RepositoryIndex
from ultron.core.coding.intelligence.resolve import (
    format_definition_result,
    normalize_symbol_phrase,
    resolve_definition,
    resolve_references,
)
from ultron.core.coding.intelligence.tools import (
    code_search,
    find_definition,
    find_references,
    find_symbol,
)
from ultron.core.coding.intelligence_bridge import CodeIntelligenceBridge
from ultron.core.nlp.intent import IntentCategory, route_request
from ultron.core.tools import paths as tools_paths


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A temp workspace that is the ALLOWED_BASE_DIR (so tools work)."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(root, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------


def test_normalize_symbol_phrase_candidates():
    # Lowercase single-word phrases cannot be camel-split without more
    # context — the case-insensitive index lookup handles those (covered in
    # the resolver tests below). Normalization covers spaced/multi-word.
    assert "taskstate" in normalize_symbol_phrase("taskstate")
    assert "TaskState" in normalize_symbol_phrase("task state")
    assert "TaskState" in normalize_symbol_phrase("Task State")
    assert "taskstate" in normalize_symbol_phrase("TASKSTATE")
    assert "TaskState" in normalize_symbol_phrase("TaskState")
    assert "codingexecutor" in normalize_symbol_phrase("codingexecutor")
    assert "CodingExecutor" in normalize_symbol_phrase("coding executor")
    assert "CodingExecutor" in normalize_symbol_phrase("coding_executor")
    assert "Supervisor" in normalize_symbol_phrase("the supervisor")
    assert "Supervisor" in normalize_symbol_phrase("supervisor")
    assert "OrchestrationValidator" in normalize_symbol_phrase(
        "orchestration validator"
    )
    assert "orchestrationvalidator" in normalize_symbol_phrase("orchestrationvalidator")


def test_normalize_symbol_phrase_strips_articles():
    assert "supervisor" in normalize_symbol_phrase("the supervisor")
    assert "orchestration validator" in normalize_symbol_phrase("an orchestration validator")
    # A phrase that is genuinely about the article word is not corrupted.
    assert "TaskState" in normalize_symbol_phrase("Task State")


# ---------------------------------------------------------------------------
# Case-insensitive index queries
# ---------------------------------------------------------------------------


def test_index_case_insensitive_lookup(sandbox):
    _write(sandbox, "models.py", "class TaskState:\n    pass\n")
    index = RepositoryIndex(str(sandbox))
    index.refresh()

    assert index.find_definition("taskstate", case_insensitive=True)
    assert not index.find_definition("taskstate")  # exact stays strict
    assert index.find_symbol("TASKSTATE", case_insensitive=True)
    index.close()


# ---------------------------------------------------------------------------
# Definition lookup — the observed failure matrix
# ---------------------------------------------------------------------------


def test_definition_lookup_case_variants(sandbox):
    _write(sandbox, "src/models.py", "class TaskState:\n    pass\n")
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    for query in ("TaskState", "taskstate", "TASKSTATE", "Task State", "task state"):
        result = resolve_definition(ci, query)
        assert result.status == "VERIFIED", query
        assert result.matched_name == "TaskState", query
        assert result.definitions[0].location.file == "src/models.py", query
    ci.close()


def test_definition_lookup_multiword_and_article(sandbox):
    _write(sandbox, "core.py", "class CodingExecutor:\n    pass\n")
    _write(sandbox, "sup.py", "class Supervisor:\n    pass\n")
    _write(sandbox, "val.py", "class OrchestrationValidator:\n    pass\n")
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    cases = {
        "codingexecutor": "CodingExecutor",
        "coding executor": "CodingExecutor",
        "Coding Executor": "CodingExecutor",
        "supervisor": "Supervisor",
        "the supervisor": "Supervisor",
        "orchestration validator": "OrchestrationValidator",
        "orchestrationvalidator": "OrchestrationValidator",
        "OrchestrationValidator": "OrchestrationValidator",
    }
    for query, expected in cases.items():
        result = resolve_definition(ci, query)
        assert result.status == "VERIFIED", query
        assert result.matched_name == expected, query
    ci.close()


def test_definition_tool_output_case_insensitive(sandbox):
    _write(sandbox, "models.py", "class TaskState:\n    pass\n")
    out = find_definition("taskstate", str(sandbox))
    assert "Definitions of 'TaskState'" in out
    assert "models.py" in out


def test_definition_tool_negative_no_speculation(sandbox):
    _write(sandbox, "models.py", "class TaskState:\n    pass\n")
    out = find_definition("NoSuchSymbolZZZ", str(sandbox))
    assert "No definition found" in out
    # Never a speculative filename guess.
    assert "likely" not in out.lower()
    assert "NoSuchSymbolZZZ.py" not in out
    assert "is probably" not in out.lower()
    assert "might be" not in out.lower()


# ---------------------------------------------------------------------------
# Reference lookup
# ---------------------------------------------------------------------------


def test_reference_lookup_case_variants(sandbox):
    _write(sandbox, "defs.py", "class TaskState:\n    pass\n")
    _write(sandbox, "use.py", "from defs import TaskState\n\nx = TaskState()\n")
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    for query in ("TaskState", "taskstate", "TASKSTATE", "Task State"):
        result = resolve_references(ci, query)
        assert result.status == "VERIFIED", query
        files = {r.location.file for r in result.references}
        assert "use.py" in files, query
        assert "defs.py" not in files, query  # definition line is not a reference
    ci.close()


def test_reference_tool_case_insensitive(sandbox):
    _write(sandbox, "defs.py", "class TaskState:\n    pass\n")
    _write(sandbox, "use.py", "x = TaskState()\n")
    out = find_references("taskstate", str(sandbox))
    assert "References to 'TaskState'" in out
    assert "use.py" in out


def test_reference_tool_negative_no_speculation(sandbox):
    _write(sandbox, "models.py", "class TaskState:\n    pass\n")
    out = find_references("NoSuchRefZZZ", str(sandbox))
    assert "No references found" in out
    assert "likely" not in out.lower()
    assert "NoSuchRefZZZ.py" not in out


# ---------------------------------------------------------------------------
# find_symbol (all kinds)
# ---------------------------------------------------------------------------


def test_find_symbol_case_insensitive(sandbox):
    _write(sandbox, "app.py", "import os\n\ndef run():\n    return 1\n")
    out = find_symbol("RUN", str(sandbox))
    assert "Symbols named 'run'" in out
    assert "function" in out


# ---------------------------------------------------------------------------
# Lexical fallback (symbol index misses, source search proves it)
# ---------------------------------------------------------------------------


def test_lexical_fallback_when_index_misses(sandbox, monkeypatch):
    _write(sandbox, "src/target.py", "class WidgetThing:\n    pass\n")
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()

    # Force the symbol index to miss: simulate a stale/empty index by
    # deleting its rows after refresh (the source file still proves it).
    with ci.index._conn:
        ci.index._conn.execute("DELETE FROM symbols")
        ci.index._conn.commit()
    ci.semantic.invalidate()

    result = resolve_definition(ci, "widgetthing")
    # The index has no rows, so resolution must fall back to a verified
    # source definition line (class WidgetThing) — not "not found".
    assert result.status == "VERIFIED"
    assert result.strategy == "lexical"
    assert any("WidgetThing" in hit for hit in result.lexical_hits)
    ci.close()


def test_lexical_fallback_not_speculative_for_absent_symbol(sandbox):
    _write(sandbox, "src/anything.py", "x = 1\n")
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()
    result = resolve_definition(ci, "completelynonexistentsymbol")
    assert result.status == "UNKNOWN"
    ci.close()


# ---------------------------------------------------------------------------
# VERIFIED / INFERRED / UNKNOWN semantics
# ---------------------------------------------------------------------------


def test_inferred_when_only_references_exist(sandbox):
    # "Missing" is imported/used but never defined -> INFERRED, not VERIFIED.
    _write(sandbox, "use.py", "from nowhere import PhantomThing\n\ny = PhantomThing()\n")
    ci = CodeIntelligence(root=str(sandbox))
    ci.refresh()
    result = resolve_definition(ci, "phantomthing")
    # The import row makes find_symbol/exact fail for definitions; the
    # reference fallback yields INFERRED with a clear explanation.
    assert result.status in ("INFERRED", "UNKNOWN")
    out = format_definition_result(result)
    assert "No definition found" in out or "not verified" in out
    assert "likely" not in out.lower()
    ci.close()


def test_definition_result_never_claims_likely_location(sandbox):
    _write(sandbox, "models.py", "class TaskState:\n    pass\n")
    out = find_definition("taskstate", str(sandbox))
    assert "likely" not in out.lower()
    assert "probably" not in out.lower()


# ---------------------------------------------------------------------------
# code_search multi-word normalization
# ---------------------------------------------------------------------------


def test_code_search_normalizes_multiword_query(sandbox):
    _write(sandbox, "executor.py", "class CodingExecutor:\n    pass\n")
    out = code_search("coding executor", str(sandbox))
    assert "CodingExecutor" in out or "normalized from 'coding executor'" in out


# ---------------------------------------------------------------------------
# Query classification (intent layer)
# ---------------------------------------------------------------------------


def test_definition_intent_article_and_multiword():
    cases = {
        "find where the supervisor is defined": "supervisor",
        "find where supervisor is defined": "supervisor",
        "find where taskstate is defined": "taskstate",
        "find where Task State is defined": "Task State",
        "Where is the OrchestrationValidator defined?": "OrchestrationValidator",
        "Find where coding executor is defined": "coding executor",
    }
    for phrase, expected in cases.items():
        it = route_request(phrase)
        assert it is not None, phrase
        assert it.tool == "find_definition", phrase
        assert expected.lower() in it.arguments.get("name", "").lower(), phrase


def test_reference_intent_article_and_multiword():
    cases = {
        "find where taskstate is used": "taskstate",
        "find references to the supervisor": "supervisor",
        "find references to taskstate": "taskstate",
        "Where is Task State used?": "Task State",
        "who calls the supervisor": "supervisor",
    }
    for phrase, expected in cases.items():
        it = route_request(phrase)
        assert it is not None, phrase
        assert it.tool == "find_references", phrase
        assert expected.lower() in it.arguments.get("name", "").lower(), phrase


def test_code_search_intent_strips_articles():
    # "where X implemented" routes to repository investigation, with the
    # leading article stripped from the subject.
    it = route_request("Where is the OrchestrationValidator implemented?")
    assert it is not None
    assert it.tool == "code_investigation"
    assert it.arguments["query"] == "OrchestrationValidator"


def test_semantic_intent_where_does_x_work():
    it = route_request("where does the supervisor delegate work?")
    assert it is not None
    assert it.intent_type is IntentCategory.SEMANTIC_SEARCH
    assert it.tool == "semantic_search"
    assert "supervisor" in it.arguments.get("query", "")


def test_symbol_inspection_what_does_x_do():
    it = route_request("What does CodingExecutor do?")
    assert it is not None
    assert it.intent_type is IntentCategory.SYMBOL_INSPECTION
    assert it.tool == "report_symbol"
    assert it.arguments.get("name") == "CodingExecutor"


# ---------------------------------------------------------------------------
# Bridge integration (used by the ReAct / coding-executor path)
# ---------------------------------------------------------------------------


def test_bridge_resolves_case_insensitively(sandbox):
    _write(sandbox, "src/service.py", "class UserService:\n    pass\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()

    out = bridge.query("find_definition", name="userservice")
    assert "Definitions of 'UserService'" in out
    assert "src/service.py" in out
    bridge.close()


def test_bridge_negative_no_speculation(sandbox):
    _write(sandbox, "a.py", "x = 1\n")
    bridge = CodeIntelligenceBridge()
    assert bridge.enable(str(sandbox))
    bridge.refresh()
    out = bridge.query("find_definition", name="MissingZZZ")
    assert out.startswith("No definition")
    assert "likely" not in out.lower()
    bridge.close()
