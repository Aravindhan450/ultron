"""
Tests for the knowledge-graph memory upgrade (``ultron.core.tools.memory.graph``).

Covers entity normalization, triple storage + dedup, extraction patterns
(including the conservative negative case), flat-fact fallback, filtered
queries, multi-hop deduction (``query_chain``), reasoning templates
(``answer_question``), union recall, security classification of the new
tools, and the agent-level handlers.

All database tests run against a temporary SQLite file — never the
developer's real ``.ultron_memory.db``.
"""

import pytest

from ultron.core.agents.simple import (
    detect_deduction_question,
    handle_deduction_question,
    handle_memory_question,
)
from ultron.core.tools.memory import graph
from ultron.core.tools.registry import get_tool
from ultron.core.types import ChatMessage, Role


@pytest.fixture()
def graph_db(tmp_path, monkeypatch):
    """Points both the graph and flat-fact stores at a throwaway DB file."""
    db_path = tmp_path / "memory_test.db"
    from ultron.core.tools.memory import sqlite

    monkeypatch.setattr(graph, "MEMORY_DB_PATH", db_path)
    monkeypatch.setattr(sqlite, "MEMORY_DB_PATH", db_path)
    graph.init_graph_db()
    sqlite.init_memory_db()
    return db_path


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_entity_case_and_punctuation():
    assert graph.normalize_entity("  Paris. ") == "paris"
    assert graph.normalize_entity("Paris") == graph.normalize_entity("paris")
    assert graph.normalize_entity("United   States") == "united states"


def test_normalize_predicate_phrases():
    assert graph.normalize_predicate("is the capital of") == "capital_of"
    assert graph.normalize_predicate("IS THE CAPITAL CITY OF") == "capital_of"
    assert graph.normalize_predicate("borders") == "borders"
    assert graph.normalize_predicate("is a") == "is_a"


# ---------------------------------------------------------------------------
# Storage + dedup
# ---------------------------------------------------------------------------


def test_add_triple_and_idempotent_dedup(graph_db):
    first = graph.add_triple("Paris", "is the capital of", "France")
    assert "Stored" in first

    second = graph.add_triple("Paris", "is the capital of", "France")
    assert "Already known" in second

    # Same triple with different casing / punctuation resolves to one edge.
    third = graph.add_triple("paris", "capital of", "FRANCE.")
    assert "Already known" in third


def test_add_triple_creates_shared_entities(graph_db):
    graph.add_triple("France", "borders", "Germany")
    graph.add_triple("Paris", "is the capital of", "France")
    stats = graph.get_graph_stats()
    assert stats["entities"] == 3  # paris, france, germany
    assert stats["triples"] == 2


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("Paris is the capital of France", ("Paris", "capital_of", "France")),
        ("Paris is the capital city of France", ("Paris", "capital_of", "France")),
        ("the capital of France is Paris", ("Paris", "capital_of", "France")),
        ("France is located in Europe", ("France", "located_in", "Europe")),
        ("Ultron is a CLI assistant", ("Ultron", "is_a", "CLI assistant")),
        ("France borders Germany", ("France", "borders", "Germany")),
        ("Ada was born in London", ("Ada", "born_in", "London")),
        ("Linus founded Linux", ("Linus", "founded", "Linux")),
        ("Linus is the founder of Linux", ("Linus", "founded", "Linux")),
        ("Ada works at Babbage Inc", ("Ada", "works_at", "Babbage Inc")),
        ("I created ultron", ("I", "created", "ultron")),
    ],
)
def test_extract_triples_patterns(sentence, expected):
    assert graph.extract_triples(sentence) == [expected]


def test_extract_triples_no_overlapping_edges():
    # "is the capital of" must win over the generic "is" — exactly one triple.
    assert graph.extract_triples("Paris is the capital of France") == [
        ("Paris", "capital_of", "France")
    ]
    assert graph.extract_triples("the capital of France is Paris") == [
        ("Paris", "capital_of", "France")
    ]


def test_extract_triples_unparseable_returns_empty():
    # Generic "is" must NOT be extracted — a bare is-relationship is too
    # likely to be a false positive (e.g. "this project is going").
    assert graph.extract_triples("remember to buy milk tomorrow") == []
    assert graph.extract_triples("I really like the way this project is going") == []
    assert graph.extract_triples("my favorite color is blue") == []


# ---------------------------------------------------------------------------
# Unified write path
# ---------------------------------------------------------------------------


def test_store_memory_text_stores_triples(graph_db):
    result = get_tool("add_memory")("Paris is the capital of France")
    assert "Stored as knowledge graph" in result
    assert graph.query_triples(predicate="capital_of") == [("Paris", "capital_of", "France")]


def test_store_memory_text_falls_back_to_flat_fact(graph_db):
    result = get_tool("add_memory")("my favorite color is blue")
    assert "Remembered" in result
    # Nothing landed in the graph...
    assert graph.get_all_triples() == []
    # ...but it is recallable via the flat fact store.
    from ultron.core.tools.memory.sqlite import search_memories

    assert search_memories("favorite") == ["my favorite color is blue"]


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_query_triples_filters(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    graph.add_triple("Berlin", "is the capital of", "Germany")
    graph.add_triple("France", "borders", "Germany")

    assert graph.query_triples(predicate="capital_of") == [
        ("Paris", "capital_of", "France"),
        ("Berlin", "capital_of", "Germany"),
    ]
    assert graph.query_triples(subject="Paris") == [("Paris", "capital_of", "France")]
    assert graph.query_triples(object="Germany") == [
        ("Berlin", "capital_of", "Germany"),
        ("France", "borders", "Germany"),
    ]
    assert graph.query_triples(subject="Paris", predicate="borders") == []


def test_search_triples_keyword(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    hits = graph.search_triples("paris")
    assert any("Paris is the capital of France" in h for h in hits)


def test_recall_about_unions_subject_and_object(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    graph.add_triple("France", "borders", "Germany")
    edges = graph.recall_about("France")
    assert "Paris is the capital of France" in edges
    assert "France borders Germany" in edges


def test_remove_triple(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    assert graph.remove_triple("Paris", "is the capital of", "France") == "Removed triple."
    assert graph.query_triples() == []
    # Removing a non-existent edge is a no-op, not an error.
    assert graph.remove_triple("Paris", "is the capital of", "France") == "No matching triple found."


def test_clear_all_triples(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    graph.add_triple("France", "borders", "Germany")
    assert graph.clear_all_triples() == "All triples cleared."
    assert graph.get_graph_stats()["triples"] == 0
    assert graph.get_graph_stats()["entities"] == 3  # nodes are kept


def test_render_triple_natural_phrasing():
    assert graph.render_triple("Paris", "capital_of", "France") == "Paris is the capital of France"
    assert graph.render_triple("France", "borders", "Germany") == "France borders Germany"
    assert graph.render_triple("X", "custom_pred", "Y") == "X custom_pred Y"


# ---------------------------------------------------------------------------
# Deduction
# ---------------------------------------------------------------------------


def test_query_chain_two_hop_capital_of_bordering_country(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    graph.add_triple("Berlin", "is the capital of", "Germany")
    graph.add_triple("France", "borders", "Germany")
    graph.add_triple("Spain", "borders", "France")

    # Capital of a country that borders Germany: France borders Germany,
    # Paris is the capital of France. ``capital_of`` edges point city→country,
    # so the second hop walks backward.
    capitals = graph.query_chain(
        "germany",
        [("borders", "backward"), ("capital_of", "backward")],
    )
    assert capitals == ["Paris"]


def test_query_chain_returns_empty_when_hop_fails(graph_db):
    graph.add_triple("France", "borders", "Germany")
    assert graph.query_chain("germany", [("capital_of", "forward")]) == []


def test_answer_question_capital_of(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    assert graph.answer_question("what is the capital of France?") == (
        "The capital of France is Paris."
    )


def test_answer_question_capital_of_bordering_country(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    graph.add_triple("France", "borders", "Germany")
    answer = graph.answer_question("what is the capital of a country that borders Germany?")
    assert answer == "The capital of a country that borders Germany is Paris."


def test_answer_question_countries_bordering(graph_db):
    graph.add_triple("France", "borders", "Germany")
    answer = graph.answer_question("what countries border Germany?")
    assert "France" in answer


def test_answer_question_returns_none_without_data(graph_db):
    assert graph.answer_question("what is the capital of France?") is None
    assert graph.answer_question("who founded Linux?") is None


def test_is_deduction_question_templates():
    assert graph.is_deduction_question("what is the capital of France?")
    assert graph.is_deduction_question("capital of a country that borders Germany")
    assert graph.is_deduction_question("what countries border France")
    assert not graph.is_deduction_question("what do you remember about databases")


# ---------------------------------------------------------------------------
# Security classification
# ---------------------------------------------------------------------------


def test_graph_tools_classified_low():
    from ultron.security import RiskTier, SecurityBoundary

    boundary = SecurityBoundary(mode="interactive")
    for action in ("add_triple", "query_triples", "search_triples", "get_all_triples", "query_chain"):
        assert boundary.classify_action(action) == RiskTier.LOW
        assert boundary.decide(boundary.classify_action(action)) == "allow"


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def test_detect_deduction_question():
    assert detect_deduction_question("what is the capital of France?") is not None
    assert detect_deduction_question("what do you remember about databases") is None


def test_handle_deduction_question_answers(graph_db):
    graph.add_triple("Paris", "is the capital of", "France")
    message = handle_deduction_question("what is the capital of France?")
    assert isinstance(message, ChatMessage)
    assert message.role == Role.ASSISTANT
    assert "Paris" in message.content


def test_handle_deduction_question_no_data_is_honest(graph_db):
    message = handle_deduction_question("what is the capital of France?")
    assert "can't deduce" in message.content


def test_handle_memory_question_unions_graph_and_facts(graph_db):
    from ultron.core.tools.memory.sqlite import add_memory

    add_memory("I use FastAPI for my projects.")
    graph.add_triple("Paris", "is the capital of", "France")

    message = handle_memory_question("France")
    assert "Paris is the capital of France" in message.content

    message = handle_memory_question("FastAPI")
    assert "FastAPI" in message.content

    message = handle_memory_question("Mars")
    assert "I don't have anything stored" in message.content
