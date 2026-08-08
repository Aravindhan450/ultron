"""
Tests for the personalized learning layer (``ultron.core.learning.associations``).

Covers deterministic feature extraction (proper nouns / keywords / domains),
the curated concept-bridge model (renaissance art ↔ Medici politics), scoring
thresholds, connection persistence + announcements, transitive novel-link
discovery, the four registered tools, LOW-risk security classification, and
the agent-level detector + handlers.

All database tests run against temporary SQLite files — never the developer's
real ``.ultron_memory.db`` / ``.ultron_connections.db``.
"""

import pytest

from ultron.core.agents import security as agent_security
from ultron.core.agents.simple import (
    detect_association_intent,
    handle_association,
    handle_remember,
)
from ultron.core.learning import associations as assoc
from ultron.core.tools.memory import graph, sqlite
from ultron.core.tools.registry import get_tool
from ultron.core.types import Role
from ultron.security import SecurityBoundary
from ultron.security.models import Decision, RiskTier


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    """Points the connections, flat-fact, and graph stores at throwaway files."""
    monkeypatch.setattr(assoc, "CONNECTIONS_DB_PATH", tmp_path / "conn.db")
    monkeypatch.setattr(sqlite, "MEMORY_DB_PATH", tmp_path / "memory.db")
    monkeypatch.setattr(graph, "MEMORY_DB_PATH", tmp_path / "memory.db")
    # The schema is normally created at module import (against the real path),
    # so create the tables in the throwaway file before storing anything.
    sqlite.init_memory_db()
    return tmp_path


@pytest.fixture()
def interactive_mode(monkeypatch):
    monkeypatch.setattr(
        agent_security, "_boundary", SecurityBoundary(mode="interactive")
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_extract_proper_nouns_strip_sentence_starters():
    feats = assoc.extract_fact_features(
        "The Medici family ruled Florence during the Renaissance"
    )
    assert feats["proper"] == {"medici", "florence", "renaissance"}


def test_extract_keywords_and_domains():
    feats = assoc.extract_fact_features(
        "The Medici used banking for political power in Italy"
    )
    assert "medici" in feats["keywords"]
    assert "political" in feats["keywords"]
    assert feats["domains"] == {"politics"}


def test_extract_domains_art():
    feats = assoc.extract_fact_features("Renaissance painting flourished in Florence")
    assert "art" in feats["domains"]


# ---------------------------------------------------------------------------
# Scoring + bridges (the Renaissance ↔ Medici politics scenario)
# ---------------------------------------------------------------------------


def test_concept_bridge_connects_art_to_politics(stores):
    # Two facts with ZERO shared words — only the curated bridge connects them.
    art = "Renaissance painting flourished in Florence"
    politics = "The Medici used banking for political power"
    conn = assoc._score_pair(
        assoc.extract_fact_features(art),
        assoc.extract_fact_features(politics),
    )
    assert conn is not None
    assert conn["relation"] == "concept_bridge"
    assert conn["cross_domain"] is True
    assert any("bridge" in r for r in conn["reasons"])


def test_shared_topic_connection(stores):
    a = "The Medici family ruled Florence during the Renaissance"
    b = "The Medici were powerful bankers who influenced Italian politics"
    conn = assoc._score_pair(
        assoc.extract_fact_features(a),
        assoc.extract_fact_features(b),
    )
    assert conn is not None
    assert conn["relation"] == "shared_topic"
    assert conn["cross_domain"] is True
    assert conn["score"] >= 2.0


def test_below_threshold_returns_none(stores):
    a = "The Medici family ruled Florence during the Renaissance"
    b = "My cat likes to sleep on the windowsill"
    conn = assoc._score_pair(
        assoc.extract_fact_features(a),
        assoc.extract_fact_features(b),
    )
    assert conn is None


def test_keyword_only_connection_labels_shared_keywords(stores):
    # Two facts sharing only generic keywords (no proper nouns, no domains,
    # no bridges) must not be mislabeled as a shared domain.
    a = "people enjoy hiking trails"
    b = "many hiking trails exist"
    conn = assoc._score_pair(
        assoc.extract_fact_features(a),
        assoc.extract_fact_features(b),
    )
    assert conn is not None
    assert conn["relation"] == "shared_keywords"
    assert conn["cross_domain"] is False


def test_shared_topic_reasons_not_duplicated(stores):
    a = "The Medici family ruled Florence during the Renaissance"
    b = "The Medici were powerful bankers who influenced Italian politics"
    conn = assoc._score_pair(
        assoc.extract_fact_features(a),
        assoc.extract_fact_features(b),
    )
    reasons = conn["reasons"]
    assert len([r for r in reasons if r == "shared: medici"]) == 1


def test_sixteenth_century_extracted_as_topic(stores):
    feats = assoc.extract_fact_features(
        "The Medici influenced 16th century Italian politics"
    )
    assert "16th century" in feats["topics"]
    assert "history" in feats["domains"]


def test_sixteenth_century_bridge_fires(stores):
    # Regression: "16th century" was a dead CONCEPT_BRIDGES key — extraction
    # never produced it as a topic, so the proposal's own example (16th-century
    # politics) never connected. Now it must bridge to the Reformation.
    a = "Renaissance art flourished in 16th century Italy"
    b = "The Reformation challenged the Church"
    conn = assoc._score_pair(
        assoc.extract_fact_features(a),
        assoc.extract_fact_features(b),
    )
    assert conn is not None
    assert conn["relation"] == "concept_bridge"
    assert any("16th century" in r for r in conn["reasons"])


# ---------------------------------------------------------------------------
# Announcements on remember
# ---------------------------------------------------------------------------


def test_connect_new_fact_announces_cross_domain(stores):
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    announcement = assoc.connect_new_fact(
        "The Medici were powerful bankers who influenced Italian politics"
    )
    assert "🔗 Connected to existing memories" in announcement
    assert "cross-domain" in announcement
    assert "medici" in announcement.lower()


def test_connect_new_fact_silent_without_matches(stores):
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    assert assoc.connect_new_fact("My cat likes to sleep on the windowsill") == ""


def test_handle_remember_appends_announcement(stores, interactive_mode):
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    msg = handle_remember(
        "The Medici were powerful bankers who influenced Italian politics"
    )
    assert msg.role == Role.ASSISTANT
    assert "🔗 Connected to existing memories" in msg.content


def test_handle_remember_first_fact_no_announcement(stores, interactive_mode):
    msg = handle_remember("The Medici family ruled Florence during the Renaissance")
    assert "🔗 Connected" not in msg.content


def test_handle_remember_survives_corrupt_connections_db(stores, interactive_mode, monkeypatch, tmp_path):
    # The learning layer is best-effort: an unwritable connections store must
    # never break remembering.
    monkeypatch.setattr(assoc, "CONNECTIONS_DB_PATH", tmp_path / "nope" / "x.db")
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    msg = handle_remember(
        "The Medici were powerful bankers who influenced Italian politics"
    )
    assert msg.content.startswith("Remembered:")


def test_corpus_dedups_identical_facts(stores):
    sqlite.add_memory("The Medici ruled Florence")
    sqlite.add_memory("The Medici ruled Florence")
    assert len(assoc._facts_corpus()) == 1


# ---------------------------------------------------------------------------
# Transitive discovery
# ---------------------------------------------------------------------------


def test_discover_connections_transitive_novel_link(stores, interactive_mode):
    sqlite.add_memory("Botticelli painted frescoes")
    sqlite.add_memory("Botticelli knew Leonardo")
    sqlite.add_memory("Leonardo studied anatomy")
    out = assoc.discover_connections()
    assert "2 direct and 1 novel transitive" in out
    assert "novel link through" in out
    assert "Leonardo studied anatomy" in out


def test_discover_requires_two_facts(stores):
    sqlite.add_memory("Only one fact here")
    assert "at least two stored facts" in assoc.discover_connections()


def test_memory_connections_empty_store(stores, interactive_mode):
    assert "No memories stored yet" in assoc.memory_connections()


# ---------------------------------------------------------------------------
# Registered tools + security
# ---------------------------------------------------------------------------


def test_tools_registered():
    for name in (
        "memory_connections",
        "related_facts",
        "discover_connections",
        "explain_relation",
    ):
        tool = get_tool(name)
        assert tool is not None
        assert callable(tool)


def test_tools_classified_low_risk():
    boundary = SecurityBoundary(mode="interactive")
    for name in (
        "memory_connections",
        "related_facts",
        "discover_connections",
        "explain_relation",
    ):
        assert boundary.classify_action(name, "sample target") == RiskTier.LOW
        verdict = boundary.check(name, "", "sample target")
        assert verdict.decision == Decision.ALLOW, f"{name} should auto-run"


def test_memory_connections_summary(stores, interactive_mode):
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    sqlite.add_memory("The Medici were powerful bankers who influenced Italian politics")
    out = assoc.memory_connections()
    assert "facts stored" in out
    assert "cross-domain" in out


def test_memory_connections_around_topic(stores, interactive_mode):
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    sqlite.add_memory("The Medici were powerful bankers who influenced Italian politics")
    out = assoc.memory_connections(topic="medici")
    assert "Memory connections around 'medici'" in out
    assert "cross-domain" in out


def test_related_facts(stores, interactive_mode):
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    out = assoc.related_facts(
        "The Medici were powerful bankers who influenced Italian politics"
    )
    assert "connects to" in out
    assert "medici" in out.lower()


def test_explain_relation(stores):
    out = assoc.explain_relation("Renaissance painting", "The Medici")
    assert "connected" in out
    assert "bridge" in out or "shared" in out


def test_explain_relation_no_connection(stores):
    out = assoc.explain_relation("My cat", "The London Tube")
    assert "can't see a connection" in out


# ---------------------------------------------------------------------------
# Agent-level detector + handlers
# ---------------------------------------------------------------------------


def test_detect_relate_intent():
    result = detect_association_intent(
        "how is renaissance art related to the medici"
    )
    assert result["action"] == "relate"
    assert "renaissance art" in result["a"]
    assert "medici" in result["b"]


def test_detect_connections_intent_with_topic():
    result = detect_association_intent("connections for renaissance")
    assert result["action"] == "connections"
    assert result["topic"] == "renaissance"


def test_detect_discover_intent():
    assert detect_association_intent("discover new connections")["action"] == "discover"


def test_detect_negatives():
    for text in (
        "what time is it",
        "run ls -la",
        "tell me a joke",
        "remember that I like coffee",
        "how heavy is pip install",
    ):
        assert detect_association_intent(text) is None, text


def test_handle_association_relate(stores, interactive_mode):
    msg = handle_association("relate", a="Renaissance painting", b="The Medici")
    assert msg.role == Role.ASSISTANT
    assert "connected" in msg.content


def test_handle_association_relate_asks_for_subjects(stores, interactive_mode):
    msg = handle_association("relate")
    assert "Which two things" in msg.content


def test_handle_association_connections(stores, interactive_mode):
    sqlite.add_memory("The Medici family ruled Florence during the Renaissance")
    sqlite.add_memory("The Medici were powerful bankers who influenced Italian politics")
    msg = handle_association("connections", topic="medici")
    assert "Memory connections around 'medici'" in msg.content


def test_handle_association_discover(stores, interactive_mode):
    sqlite.add_memory("Botticelli painted frescoes")
    sqlite.add_memory("Botticelli knew Leonardo")
    msg = handle_association("discover")
    assert "Discovered" in msg.content
