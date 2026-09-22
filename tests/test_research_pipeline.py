"""
Tests for Grounded Research Pipeline (ultron.core.intelligence.research).
"""

from __future__ import annotations

from ultron.core.intelligence.research import (
    EvidenceRecord,
    FreshnessCategory,
    SourceQualityTier,
    build_research_context,
    classify_source_tier,
    detect_conflicts,
    evaluate_freshness,
    extract_domain,
    extract_evidence_records,
    parse_search_snippets,
)


def test_extract_domain():
    assert extract_domain("https://docs.python.org/3/whatsnew/3.12.html") == "docs.python.org"
    assert extract_domain("http://www.github.com/torvalds/linux") == "github.com"
    assert extract_domain("https://reuters.com/technology") == "reuters.com"
    assert extract_domain("reddit.com/r/python") == "reddit.com"


def test_classify_source_tier():
    tier, score = classify_source_tier("docs.python.org")
    assert tier == SourceQualityTier.OFFICIAL
    assert score >= 0.9

    tier, score = classify_source_tier("wikipedia.org")
    assert tier == SourceQualityTier.AUTHORITATIVE
    assert score >= 0.8

    tier, score = classify_source_tier("theverge.com")
    assert tier == SourceQualityTier.MAJOR_MEDIA
    assert score >= 0.7

    tier, score = classify_source_tier("medium.com")
    assert tier == SourceQualityTier.AGGREGATOR
    assert score >= 0.55

    tier, score = classify_source_tier("reddit.com")
    assert tier == SourceQualityTier.FORUM
    assert score <= 0.5


def test_evaluate_freshness():
    assert evaluate_freshness("Python 3.12 was officially released in October 2023.") == FreshnessCategory.CONFIRMED_CURRENT
    assert evaluate_freshness("Rumor suggests the new chip might be released next year.") == FreshnessCategory.RUMOR_OR_SPECULATIVE
    assert evaluate_freshness("Python 2.7 is deprecated and reached end of life.") == FreshnessCategory.STALE
    assert evaluate_freshness("The conference is scheduled to begin in June.") == FreshnessCategory.REPORTED


def test_parse_search_snippets():
    obs = (
        "1. Python 3.12 introduces improved error messages and isolated subinterpreters.\n"
        "   **Title:** What's New In Python 3.12 - Python documentation\n"
        "   **Link:** https://docs.python.org/3/whatsnew/3.12.html\n\n"
        "2. Leaked roadmap claims Python 4.0 will eliminate GIL entirely.\n"
        "   **Title:** Python 4.0 Rumors\n"
        "   **Link:** https://reddit.com/r/Python/comments/123\n"
    )
    snippets = parse_search_snippets(obs)
    assert len(snippets) == 2
    assert "Python 3.12" in snippets[0]["snippet"]
    assert snippets[0]["title"] == "What's New In Python 3.12 - Python documentation"
    assert snippets[0]["link"] == "https://docs.python.org/3/whatsnew/3.12.html"


def test_extract_evidence_records():
    obs = (
        "1. Python 3.12 is the latest stable release featuring significant performance gains.\n"
        "   **Title:** Python 3.12 Release\n"
        "   **Link:** https://python.org/downloads/release/python-3120/\n\n"
        "2. Rumors say Python might drop syntax.\n"
        "   **Title:** Forum Discussion\n"
        "   **Link:** https://reddit.com/r/python\n"
    )
    records = extract_evidence_records(obs)
    assert len(records) == 2
    # First record should be official doc with high authority and confidence
    assert records[0].source_tier == SourceQualityTier.OFFICIAL
    assert records[0].freshness == FreshnessCategory.CONFIRMED_CURRENT
    assert records[0].authority_score > records[1].authority_score


def test_detect_conflicts():
    records = [
        EvidenceRecord(
            claim="Official announcement: product released and stable.",
            source_title="Official Docs",
            source_url="https://python.org/release",
            domain="python.org",
            source_tier=SourceQualityTier.OFFICIAL,
            freshness=FreshnessCategory.CONFIRMED_CURRENT,
            raw_snippet="Released in 2023.",
        ),
        EvidenceRecord(
            claim="Rumor and unconfirmed leaks say product delayed to 2025.",
            source_title="Tech Leaks Blog",
            source_url="https://reddit.com/r/leaks",
            domain="reddit.com",
            source_tier=SourceQualityTier.FORUM,
            freshness=FreshnessCategory.RUMOR_OR_SPECULATIVE,
            raw_snippet="Delayed to 2025 according to leak.",
        ),
    ]
    conflicts = detect_conflicts(records)
    assert len(conflicts) > 0
    assert any("confirmed_vs_speculative" == c["conflict_type"] for c in conflicts)
    assert "Official / confirmed source" in conflicts[0]["resolution"]


def test_build_research_context():
    records = [
        EvidenceRecord(
            claim="Python 3.12 released.",
            source_title="Python 3.12 Documentation",
            source_url="https://docs.python.org",
            domain="docs.python.org",
            source_tier=SourceQualityTier.OFFICIAL,
            freshness=FreshnessCategory.CONFIRMED_CURRENT,
            raw_snippet="Python 3.12 released with subinterpreters.",
            confidence=0.95,
        )
    ]
    conflicts = [{
        "topic": "Release date",
        "claim_a": "2023",
        "claim_b": "2024",
        "resolution": "Use official docs",
    }]
    ctx = build_research_context(records, conflicts)
    assert "### Grounded Research Evidence:" in ctx
    assert "OFFICIAL | CONFIRMED CURRENT" in ctx
    assert "### Identified Discrepancies" in ctx
