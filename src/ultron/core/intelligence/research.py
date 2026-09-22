"""
ultron.core.intelligence.research
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grounded Research Pipeline: Evidence extraction, source authority ranking,
freshness categorization, conflict detection, and evidence-grounded synthesis.

Architecture:
    User Request / Web Search Results
        ↓
    Evidence Extraction (extract_evidence_records)
        ↓
    Source Ranking (rank_sources: Official > Authoritative Docs > Major Media > Aggregators > Forums)
        ↓
    Freshness Categorization (evaluate_freshness: Confirmed Current > Reported > Rumors)
        ↓
    Conflict Detection (detect_conflicts: identifies divergent claims)
        ↓
    Grounded Research Bundle & Structured Synthesis (build_research_bundle / format_research_context)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class SourceQualityTier(str, Enum):
    """Authority tier for information sources."""
    OFFICIAL = "official"             # Official doc/site (python.org, github.com, apple.com, etc.)
    AUTHORITATIVE = "authoritative"   # Established technical publications / docs (mdn, arxiv, wikipedia)
    MAJOR_MEDIA = "major_media"       # Reputable news / media outlets (reuters, bbc, verge, ars)
    AGGREGATOR = "aggregator"         # Aggregators / community blogs (medium, dev.to)
    FORUM = "forum"                   # Forums / discussions / rumors (reddit, x.com, hackernews)
    UNKNOWN = "unknown"               # Unclassified sources


class FreshnessCategory(str, Enum):
    """Temporal freshness and factual status."""
    CONFIRMED_CURRENT = "confirmed_current"  # Explicitly released / active / current version
    REPORTED = "reported"                    # Formally reported / scheduled / in progress
    RUMOR_OR_SPECULATIVE = "speculative"     # Leaks, rumors, speculation, unconfirmed
    STALE = "stale"                          # Outdated / deprecated / legacy versions


class EvidenceRecord(BaseModel):
    """
    Normalized, grounded piece of evidence extracted from search or page fetches.
    """
    claim: str
    source_title: str = ""
    source_url: str = ""
    domain: str = ""
    source_tier: SourceQualityTier = SourceQualityTier.UNKNOWN
    freshness: FreshnessCategory = FreshnessCategory.REPORTED
    authority_score: float = 0.5
    raw_snippet: str = ""
    contradicts: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    key_facts: list[str] = Field(default_factory=list)


# Domain classification rules
_OFFICIAL_DOMAINS = frozenset({
    "python.org", "pypi.org", "github.com", "docs.github.com", "gitlab.com",
    "apple.com", "developer.apple.com", "microsoft.com", "learn.microsoft.com",
    "google.com", "developers.google.com", "mozilla.org", "developer.mozilla.org",
    "rust-lang.org", "golang.org", "go.dev", "nodejs.org", "anthropic.com",
    "openai.com", "deepmind.google", "meta.com", "react.dev", "vuejs.org",
    "kubernetes.io", "docker.com", "apache.org", "linuxfoundation.org",
    "iso.org", "w3.org", "ietf.org", "whitehouse.gov", "nasa.gov",
})

_AUTHORITATIVE_DOMAINS = frozenset({
    "wikipedia.org", "arxiv.org", "stackoverflow.com", "stackexchange.com",
    "en.wikipedia.org", "w3schools.com", "geeksforgeeks.org", "realpython.com",
    "docs.rs", "pkg.go.dev", "packagist.org", "npmjs.com", "crates.io",
    "bloomberg.com", "nature.com", "science.org", "ieee.org", "acm.org",
})

_MAJOR_MEDIA_DOMAINS = frozenset({
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "theverge.com", "arstechnica.com", "techcrunch.com", "wired.com",
    "zdnet.com", "theregister.com", "engadget.com", "venturebeat.com",
    "wsj.com", "bloomberg.com", "forbes.com", "theguardian.com",
})

_AGGREGATOR_DOMAINS = frozenset({
    "medium.com", "dev.to", "substack.com", "hackernoon.com", "hashnode.dev",
    "towardsdatascience.com", "freecodecamp.org", "dzone.com", "infoq.com",
})

_FORUM_DOMAINS = frozenset({
    "reddit.com", "twitter.com", "x.com", "news.ycombinator.com", "threads.net",
    "quora.com", "discussions.apple.com", "community.openai.com", "discord.com",
})


def extract_domain(url: str) -> str:
    """Extracts clean base domain without www from URL."""
    if not url:
        return ""
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().removeprefix("www.")
        return netloc.split(":")[0]
    except (ValueError, AttributeError):
        return ""


def classify_source_tier(domain: str) -> tuple[SourceQualityTier, float]:
    """
    Returns (SourceQualityTier, authority_score).
    Scores range from 0.95 (official) to 0.35 (forums).
    """
    clean_domain = domain.lower().strip().removeprefix("www.")

    # Check suffix/subdomains (e.g. docs.python.org -> python.org)
    parts = clean_domain.split(".")
    base_domain = ".".join(parts[-2:]) if len(parts) >= 2 else clean_domain

    if clean_domain in _OFFICIAL_DOMAINS or base_domain in _OFFICIAL_DOMAINS or clean_domain.endswith(".gov"):
        return SourceQualityTier.OFFICIAL, 0.95
    if clean_domain in _AUTHORITATIVE_DOMAINS or base_domain in _AUTHORITATIVE_DOMAINS or clean_domain.endswith(".edu"):
        return SourceQualityTier.AUTHORITATIVE, 0.85
    if clean_domain in _MAJOR_MEDIA_DOMAINS or base_domain in _MAJOR_MEDIA_DOMAINS:
        return SourceQualityTier.MAJOR_MEDIA, 0.75
    if clean_domain in _AGGREGATOR_DOMAINS or base_domain in _AGGREGATOR_DOMAINS:
        return SourceQualityTier.AGGREGATOR, 0.60
    if clean_domain in _FORUM_DOMAINS or base_domain in _FORUM_DOMAINS:
        return SourceQualityTier.FORUM, 0.40

    return SourceQualityTier.UNKNOWN, 0.50


_SPECULATIVE_MARKERS = re.compile(
    r"\b(rumor|leak|speculat\w*|alleged\w*|unconfirmed|claim\w*|might\s+be|could\s+be|expected\s+to|reportedly)\b",
    re.IGNORECASE,
)

_CURRENT_CONFIRMED_MARKERS = re.compile(
    r"\b(released|announced|officially|available\s+now|stable|production|launch(?:ed)?|version\s+\d+|v\d+)\b",
    re.IGNORECASE,
)

_STALE_MARKERS = re.compile(
    r"\b(deprecated|discontinued|legacy|unsupported|end\s+of\s+life|eol|sunsetted)\b",
    re.IGNORECASE,
)


def evaluate_freshness(text: str, domain: str = "") -> FreshnessCategory:
    """Evaluates the temporal / factual reliability of a snippet or claim."""
    lower_text = text.lower()
    if _STALE_MARKERS.search(lower_text):
        return FreshnessCategory.STALE
    if _SPECULATIVE_MARKERS.search(lower_text):
        return FreshnessCategory.RUMOR_OR_SPECULATIVE
    if _CURRENT_CONFIRMED_MARKERS.search(lower_text):
        return FreshnessCategory.CONFIRMED_CURRENT
    return FreshnessCategory.REPORTED


def parse_search_snippets(observation: str) -> list[dict[str, str]]:
    """
    Parses search output formatted by `search_web` (or raw format):
    e.g.
    1. Snippet text...
       **Title:** Example Title
       **Link:** https://example.com
    """
    results: list[dict[str, str]] = []
    blocks = re.split(r"\n\s*\n", observation.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        title_match = re.search(r"\*\*Title:\*\*\s*(.+)$", block, re.MULTILINE)
        link_match = re.search(r"\*\*Link:\*\*\s*(.+)$", block, re.MULTILINE)

        title = title_match.group(1).strip() if title_match else ""
        link = link_match.group(1).strip() if link_match else ""

        # Extract snippet by removing Title and Link lines
        snippet_lines = []
        for line in block.splitlines():
            line_str = line.strip()
            if line_str.startswith(("**Title:**", "**Link:**")):
                continue
            # Strip numeric prefix if present (e.g. "1. ")
            cleaned_line = re.sub(r"^\d+\.\s*", "", line_str)
            if cleaned_line:
                snippet_lines.append(cleaned_line)

        snippet = " ".join(snippet_lines).strip()
        if snippet or title or link:
            results.append({"snippet": snippet, "title": title, "link": link})

    return results


def extract_key_facts(text: str) -> list[str]:
    """Extracts key factual phrases / numbers / dates / version strings."""
    facts = []
    # Dates / years
    years = re.findall(r"\b(?:202[0-9]|19[0-9]{2})\b", text)
    if years:
        facts.append(f"Years mentioned: {', '.join(sorted(set(years)))}")
    # Version numbers
    versions = re.findall(r"\b(?:v\d+(?:\.\d+)+|\b(?:version|python|release)\s+\d+(?:\.\d+)*)\b", text, re.IGNORECASE)
    if versions:
        facts.append(f"Versions: {', '.join(sorted(set(versions))[:3])}")
    # Numbers with units / currency / stats
    stats = re.findall(r"\b(?:\$\d+(?:\.\d+)?|\d+%\b|\d+\s*(?:gb|mb|ghz|billion|million))\b", text, re.IGNORECASE)
    if stats:
        facts.append(f"Metrics: {', '.join(sorted(set(stats))[:3])}")
    return facts


def extract_evidence_records(observation: str) -> list[EvidenceRecord]:
    """
    Extracts structured EvidenceRecord items from search or observation text.
    """
    parsed_snippets = parse_search_snippets(observation)
    records: list[EvidenceRecord] = []

    if not parsed_snippets:
        # If not in block format, treat the observation as a single snippet
        domain = ""
        tier, score = SourceQualityTier.UNKNOWN, 0.5
        freshness = evaluate_freshness(observation)
        records.append(
            EvidenceRecord(
                claim=observation[:300].strip(),
                raw_snippet=observation,
                domain=domain,
                source_tier=tier,
                authority_score=score,
                freshness=freshness,
                key_facts=extract_key_facts(observation),
            )
        )
        return records

    for item in parsed_snippets:
        snippet = item["snippet"]
        title = item["title"]
        link = item["link"]
        domain = extract_domain(link)
        tier, auth_score = classify_source_tier(domain)
        freshness = evaluate_freshness(f"{title} {snippet}", domain)

        # Base confidence calculation combining authority and freshness
        freshness_mod = {
            FreshnessCategory.CONFIRMED_CURRENT: 1.0,
            FreshnessCategory.REPORTED: 0.85,
            FreshnessCategory.RUMOR_OR_SPECULATIVE: 0.60,
            FreshnessCategory.STALE: 0.40,
        }.get(freshness, 0.7)

        confidence = round(auth_score * freshness_mod, 2)

        records.append(
            EvidenceRecord(
                claim=snippet if len(snippet) <= 250 else snippet[:247] + "...",
                source_title=title,
                source_url=link,
                domain=domain,
                source_tier=tier,
                authority_score=auth_score,
                freshness=freshness,
                raw_snippet=snippet,
                confidence=confidence,
                key_facts=extract_key_facts(snippet),
            )
        )

    # Sort records by authority and confidence descending
    records.sort(key=lambda r: (r.authority_score, r.confidence), reverse=True)
    return records


def detect_conflicts(records: list[EvidenceRecord]) -> list[dict[str, Any]]:
    """
    Detects potential conflicting evidence among records (e.g. confirmed vs speculative,
    opposing release years, conflicting status claims).
    """
    conflicts: list[dict[str, Any]] = []
    if len(records) < 2:
        return conflicts

    # Look for status divergence (Confirmed vs Speculative on similar topics)
    confirmed_records = [r for r in records if r.freshness == FreshnessCategory.CONFIRMED_CURRENT]
    speculative_records = [r for r in records if r.freshness == FreshnessCategory.RUMOR_OR_SPECULATIVE]

    if confirmed_records and speculative_records:
        for spec in speculative_records:
            for conf in confirmed_records:
                conflicts.append({
                    "topic": "Status / Release Certainty",
                    "conflict_type": "confirmed_vs_speculative",
                    "claim_a": f"[{conf.domain or 'Confirmed Source'}] {conf.claim[:120]}",
                    "claim_b": f"[{spec.domain or 'Speculative Source'}] {spec.claim[:120]}",
                    "resolution": (
                        f"Official / confirmed source ({conf.domain or conf.source_title}) "
                        f"takes precedence over rumor/speculative reporting ({spec.domain or spec.source_title})."
                    ),
                })

    # Look for date/year conflicts across snippets
    year_map: dict[str, list[EvidenceRecord]] = {}
    for r in records:
        years = re.findall(r"\b(202[0-9])\b", r.raw_snippet)
        for y in set(years):
            year_map.setdefault(y, []).append(r)

    if len(year_map) > 1:
        years = sorted(year_map.keys())
        # If multiple years are cited as releases/timelines
        r_first = year_map[years[0]][0]
        r_last = year_map[years[-1]][0]
        if r_first.domain != r_last.domain:
            conflicts.append({
                "topic": "Timeline / Year Divergence",
                "conflict_type": "timeline_discrepancy",
                "claim_a": f"{years[0]} cited by {r_first.domain or 'source'}",
                "claim_b": f"{years[-1]} cited by {r_last.domain or 'source'}",
                "resolution": (
                    f"Prioritize higher authority ({r_first.domain} [{r_first.source_tier.value}] "
                    f"vs {r_last.domain} [{r_last.source_tier.value}]) and explicit version announcements."
                ),
            })

    return conflicts


def build_research_context(records: list[EvidenceRecord], conflicts: list[dict[str, Any]]) -> str:
    """
    Builds a grounded, evidence-rich structured prompt block for the synthesizer LLM.
    """
    if not records:
        return "No structured evidence records available."

    lines = ["### Grounded Research Evidence:"]
    for i, r in enumerate(records[:5], start=1):
        tier_str = r.source_tier.value.upper()
        fresh_str = r.freshness.value.replace("_", " ").upper()
        domain_str = f" ({r.domain})" if r.domain else ""
        lines.append(
            f"{i}. [{tier_str} | {fresh_str}] {r.source_title}{domain_str}\n"
            f"   - Evidence: {r.raw_snippet}\n"
            f"   - Confidence: {r.confidence:.2f} | URL: {r.source_url}"
        )

    if conflicts:
        lines.append("\n### Identified Discrepancies & Conflict Resolutions:")
        for c in conflicts[:3]:
            lines.append(
                f"- **{c['topic']}**: {c['claim_a']} VS {c['claim_b']}\n"
                f"  *Resolution Strategy:* {c['resolution']}"
            )

    return "\n".join(lines)
