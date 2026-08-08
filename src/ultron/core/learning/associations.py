"""ultron.core.learning.associations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Personalized learning: correlate stored facts across domains.

Every newly stored memory is compared against the existing corpus (flat
facts + rendered knowledge-graph edges) and genuine connections — especially
cross-domain ones — are surfaced with human-readable reasons. Extraction and
scoring are fully deterministic: proper-noun runs, content keywords, and
curated domain tags feed a scoring model that rewards shared proper nouns,
shared keywords, curated concept-bridge hits (e.g. ``medici ↔ renaissance``),
and shared domains.

Everything is grounded in stored text — no LLM, no hallucination. The
discovery layer also finds transitive novel links (A-B, B-C ⇒ A-C via B).
See docs/personalized-learning.md.
"""

from __future__ import annotations

import re
import sqlite3

from ultron.core.tools.memory import graph
from ultron.core.tools.memory.sqlite import get_all_memories
from ultron.core.tools.paths import ALLOWED_BASE_DIR

# Connection store (source, target, relation, strength, reason, cross_domain).
# Tests may repoint this at a temp file.
CONNECTIONS_DB_PATH = ALLOWED_BASE_DIR / ".ultron_connections.db"

# Facts that can be remembered past the corpus-scan bound.
MAX_CORPUS_SCAN = 300

# --- Domain keyword dictionaries -----------------------------------------
# A fact is tagged with a domain when any of its keywords appear in it.
DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "art": frozenset({
        "art", "artist", "painting", "painter", "sculpture", "sculptor",
        "renaissance", "baroque", "impressionism", "museum", "gallery",
        "canvas", "fresco", "portrait", "masterpiece",
    }),
    "politics": frozenset({
        "politics", "political", "government", "election", "parliament",
        "senate", "congress", "treaty", "diplomacy", "policy", "minister",
        "president", "vote", "legislation",
    }),
    "history": frozenset({
        "history", "historical", "century", "16th century", "medieval", "ancient",
        "dynasty", "empire", "era", "civilization", "archaeology", "artifact",
    }),
    "tech": frozenset({
        "computer", "software", "programming", "code", "python", "javascript",
        "algorithm", "internet", "startup", "developer", "github", "database",
        "machine learning",
    }),
    "science": frozenset({
        "science", "physics", "chemistry", "biology", "astronomy",
        "experiment", "research", "quantum", "genome", "theory", "mathematics",
    }),
    "food": frozenset({
        "food", "cooking", "recipe", "cuisine", "restaurant", "chef", "wine",
        "coffee", "baking", "dish",
    }),
    "music": frozenset({
        "music", "song", "album", "concert", "guitar", "piano", "jazz",
        "classical", "orchestra", "composer",
    }),
    "sports": frozenset({
        "sport", "sports", "football", "soccer", "basketball", "cricket",
        "tennis", "olympics", "team", "tournament",
    }),
    "travel": frozenset({
        "travel", "trip", "airport", "hotel", "vacation", "tourist",
        "journey", "visit",
    }),
    "business": frozenset({
        "business", "company", "market", "finance", "investment", "stock",
        "economy", "revenue", "entrepreneur",
    }),
    "literature": frozenset({
        "book", "novel", "author", "writer", "poetry", "poem", "literature",
        "fiction", "reading",
    }),
    "religion": frozenset({
        "religion", "church", "temple", "faith", "prayer", "catholic",
        "bible", "monastery",
    }),
    "architecture": frozenset({
        "architecture", "building", "cathedral", "castle", "palace",
        "monument", "dome", "facade",
    }),
    "medicine": frozenset({
        "medicine", "doctor", "health", "disease", "hospital", "vaccine",
        "therapy", "patient", "surgery",
    }),
    "education": frozenset({
        "school", "university", "college", "student", "teacher", "professor",
        "degree", "education", "course",
    }),
    "film": frozenset({
        "film", "movie", "cinema", "director", "actor", "actress",
        "hollywood", "screenplay", "documentary",
    }),
}

# --- Cross-domain concept bridges ----------------------------------------
# concept -> related concepts in OTHER domains. Lets two facts with no
# shared words still connect (e.g. "renaissance" ↔ "medici").
CONCEPT_BRIDGES: dict[str, frozenset[str]] = {
    "renaissance": frozenset({"florence", "medici", "italy", "vatican", "patronage", "leonardo", "michelangelo", "16th century", "sculpture", "painting"}),
    "medici": frozenset({"florence", "renaissance", "banking", "patronage", "italy", "tuscany"}),
    "florence": frozenset({"renaissance", "medici", "italy", "tuscany", "art"}),
    "leonardo": frozenset({"renaissance", "florence", "milan", "painting", "art"}),
    "michelangelo": frozenset({"renaissance", "florence", "sculpture", "vatican"}),
    "italy": frozenset({"rome", "florence", "venice", "europe", "renaissance"}),
    "vatican": frozenset({"rome", "catholic", "michelangelo", "renaissance", "italy"}),
    "rome": frozenset({"italy", "vatican", "empire", "ancient"}),
    "16th century": frozenset({"renaissance", "reformation", "medici", "italy"}),
    "patronage": frozenset({"medici", "renaissance", "art", "funding"}),
    "banking": frozenset({"medici", "finance", "florence", "business"}),
    "catholic": frozenset({"vatican", "rome", "religion", "church", "reformation"}),
    "reformation": frozenset({"16th century", "catholic", "europe", "protestant"}),
    "ai": frozenset({"machine learning", "algorithm", "computer", "software"}),
    "machine learning": frozenset({"ai", "data", "algorithm", "computer"}),
    "quantum": frozenset({"physics", "science", "computer"}),
    "olympics": frozenset({"sports", "greece", "ancient", "competition"}),
    "greece": frozenset({"olympics", "ancient", "philosophy", "europe"}),
    "philosophy": frozenset({"greece", "thought", "science", "politics"}),
}

# Common sentence-start words that must not be captured as proper nouns
# (leading words of a capitalized run are stripped).
_SENTENCE_STARTERS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she",
    "we", "they", "you", "my", "his", "her", "our", "their", "there",
    "when", "what", "who", "how", "why", "but", "and", "or", "so", "in",
    "on", "at", "to", "of", "for", "with", "by", "is", "are", "was", "were",
    "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "might", "must", "please", "however", "meanwhile", "also", "then",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "while", "of", "to", "in",
    "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from", "up",
    "down", "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",        "because", "until", "which", "who", "whom", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "it", "its", "they", "them", "their", "what",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should", "may",
    "might", "must", "can", "could", "like", "love", "really",
})


# ---------------------------------------------------------------------------
# Connection store
# ---------------------------------------------------------------------------

_CONNECT_SCHEMA = """
CREATE TABLE IF NOT EXISTS connections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    strength REAL NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    cross_domain INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source, target, relation)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CONNECTIONS_DB_PATH))
    conn.executescript(_CONNECT_SCHEMA)
    return conn


def _persist(source: str, target: str, conn: dict) -> None:
    """Best-effort insert of one discovered connection (deduplicated)."""
    try:
        with _connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO connections "
                "(source, target, relation, strength, reason, cross_domain) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source,
                    target,
                    conn["relation"],
                    conn["score"],
                    "; ".join(conn["reasons"]),
                    int(conn["cross_domain"]),
                ),
            )
            db.commit()
    except (sqlite3.Error, OSError):
        pass


# ---------------------------------------------------------------------------
# Fact features
# ---------------------------------------------------------------------------

def extract_fact_features(text: str) -> dict:
    """
    Deterministically extracts (proper, keywords, domains) from a fact.

    - proper: capitalized multi-word runs, with sentence-start words
      stripped ("The Medici ruled…" → {"medici"}).
    - keywords: lower-cased non-stopword content words (length >= 3) plus
      domain-keyword hits.
    - domains: the curated domain tags whose keywords appear in the text.
    """
    lowered = text.lower()

    proper: set[str] = set()
    for match in re.finditer(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text):
        parts = match.group(0).split()
        while parts and parts[0].lower() in _SENTENCE_STARTERS:
            parts = parts[1:]
        if parts and len(parts[0]) >= 2:
            proper.add(" ".join(parts).lower())

    words = [
        w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]*", lowered)
        if w not in _STOPWORDS and len(w) >= 3
    ]
    keywords: set[str] = set(words)

    domains: set[str] = set()
    for domain, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if " " in kw:
                if kw in lowered:
                    domains.add(domain)
                    keywords.add(kw)
            elif re.search(rf"\b{re.escape(kw)}\b", lowered):
                domains.add(domain)
                keywords.add(kw)

    return {
        "proper": proper,
        "keywords": keywords,
        "domains": domains,
        "topics": proper | keywords,
    }


def _facts_corpus() -> list[str]:
    """The correlation corpus: flat facts + rendered triples, deduplicated."""
    seen: set[str] = set()
    corpus: list[str] = []
    for fact in [*get_all_memories(), *graph.get_all_triples()]:
        key = re.sub(r"\s+", " ", fact.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        corpus.append(fact.strip())
    return corpus[-MAX_CORPUS_SCAN:]


def _is_cross_domain(features_a: dict, features_b: dict) -> bool:
    """True when both facts carry domains but share none of them."""
    da = features_a["domains"]
    db = features_b["domains"]
    return bool(da and db and not (da & db))


def _score_pair(features_a: dict, features_b: dict) -> dict | None:
    """
    Scores one fact pair. Returns a connection dict (or None when below the
    threshold). See docs/personalized-learning.md for the scoring model.
    """
    shared_proper = features_a["proper"] & features_b["proper"]
    shared_kw = features_a["keywords"] & features_b["keywords"]
    shared_domains = features_a["domains"] & features_b["domains"]

    bridges: list[tuple[str, str]] = []
    for concept in features_a["topics"]:
        related = CONCEPT_BRIDGES.get(concept, ())
        for candidate in features_b["topics"]:
            if (
                candidate in related or concept in CONCEPT_BRIDGES.get(candidate, ())
            ) and (concept, candidate) not in bridges and (candidate, concept) not in bridges:
                bridges.append((concept, candidate))

    reasons: list[str] = []
    proper_score = 0.0
    for topic in sorted(shared_proper):
        proper_score += 1.0
        reasons.append(f"shared: {topic}")

    kw_score = 0.0
    kw_reasons: list[str] = []
    # Proper nouns are already reported above and are always keywords too,
    # so only report keywords that are not shared proper nouns.
    for keyword in sorted(shared_kw - shared_proper):
        kw_score += 0.4
        kw_reasons.append(f"shared: {keyword}")
    kw_score = min(kw_score, 0.8)
    reasons.extend(kw_reasons[:3])

    bridge_score = min(0.7 * len(bridges), 1.4)
    reasons.extend(f"bridge: {c} ↔ {d}" for c, d in bridges[:2])

    domain_score = 0.3 if shared_domains else 0.0
    if shared_domains:
        reasons.append(f"shared domain: {', '.join(sorted(shared_domains))}")

    score = proper_score + kw_score + bridge_score + domain_score
    if score < 0.7:
        return None

    if shared_proper:
        relation = "shared_topic"
    elif bridges:
        relation = "concept_bridge"
    elif shared_kw:
        relation = "shared_keywords"
    else:
        relation = "shared_domain"

    return {
        "score": round(score, 2),
        "relation": relation,
        "reasons": reasons[:4],
        "cross_domain": _is_cross_domain(features_a, features_b),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_connections(fact_text: str, limit: int = 5) -> list[dict]:
    """
    Correlates one fact against the stored corpus and returns its strongest
    connections (excluding self-matches and duplicates).
    """
    features = extract_fact_features(fact_text)
    key = re.sub(r"\s+", " ", fact_text.strip().lower())
    results: list[dict] = []
    for fact in _facts_corpus():
        if re.sub(r"\s+", " ", fact.lower()) == key:
            continue
        conn = _score_pair(features, extract_fact_features(fact))
        if conn:
            results.append({"fact": fact, **conn})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def connect_new_fact(fact_text: str) -> str:
    """
    Persists the connections of a freshly stored fact and returns a
    user-facing announcement ("" when nothing connects).
    """
    connections = find_connections(fact_text, limit=3)
    if not connections:
        return ""
    for conn in connections:
        _persist(fact_text, conn["fact"], conn)
    lines = ["🔗 Connected to existing memories:"]
    for conn in connections:
        cross = " (cross-domain)" if conn["cross_domain"] else ""
        lines.append(
            f'• "{conn["fact"]}" — {"; ".join(conn["reasons"])}{cross}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registered tools
# ---------------------------------------------------------------------------

def memory_connections(topic: str = "") -> str:
    """
    Connection map around a topic (or a store summary when no topic is
    given). Scans live, so newly remembered facts appear immediately.
    """
    corpus = _facts_corpus()
    if not corpus:
        return "No memories stored yet — nothing to connect."

    if topic.strip():
        lowered = topic.strip().lower()
        lines = [f"Memory connections around '{topic.strip()}':"]
        found = 0
        for fact in corpus:
            if lowered not in fact.lower():
                continue
            links = find_connections(fact, limit=3)
            if not links:
                continue
            found += 1
            lines.append(f"\n- \"{fact}\"")
            for conn in links:
                cross = " (cross-domain)" if conn["cross_domain"] else ""
                lines.append(f"  ⟷ \"{conn['fact']}\" — {'; '.join(conn['reasons'])}{cross}")
        if found == 0:
            return f"No memories or connections found for '{topic.strip()}'."
        return "\n".join(lines)

    # Summary over the whole store.
    cross_domain = 0
    connected = 0
    for fact in corpus:
        links = find_connections(fact, limit=1)
        if links:
            connected += 1
            if any(l["cross_domain"] for l in links):
                cross_domain += 1
    return (
        f"Memory connections: {len(corpus)} facts stored, {connected} with "
        f"links to other facts, {cross_domain} cross-domain. "
        "Ask 'connections for <topic>' to see the map around a topic."
    )


def related_facts(fact_text: str) -> str:
    """
    What one fact relates to, on demand — the same correlation the store
    announcement uses.
    """
    if not fact_text or not fact_text.strip():
        return "Error: related_facts needs a fact to correlate."
    connections = find_connections(fact_text, limit=5)
    if not connections:
        return f"No stored memories connect to \"{fact_text}\" yet."
    lines = [f"\"{fact_text}\" connects to:"]
    for conn in connections:
        cross = " (cross-domain)" if conn["cross_domain"] else ""
        lines.append(f'• "{conn["fact"]}" — {"; ".join(conn["reasons"])}{cross}')
    return "\n".join(lines)


def explain_relation(a: str, b: str) -> str:
    """
    Explains how two subjects relate, whether or not they are stored.
    Analyzes the pair directly through the same feature/scoring model.
    """
    if not a.strip() or not b.strip():
        return "Error: explain_relation needs two subjects."
    conn = _score_pair(extract_fact_features(a), extract_fact_features(b))
    if not conn:
        return f"I can't see a connection between \"{a}\" and \"{b}\" yet."
    cross = " — and these span different domains" if conn["cross_domain"] else ""
    return (
        f"\"{a}\" and \"{b}\" are connected{cross}: "
        + "; ".join(conn["reasons"])
        + "."
    )


def discover_connections() -> str:
    """
    Full-corpus sweep: persists every pairwise connection plus transitive
    novel links (A-B and B-C imply A-C via B) and reports what it found.
    """
    corpus = _facts_corpus()
    if len(corpus) < 2:
        return "Need at least two stored facts to discover connections."

    features = {fact: extract_fact_features(fact) for fact in corpus}
    direct: dict[tuple[int, int], dict] = {}
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            conn = _score_pair(features[corpus[i]], features[corpus[j]])
            if conn:
                direct[(i, j)] = conn
                _persist(corpus[i], corpus[j], conn)

    # Transitive novel links: i-j not directly connected, but both connect
    # through some k.
    novel: list[tuple[str, str, str]] = []
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            if (i, j) in direct:
                continue
            for k in range(len(corpus)):
                if k == i or k == j:
                    continue
                has_ik = (i, k) in direct or (k, i) in direct
                has_kj = (k, j) in direct or (j, k) in direct
                if has_ik and has_kj:
                    novel.append((corpus[i], corpus[k], corpus[j]))
                    _persist(
                        corpus[i],
                        corpus[j],
                        {
                            "relation": "transitive",
                            "score": 0.8,
                            "reasons": [f"novel link via: {corpus[k]}"],
                            "cross_domain": int(
                                _is_cross_domain(features[corpus[i]], features[corpus[j]])
                            ),
                        },
                    )
                    break

    lines = [
        (
            f"Discovered {len(direct)} direct and {len(novel)} novel transitive "
            f"connections across {len(corpus)} stored facts."
        )
    ]
    for a, via, b in novel[:5]:
        lines.append(f"• \"{a}\" ⟷ \"{b}\" — novel link through \"{via}\"")
    return "\n".join(lines)
