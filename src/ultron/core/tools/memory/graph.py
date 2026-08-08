"""
ultron.core.tools.memory.graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Knowledge-graph long-term memory: facts are stored as
``subject → predicate → object`` triples so Ultron can answer multi-hop
reasoning questions deterministically, without involving the LLM.

Design (see docs/memory-graph.md):

- Entities are deduplicated by a normalized canonical ``name`` (lower-cased,
  whitespace collapsed, trailing punctuation stripped) while ``display_name``
  keeps the first-seen casing for output.
- Predicates are canonicalized from surface phrases ("is the capital of" →
  ``capital_of``) so deduction never depends on wording.
- ``store_memory_text`` is the unified write path: extract triples when the
  sentence parses, otherwise fall back to the existing flat ``memories``
  table (backward compatible).
- ``query_chain`` walks edges hop-by-hop in SQL; ``answer_question`` maps a
  small set of question templates onto chains.

Everything is deterministic and grounded in stored rows — the same
zero-hallucination guarantee the flat recall path already provides.
"""

import re
import sqlite3

from ultron.core.tools.memory.sqlite import add_memory as _add_flat_fact
from ultron.core.tools.paths import ALLOWED_BASE_DIR

# Same database file as the flat fact store — one memory store, two tables.
MEMORY_DB_PATH = ALLOWED_BASE_DIR / ".ultron_memory.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS triples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    predicate  TEXT NOT NULL,
    object_id  INTEGER NOT NULL REFERENCES entities(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, predicate, object_id)
);
CREATE INDEX IF NOT EXISTS idx_triples_object    ON triples(object_id);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
"""


def init_graph_db() -> None:
    """Creates the graph tables if they do not exist yet."""
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            conn.executescript(_INIT_SQL)
    except (sqlite3.Error, OSError) as exc:
        print(f"Warning: Failed to initialize memory graph DB: {exc}")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Surface phrases -> canonical predicates. Order matters: more specific
# phrases must be matched before the generic ones below.
_PREDICATE_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"is the capital city of", re.IGNORECASE), "capital_of"),
    (re.compile(r"is the capital of", re.IGNORECASE), "capital_of"),
    (re.compile(r"is the capital", re.IGNORECASE), "capital_of"),
    (re.compile(r"is located in", re.IGNORECASE), "located_in"),
    (re.compile(r"lives in", re.IGNORECASE), "located_in"),
    (re.compile(r"was born in", re.IGNORECASE), "born_in"),
    (re.compile(r"is the founder of", re.IGNORECASE), "founded"),
    (re.compile(r"founded", re.IGNORECASE), "founded"),
    (re.compile(r"works at", re.IGNORECASE), "works_at"),
    (re.compile(r"works for", re.IGNORECASE), "works_at"),
    (re.compile(r"created", re.IGNORECASE), "created"),
    (re.compile(r"borders", re.IGNORECASE), "borders"),
    (re.compile(r"is an?", re.IGNORECASE), "is_a"),
    (re.compile(r"\bis\b", re.IGNORECASE), "is"),
]


def normalize_entity(name: str) -> str:
    """Canonical key for an entity: lower-cased, whitespace collapsed, trailing punctuation stripped."""
    text = re.sub(r"\s+", " ", name.strip()).lower()
    return re.sub(r"[\s.,;:!?'\"()]+$", "", text).strip()


def normalize_predicate(predicate: str) -> str:
    """Canonical predicate key; maps surface phrases onto the fixed vocabulary."""
    text = re.sub(r"\s+", " ", predicate.strip()).lower()
    for pattern, canonical in _PREDICATE_MAP:
        if pattern.search(text):
            return canonical
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_") or "is"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def get_or_create_entity(name: str) -> tuple[int, str]:
    """Returns (entity_id, display_name), creating the node if unknown."""
    normalized = normalize_entity(name)
    display = name.strip()
    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, display_name FROM entities WHERE name = ?", (normalized,))
        row = cur.fetchone()
        if row:
            return row[0], row[1]
        cur.execute(
            "INSERT INTO entities (name, display_name) VALUES (?, ?)",
            (normalized, display),
        )
        return cur.lastrowid, display


def add_triple(subject: str, predicate: str, object: str) -> str:
    """
    Stores one edge, deduplicated. Returns a human-readable confirmation.

    Re-remembering the same triple is a no-op (idempotent).
    """
    try:
        subject_id, subject_name = get_or_create_entity(subject)
        object_id, object_name = get_or_create_entity(object)
        canonical_predicate = normalize_predicate(predicate)
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO triples (subject_id, predicate, object_id) "
                "VALUES (?, ?, ?)",
                (subject_id, canonical_predicate, object_id),
            )
            conn.commit()
            inserted = cur.rowcount
        edge = f"{subject_name} -{canonical_predicate}-> {object_name}"
        return f"Stored: {edge}" if inserted else f"Already known: {edge}"
    except (sqlite3.Error, OSError) as exc:
        return f"Error storing triple: {exc}"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_CAPITAL_OF_S = re.compile(r"(.+?)\s+is\s+the\s+capital\s+(?:city\s+)?of\s+(.+)", re.IGNORECASE)
_CAPITAL_OF_O = re.compile(r"the\s+capital\s+(?:city\s+)?of\s+(.+?)\s+is\s+(.+)", re.IGNORECASE)
_LOCATED_IN = re.compile(r"(.+?)\s+is\s+located\s+in\s+(.+)", re.IGNORECASE)
_BORN_IN = re.compile(r"(.+?)\s+was\s+born\s+in\s+(.+)", re.IGNORECASE)
_FOUNDED = re.compile(r"(.+?)\s+(?:is\s+the\s+founder\s+of|founded)\s+(.+)", re.IGNORECASE)
_WORKS_AT = re.compile(r"(.+?)\s+works\s+(?:at|for)\s+(.+)", re.IGNORECASE)
_CREATED = re.compile(r"(.+?)\s+created\s+(.+)", re.IGNORECASE)
_BORDERS = re.compile(r"(.+?)\s+borders\s+(.+)", re.IGNORECASE)
_IS_A = re.compile(r"(.+?)\s+is\s+an?\s+(.+)", re.IGNORECASE)


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """
    Extracts at most ONE (subject, predicate, object) triple from a
    natural-language sentence, using ordered, conservative regex patterns.

    Patterns are checked most-specific-first ("is the capital of" before the
    generic "is") so a sentence never yields overlapping or contradictory
    edges. Returns an empty list when nothing parses — the caller then stores
    the sentence as a plain flat fact. False negatives are harmless; false
    positives would poison reasoning, so unmatched text is never guessed at.
    """
    extractors = [
        (_CAPITAL_OF_S, "capital_of", 1, 2),
        (_CAPITAL_OF_O, "capital_of", 2, 1),
        (_LOCATED_IN, "located_in", 1, 2),
        (_BORN_IN, "born_in", 1, 2),
        (_FOUNDED, "founded", 1, 2),
        (_WORKS_AT, "works_at", 1, 2),
        (_CREATED, "created", 1, 2),
        (_BORDERS, "borders", 1, 2),
        (_IS_A, "is_a", 1, 2),
    ]
    for pattern, predicate, subject_group, object_group in extractors:
        match = pattern.search(text)
        if match:
            return [
                (
                    match.group(subject_group).strip(),
                    predicate,
                    match.group(object_group).strip(),
                )
            ]
    return []


def store_memory_text(text: str) -> str:
    """
    Unified memory write: extract and store triples when the sentence parses,
    otherwise fall back to the flat fact store.

    This is what ``add_memory`` now points at, so every "remember" path
    (detector, LLM fallback, planner, ReAct agent) benefits from the graph.
    """
    triples = extract_triples(text)
    if triples:
        lines = [add_triple(subject, predicate, object_) for subject, predicate, object_ in triples]
        return "Stored as knowledge graph:\n" + "\n".join(lines)
    return _add_flat_fact(text)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def query_triples(
    subject: str | None = None,
    predicate: str | None = None,
    object: str | None = None,
) -> list[tuple[str, str, str]]:
    """
    Returns (subject, predicate, object) triples matching any subset of the
    given filters (exact normalized match). No filters returns all triples.
    """
    conditions: list[str] = []
    params: list[str] = []
    if subject:
        conditions.append("s.name = ?")
        params.append(normalize_entity(subject))
    if predicate:
        conditions.append("t.predicate = ?")
        params.append(normalize_predicate(predicate))
    if object:
        conditions.append("o.name = ?")
        params.append(normalize_entity(object))
    where = " WHERE " + " AND ".join(conditions) if conditions else ""

    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT s.display_name, t.predicate, o.display_name "
                "FROM triples t "
                "JOIN entities s ON s.id = t.subject_id "
                "JOIN entities o ON o.id = t.object_id"
                + where
                + " ORDER BY t.created_at ASC",
                params,
            )
            return [(row[0], row[1], row[2]) for row in cur.fetchall()]
    except (sqlite3.Error, OSError):
        return []


def render_triple(subject: str, predicate: str, object: str) -> str:
    """Renders a triple as a natural sentence for user-facing output."""
    templates = {
        "capital_of": "{subject} is the capital of {object}",
        "located_in": "{subject} is located in {object}",
        "is_a": "{subject} is a {object}",
        "borders": "{subject} borders {object}",
        "born_in": "{subject} was born in {object}",
        "founded": "{subject} founded {object}",
        "works_at": "{subject} works at {object}",
        "created": "{subject} created {object}",
        "is": "{subject} is {object}",
    }
    template = templates.get(predicate)
    if template:
        return template.format(subject=subject, object=object)
    # Unknown predicates render verbatim (underscores kept) so the edge stays
    # unambiguous.
    return f"{subject} {predicate} {object}"


def get_all_triples() -> list[str]:
    """Every stored edge, rendered as a natural sentence (oldest first)."""
    return [render_triple(s, p, o) for s, p, o in query_triples()]


def search_triples(keyword: str) -> list[str]:
    """Edges whose subject, predicate, or object contains the keyword."""
    like = f"%{keyword}%"
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT s.display_name, t.predicate, o.display_name "
                "FROM triples t "
                "JOIN entities s ON s.id = t.subject_id "
                "JOIN entities o ON o.id = t.object_id "
                "WHERE s.display_name LIKE ? OR t.predicate LIKE ? OR o.display_name LIKE ? "
                "ORDER BY t.created_at ASC",
                (like, like, like),
            )
            return [render_triple(row[0], row[1], row[2]) for row in cur.fetchall()]
    except (sqlite3.Error, OSError):
        return []


def recall_about(topic: str) -> list[str]:
    """
    Every stored edge involving the topic as subject OR object, rendered
    naturally and sorted. This is the graph half of topic recall.
    """
    edges = query_triples(subject=topic) + query_triples(object=topic)
    return sorted({render_triple(s, p, o) for s, p, o in edges})


def remove_triple(subject: str, predicate: str, object: str) -> str:
    """
    Removes one edge (exact normalized match). Exposed via the ``/memory``
    slash command so a wrong or outdated fact can be corrected.
    """
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM triples "
                "WHERE subject_id = (SELECT id FROM entities WHERE name = ?) "
                "AND predicate = ? "
                "AND object_id = (SELECT id FROM entities WHERE name = ?)",
                (normalize_entity(subject), normalize_predicate(predicate), normalize_entity(object)),
            )
            conn.commit()
            deleted = cur.rowcount
        return "Removed triple." if deleted else "No matching triple found."
    except (sqlite3.Error, OSError) as exc:
        return f"Error removing triple: {exc}"


def clear_all_triples() -> str:
    """Deletes every edge (entity nodes are kept). CLI-only, like the flat-fact clear."""
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM triples")
            conn.commit()
        return "All triples cleared."
    except (sqlite3.Error, OSError) as exc:
        return f"Error clearing triples: {exc}"


def get_graph_stats() -> dict[str, int]:
    """Counts for the /memory status view (entities, edges, flat facts)."""
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM entities")
            entities = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM triples")
            triples = cur.fetchone()[0]
            try:
                cur.execute("SELECT COUNT(*) FROM memories")
                facts = cur.fetchone()[0]
            except sqlite3.OperationalError:
                facts = 0
            return {"entities": entities, "triples": triples, "facts": facts}
    except (sqlite3.Error, OSError):
        return {"entities": 0, "triples": 0, "facts": 0}


# ---------------------------------------------------------------------------
# Deduction
# ---------------------------------------------------------------------------


def query_chain(anchor: str, steps: list[tuple[str, str]]) -> list[str]:
    """
    Walks the graph from *anchor* through an ordered list of hops.

    Each step is ``(predicate, direction)`` where direction is:

    - ``"forward"``  — from the current entities, follow ``X -pred-> ?``
    - ``"backward"`` — from the current entities, follow ``? -pred-> X``

    Returns the final reachable entity display names, or [] when any hop
    produces no results. Deterministic and bounded — one parameterized query
    per hop.
    """
    if not anchor.strip():
        return []
    # `current` always holds *normalized* entity names so each hop's SQL can
    # compare against the canonical `name` column directly.
    current = {normalize_entity(anchor)}
    for predicate, direction in steps:
        canonical = normalize_predicate(predicate)
        if direction == "forward":
            sql = (
                "SELECT o.name FROM triples t "
                "JOIN entities s ON s.id = t.subject_id "
                "JOIN entities o ON o.id = t.object_id "
                f"WHERE s.name IN ({','.join('?' for _ in current)}) AND t.predicate = ?"
            )
        else:
            sql = (
                "SELECT s.name FROM triples t "
                "JOIN entities s ON s.id = t.subject_id "
                "JOIN entities o ON o.id = t.object_id "
                f"WHERE o.name IN ({','.join('?' for _ in current)}) AND t.predicate = ?"
            )
        try:
            with sqlite3.connect(MEMORY_DB_PATH) as conn:
                cur = conn.cursor()
                cur.execute(sql, (*current, canonical))
                current = {row[0] for row in cur.fetchall()}
        except (sqlite3.Error, OSError):
            return []
        if not current:
            return []
    # Resolve the final normalized names back to display names for output.
    placeholders = ",".join("?" for _ in current)
    try:
        with sqlite3.connect(MEMORY_DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT display_name FROM entities WHERE name IN ({placeholders})",
                tuple(current),
            )
            return sorted(row[0] for row in cur.fetchall())
    except (sqlite3.Error, OSError):
        return []


# Question templates -> chains. Longer / more specific templates run first so
# "the capital of a country that borders X" is not swallowed by "capital of X".
_CAPITALS_OF_BORDERING = re.compile(
    r"capitals?\s+of\s+(?:a\s+|an\s+|the\s+)?countr(?:y|ies)\s+that\s+borders?\s+(.+)",
    re.IGNORECASE,
)
_CAPITAL_OF = re.compile(r"capital\s+of\s+(.+)", re.IGNORECASE)
_COUNTRIES_BORDERING = re.compile(
    r"countr(?:y|ies)(?:\s+that)?\s+borders?\s+(.+)", re.IGNORECASE
)
_WHO_FOUNDED = re.compile(r"who\s+founded\s+(.+)", re.IGNORECASE)
_WHERE_BORN = re.compile(r"where\s+was\s+(.+?)\s+born", re.IGNORECASE)
_WHO_WORKS_AT = re.compile(r"who\s+works\s+at\s+(.+)", re.IGNORECASE)

_REASONING_TEMPLATES = (
    _CAPITALS_OF_BORDERING,
    _CAPITAL_OF,
    _COUNTRIES_BORDERING,
    _WHO_FOUNDED,
    _WHERE_BORN,
    _WHO_WORKS_AT,
)


def is_deduction_question(question: str) -> bool:
    """True when the question matches one of the deterministic reasoning templates."""
    text = question.strip()
    return any(pattern.search(text) for pattern in _REASONING_TEMPLATES)


def _clean_entity(value: str) -> str:
    """Strips trailing question marks / whitespace from a template capture."""
    return value.strip().rstrip("?").strip()


def answer_question(question: str) -> str | None:
    """
    Answers a supported reasoning question from the graph, or None when the
    question does not match a template or the graph has no data for it.
    """
    text = question.strip()

    m = _CAPITALS_OF_BORDERING.search(text)
    if m:
        anchor = _clean_entity(m.group(1))
        # The capital of a country is stored as ``city -capital_of-> country``,
        # so both hops walk backward: ? -borders-> anchor, then ? -capital_of-> ?
        capitals = query_chain(anchor, [("borders", "backward"), ("capital_of", "backward")])
        if capitals:
            return f"The capital of a country that borders {anchor} is {', '.join(capitals)}."
        return None

    m = _CAPITAL_OF.search(text)
    if m:
        anchor = _clean_entity(m.group(1))
        # ``capital_of`` edges point from the city to the country, so walk
        # backward from the country to find its capital.
        capitals = query_chain(anchor, [("capital_of", "backward")])
        if capitals:
            return f"The capital of {anchor} is {', '.join(capitals)}."
        return None

    m = _COUNTRIES_BORDERING.search(text)
    if m:
        anchor = _clean_entity(m.group(1))
        both_directions = query_chain(anchor, [("borders", "forward")]) + query_chain(
            anchor, [("borders", "backward")]
        )
        countries = sorted(set(both_directions))
        if countries:
            return f"Countries connected to {anchor} by borders: {', '.join(countries)}."
        return None

    m = _WHO_FOUNDED.search(text)
    if m:
        anchor = _clean_entity(m.group(1))
        founders = query_chain(anchor, [("founded", "backward")])
        if founders:
            return f"{', '.join(founders)} founded {anchor}."
        return None

    m = _WHERE_BORN.search(text)
    if m:
        anchor = _clean_entity(m.group(1))
        places = query_chain(anchor, [("born_in", "forward")])
        if places:
            return f"{anchor} was born in {', '.join(places)}."
        return None

    m = _WHO_WORKS_AT.search(text)
    if m:
        anchor = _clean_entity(m.group(1))
        people = query_chain(anchor, [("works_at", "backward")])
        if people:
            return f"People who work at {anchor}: {', '.join(people)}."
        return None

    return None


# Automatically initialize the graph tables when the module is first imported.
init_graph_db()
