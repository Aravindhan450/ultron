# Knowledge-Graph Memory (triple store)

> Status: **implemented** — see `src/ultron/core/tools/memory/graph.py`.

This document describes the upgrade of Ultron's long-term memory from a flat
list of facts to a **knowledge graph** of `subject → predicate → object`
triples, plus a deterministic deduction engine that can chain facts together
to answer multi-hop questions.

## 1. Motivation

Today memory is a flat list of sentences:

```sql
CREATE TABLE memories (id INTEGER PRIMARY KEY, fact TEXT NOT NULL, ...);
```

This is fine for recall ("what did I tell you about X?") but it is useless for
*reasoning*. Consider the two facts:

```
Paris → capital_of → France
France → borders → Germany
```

With a flat store, asking *"what is the capital of a country that borders
Germany?"* requires the LLM to hold both facts in context and make an
inference. That is slow, unreliable on small local models, and — worst of all —
it can hallucinate connections between unrelated facts.

Storing memories as **nodes and edges** instead turns reasoning into a
mechanical, deterministic graph traversal:

```
Germany  --borders⁻¹-->  France  --capital_of-->  Paris
```

Because the traversal is plain SQL over rows the user actually taught Ultron,
the answer is always grounded in stored facts — the same zero-hallucination
guarantee the existing recall path already has, extended to multi-hop
questions.

## 2. Design goals

1. **Deterministic recall + deduction.** Every answer is built from DB rows.
   No LLM involvement on the read path, so no hallucination.
2. **Backward compatible.** The existing flat `memories` table stays; anything
   the triple extractor cannot parse is still stored as a plain fact. All
   existing callers (`add_memory`, `search_memories`, `get_all_memories`) keep
   working.
3. **One store.** The graph lives in the same SQLite file
   (`.ultron_memory.db`) as the existing facts.
4. **Security-gated writes.** Remembering still routes through the security
   boundary, so credential-like content is blocked exactly as before.
5. **Extensible.** New predicates, extraction patterns, and reasoning templates
   are additive.

## 3. Schema

Two new tables alongside the existing `memories` table:

```sql
CREATE TABLE IF NOT EXISTS entities (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL UNIQUE,   -- canonical key (normalized)
    display_name TEXT NOT NULL,          -- first-seen casing, for display
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS triples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES entities(id),
    predicate  TEXT NOT NULL,            -- canonical predicate
    object_id  INTEGER NOT NULL REFERENCES entities(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, predicate, object_id)   -- dedup
);
CREATE INDEX IF NOT EXISTS idx_triples_object    ON triples(object_id);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
```

### Normalization

- **Entities** are deduplicated by a canonical `name`: lower-cased, whitespace
  collapsed, trailing punctuation stripped. `"Paris"`, `"paris"`, and
  `"Paris."` all resolve to the same node, whose `display_name` keeps the
  original casing for user-facing output.
- **Predicates** are canonicalized: surface phrases map onto a small fixed
  vocabulary so that "is the capital of", "is the capital city of", and
  "capital of" all become `capital_of`. This is what makes deduction
  reliable — the query engine never has to guess about spelling or wording.

| Surface phrase | Canonical predicate |
|---|---|
| `X is the capital (city) of Y` | `capital_of` |
| `X is located in Y` / `X lives in Y` | `located_in` |
| `X is a Y` | `is_a` |
| `X borders Y` | `borders` |
| `X was born in Y` | `born_in` |
| `X founded Y` / `X is the founder of Y` | `founded` |
| `X works at Y` / `X works for Y` | `works_at` |
| `X created Y` | `created` |
| anything else | `lowercase_with_underscores` |

## 4. Write path: `store_memory_text`

`add_memory(fact)` now routes through `store_memory_text`:

1. Run the deterministic extractor (`extract_triples`) over the sentence.
2. If one or more triples are found, store each as an edge (deduplicated).
3. Otherwise, fall back to the existing flat `memories` table — behavior for
   unparseable facts is unchanged.

The security gate is unchanged: the original sentence is scanned by the
guardrails before anything is written, so a "remember" containing a leaked
API key is still denied.

### Extraction patterns

The extractor is a small, ordered set of regexes (order matters — more
specific patterns run first):

| Input | Output triple |
|---|---|
| `Paris is the capital of France` | `Paris -capital_of-> France` |
| `the capital of France is Paris` | `Paris -capital_of-> France` |
| `France is located in Europe` | `France -located_in-> Europe` |
| `Ultron is a CLI assistant` | `Ultron -is_a-> CLI assistant` |
| `France borders Germany` | `France -borders-> Germany` |
| `Ada was born in London` | `Ada -born_in-> London` |
| `Linus founded Linux` | `Linus -founded-> Linux` |
| `Ada works at Babbage Inc` | `Ada -works_at-> Babbage Inc` |
| `I created ultron` | `I -created-> ultron` |
| `my favorite color is blue` | *(no match → flat fact)* |
| `I really like the way this project is going` | *(no match → flat fact)* |

A bare generic `is` is deliberately **not** extracted — "this project is
going" is not a reliable edge, so only explicit relationship phrases become
triples. Everything else falls back to the flat fact store.

Extraction is intentionally conservative: a false negative (storing a plain
fact) is harmless; a false positive (a wrong edge) would poison reasoning, so
unmatched sentences simply stay flat facts.

## 5. Read path

### Direct queries

- `query_triples(subject=?, predicate=?, object=?)` — exact (normalized)
  match on any subset of the triple.
- `search_triples(keyword)` — `LIKE` across subject/object display names and
  predicates; returns rendered sentences.
- `get_all_triples()` — every edge, rendered naturally.
- `recall_about(topic)` — every edge where the topic is the subject *or* the
  object. The topic-recall handler unions this with the flat-fact search.

### Deduction: `query_chain(anchor, steps)`

A step is `(predicate, direction)`:

- `forward` — from the current entity set, follow `X -pred-> ?`
- `backward` — from the current entity set, follow `? -pred-> X`

Each step is one parameterized SQL query; the result set of one hop becomes
the input of the next. This is deterministic and bounded (no recursive CTE
needed for the 2–3 hop templates we support).

Example — "capital of a country that borders Germany":

```python
query_chain("germany", [("borders", "backward"), ("capital_of", "forward")])
#   germany -> [countries that border germany] -> [their capitals]
```

### Reasoning templates: `answer_question(question)`

A small set of question templates maps natural language onto chains. Anything
not matched returns `None`, and the agent falls back to ordinary recall or the
LLM — never a made-up answer.

| Question template | Chain |
|---|---|
| `what is the capital of X` | `[("capital_of", forward)]` from X |
| `(the) capital(s) of countr(y/ies) that border(s) X` | `[("borders", backward), ("capital_of", forward)]` from X |
| `what countr(y/ies) border X` | `[("borders", forward)] ∪ [("borders", backward)]` from X |
| `who founded X` | `[("founded", backward)]` from X |
| `where was X born` | `[("born_in", forward)]` from X |
| `who works at X` | `[("works_at", backward)]` from X |

If a template matches but the graph has no data, the agent answers honestly:
*"I can't deduce that from what I have stored yet."*

## 6. Agent wiring

- `detect_remember_intent` / `handle_remember` — unchanged; the tool behind
  `add_memory` is now the unified `store_memory_text`.
- **New** `detect_deduction_question` / `handle_deduction_question` — runs
  before the topic-recall handler; routes template questions to
  `answer_question`.
- `handle_memory_question(topic)` — now unions flat-fact matches with
  `recall_about(topic)` (graph edges as subject or object), so topic recall
  sees both stores.
- The LLM-fallback classification path gets the same treatment: a
  `memory_question` category tries the deduction templates first, then topic
  recall.
- The multi-step planner and the ReAct agent keep using `add_memory`, so any
  "remember" step automatically benefits from triple extraction.

## 7. Security

- New graph tool names are classified **LOW** (same as `add_memory` /
  `search_memories`): `add_triple`, `query_triples`, `search_triples`,
  `get_all_triples`, `query_chain`.
- Every verdict is still audited to `~/.ultron/security_audit.jsonl`.
- The guardrail scan still runs on the *original sentence* before any write,
  so credential leakage is denied regardless of which store is used.

## 8. Migration & compatibility

- No migration is needed: the graph tables are additive, and the flat
  `memories` table is untouched. Existing facts remain queryable via
  `search_memories` / `get_all_memories`.
- Backfilling old facts into triples is deliberately **not** done
  automatically — retrofitting triples to prose sentences is lossy. If
  desired, a one-off backfill script can run the extractor over existing
  facts and insert only confident matches.
- The dedup `UNIQUE` constraint means re-remembering a fact is idempotent.

### Contradictions

Triples are never silently overwritten. Re-remembering a subject with a
different object keeps **both** edges, and `answer_question` reports all
values it finds ("The capital of France is Lyon, Paris.") — Ultron shows
what it knows rather than guessing which version is correct. To correct a
wrong edge deliberately, use the CLI:

```
/memory remove Paris is the capital of France
/memory clear          # drop every edge
```

Removal is CLI-only, mirroring the flat-fact `clear_all_memories` — memory
deletion is not exposed to the LLM.

## 9. Testing strategy

- **Unit** (`tests/test_memory_graph.py`): entity normalization/dedup, triple
  storage + idempotency, every extraction pattern (including the negative
  case), flat-fact fallback, filtered queries, 2-hop `query_chain`, each
  `answer_question` template, natural phrasing, union recall, and the security
  classification of the new tools (LOW).
- **Agent** (`tests/test_memory_graph.py`): deduction handler end-to-end with a
  seeded temp DB; topic recall unions graph + facts.
- **CLI**: `/memory` shows graph stats and edges.
- All DB tests use a temporary SQLite file (monkeypatched module path), never
  the developer's real `.ultron_memory.db`.

## 10. Future work

- Vector-similarity entity resolution (spelling variants, pronouns) via the
  planned FAISS backend.
- More reasoning templates (e.g. "who is the X of Y", transitive chains,
  `WHERE`-style filters over typed nodes).
- LLM-assisted extraction at store time with a confidence threshold and human
  confirmation for low-confidence edges.
- Edge provenance (source conversation, timestamps) for traceable answers.
