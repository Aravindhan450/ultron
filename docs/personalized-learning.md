# Personalized Learning: Cross-Domain Memory Connections

## Motivation

The memory system stores facts (knowledge-graph triples + flat facts) and
recalls them on demand. What it does **not** do is *relate* them to each
other. The user remembers "I love Renaissance art" and later "the Medici
ruled Florence in the 16th century" — two facts in different domains that a
curious assistant would notice are connected (Medici patronage funded the
Renaissance; Florence was its capital). Today Ultron treats them as
unrelated rows.

This module adds the personalized-learning layer: every newly stored fact is
**correlated against everything already stored**, and genuine connections —
especially *cross-domain* ones — are surfaced automatically, with a
human-readable reason for each link. No LLM is involved: extraction and
scoring are deterministic and grounded in the stored text, preserving the
zero-hallucination guarantee of the rest of the memory system.

## Fact features

For every fact, three feature sets are extracted deterministically:

| Feature  | What it is                                                        |
|----------|-------------------------------------------------------------------|
| `proper` | Capitalized multi-word runs that look like proper nouns ("Renaissance", "the Medici" → "Medici"), with sentence-start words stripped |
| `keywords` | Non-stopword content words + domain keywords                       |
| `domains` | Curated domain tags (art, politics, history, tech, science, …) matched by keyword |

The corpus is the union of both memory stores: flat facts plus every
knowledge-graph edge rendered as a sentence — so triple-stored and
flat-stored knowledge correlate uniformly.

## Cross-domain bridges

`CONCEPT_BRIDGES` is a curated map of concepts to *related concepts across
domains* — the seed that lets two facts with **no shared words** still
connect. For example:

```
renaissance → {florence, medici, italy, vatican, patronage, leonardo, …}
medici      → {florence, renaissance, banking, patronage, italy}
florence    → {italy, renaissance, medici, tuscany, art}
```

A fact about the Medici connects to a fact about Renaissance art through the
`medici ↔ renaissance` bridge even though the words never overlap. The map
is a small, opinionated seed (extensible; auto-learning bridges from
co-occurrence is future work).

## Scoring

For a pair of facts:

- shared proper noun        → +1.0 each
- shared content keyword    → +0.4 each (capped at +0.8)
- concept bridge hit        → +0.7 each (capped at +1.4)
- shared domain tag         → +0.3

A pair is a *connection* when the total is ≥ 0.7, with a relation type
(`shared_topic`, `concept_bridge`, `shared_keywords`, `shared_domain`) and
the specific reasons ("shared: renaissance", "bridge: renaissance ↔
medici"). Proper nouns are reported once — they score as topics, not again
as keywords. A connection is marked **cross-domain** when the two facts
carry *different* domain tags — the interesting, novel kind. Transitive
links are also labeled with the real cross-domain flag of their endpoints.

## What the user sees

- **On store** — `handle_remember` announces fresh links: *"🔗 Connected to
  2 existing memories …"* with the reason for each. Remembering "I love
  Renaissance art" after the Medici fact answers with the bridge.
- **`memory_connections(topic)`** — the connection map around a topic (or a
  store summary when no topic is given).
- **`related_facts(fact)`** — what one new fact relates to, on demand.
- **`explain_relation(a, b)`** — "how is X related to Y": analyzes the pair
  directly, whether or not the facts are stored.
- **`discover_connections()`** — a full-corpus sweep that also finds
  **transitive** novel links (A connects to B, B connects to C ⇒ A ↔ C via
  B) and persists everything.

## Storage & security

Connections are persisted in `.ultron_connections.db` (source, target,
relation, strength, reason, cross_domain; unique per triple) so discoveries
accumulate, while every query also re-scans live for freshness.

All four tools are read-only local operations → classified **LOW** and
auto-allowed; they never execute anything or touch the network.

## Edge cases

- Self-matches and already-stored duplicates are skipped.
- Scoring is conservative: a single shared stopword-adjacent word never
  makes a connection; only proper nouns, real content words, bridges, and
  domains count.
- A failing connections DB degrades to "no announcement" — remembering is
  never blocked by the learning layer.
-The corpus scan is bounded (newest N facts) so correlation stays cheap on
large stores; `discover_connections`'s pairwise sweep is bounded by the same
cap.

## Future work

- Auto-learn concept bridges from co-occurrence (facts repeatedly connected
  through the same bridge reinforce it).
- Per-domain affinity profiles ("the user reads a lot about art") that
  weight bridges toward their interests.
- Prompt-injected connection summaries so chat responses naturally draw on
  the learned links.
