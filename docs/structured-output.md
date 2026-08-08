# Structured Output Enforcement

## Motivation

When a user asks for a report, analysis, comparison, or plan, they often want
a machine-readable shape — "answer as JSON with fields name, age", "reply in
XML with elements title, body", "as a markdown table with columns name, score".
Small local models are exactly where this fails: they wrap the answer in
prose, forget a required field, leave trailing commas, or cut the response
off mid-string. The result is a document that *looks* structured but cannot
be parsed.

This module adds a **stronger guarantee**: when a structured format is
requested — even vaguely — the reply pipeline tells the model the exact
schema *and* deterministically validates + repairs the reply before it is
shown. No LLM is involved in the repair: extraction, validation, and repair
are pure functions, so the guarantee is testable and the output is always
grounded in what the model actually wrote (nothing is fabricated).

## The pipeline

```
user_input ──► detect_schema_request() ──► SchemaSpec (or None)
                     │
        (spec found) ▼
   build_schema_instructions() ──► injected into the model prompt
                     │
        model reply (often sloppy)
                     ▼
   enforce(spec, reply)
      extract → validate → deterministic repair → serialize
                     │
        [structured] notes appended: what was fixed, what could not be
```

When **no** schema is requested, the pipeline is untouched (pure natural
language, existing polish only).

## SchemaSpec

- `format` — `json` | `xml` | `markdown`
- `fields` — ordered list of `FieldSpec(name, type, required)`; types:
  `string`, `number`, `boolean`, `list`, `object`, `any`
- `columns` — for markdown tables
- `name` — predefined-schema label when one was named

`detect_schema_request()` recognizes, all case-insensitive:

- **Format signals** — "as json", "in xml", "as a markdown table",
  "reply/return/output in json", "format: json", "json with fields …",
  "xml with elements …", "table with columns …"
- **Named schemas** — "as an analysis report", "use the comparison schema",
  "as a plan" (see `PREDEFINED_SCHEMAS`)
- **Fields without a format** — "answer with fields name, age" defaults to
  JSON (the ambiguous-input case the proposal calls out)

It never fires on unrelated phrasing: "read config.json", "what is json",
"run ls" all stay on their existing paths.

## Predefined schemas

| name        | format   | shape |
|-------------|----------|-------|
| `analysis`  | json     | summary (string), key_points (list), recommendation (string), risks (list, optional) |
| `comparison`| json     | criteria (list), items (list of {name, scores}), verdict (string) |
| `plan`      | json     | goal (string), steps (list of {action, detail}), estimated_time (string, optional) |
| `decision`  | json     | question (string), decision (string), reasoning (string), alternatives (list, optional) |
| `table`     | markdown | columns (user-supplied or kept from the reply header) |

## Deterministic repairers

### JSON
1. Extract the document: strip ```json fences, slice `{` … `}`.
2. Parse; on failure try, in order: remove trailing commas, strip `//` and
   `/* */` comments, convert single-quoted strings, unquote bare keys, then
   **recover truncation** — first the common case (unmatched open braces are
   closed: the model was cut off mid-document), then prefix recovery (retry
   prefixes cut at each `}` from the end, noting any trailing content that
   had to be dropped).
3. Conform to the spec: missing required fields are added as `null` **and
   flagged in the notes** (a presence guarantee, not a content guarantee —
   nothing is ever fabricated); type mismatches and unexpected fields are
   reported but the model's data is never rewritten.
4. Re-serialize with `indent=2`.

### XML
Token-scan with a tag stack: unclosed tags are closed at the end, mismatched
closing tags pop the stack to the matching open tag, void elements
(`<br/>`, `<img>`, …) are never pushed, and `>` inside quoted attribute
values is tokenized correctly. Illegal control characters are stripped. The
result is verified with `xml.etree`.

### Markdown tables
Locate the contiguous `|`-delimited block; insert the `|---|---|
separator if the model forgot it; pad/truncate every row to the header
column count; rebuild. Missing required columns are reported in the notes.

## The guarantee

- **Conforming output is produced whenever the content is salvageable** —
  the reply is rewritten to the schema, and `[structured]` notes list
  exactly what was fixed.
- **When repair is impossible**, the reply is returned untouched with an
  explicit `[structured] ⚠ could not be made to conform: …` note — a
  non-conforming document is never silently presented as conforming.
- **Nothing is fabricated**: the only additions are `null` placeholders for
  missing required JSON fields (flagged), never invented content.

## Wiring

- `handle_llm_fallback` (simple agent) injects the schema instructions into
  the prompt when a schema is requested, then runs `enforce_reply()` on the
  natural-text reply before the existing polish.
- The ReAct agent does the same on its final answer.
- Three tools, all **LOW** risk (pure local text processing, no I/O):
  - `enforce_schema(text, format, fields, schema)` — validate + repair
  - `schema_validate(text, format, fields, schema)` — report only, no rewrite
  - `list_schemas()` — the predefined schema catalogue

## Edge cases

- Format named but empty content → conformance failure is reported, never a
  fabricated empty document.
- A reply that is already conforming → no repair, no notes (the model's
  exact output is preserved).
- A cut-off JSON document with unmatched open braces → braces closed and the
  document recovered, with a `[structured]` note.
- Two documents in one reply (`{a: 1}{b: 2}`) → the complete first document
  is recovered and the dropped trailing content is explicitly noted.
- Repair failures surface as notes on the reply — the agent never crashes on
  malformed model output.
