# Real-Time API Schema Inference

## Motivation

When Ultron talks to an HTTP API, it builds request bodies from whatever it
knows about that API — which may be stale. APIs change: a field gets renamed,
a parameter becomes required, a type tightens, an endpoint moves. Today a
schema change means the API answers `400 Bad Request` with a message like
*"unknown field 'user', did you mean 'username'?"* — and the agent (or the
user) has to notice, figure out what changed, and remember to adjust every
future call.

The goal of this feature: **detect subtle schema changes automatically and
update usage prediction without human intervention.**

```
  observe ──► infer ──► remember ──► correct
     │           │          │           │
  2xx/4xx    parse the   persist in   apply to the
  response   validation  SQLite       next call,
             error       store        report what
                                     changed
```

## The learning loop

Every `make_http_request` call now feeds the loop:

1. **Observe** — after each request, the tool records the interaction:
   - On success (2xx): the endpoint's observed request-body keys and
     response-body keys are learned into the store.
   - On a validation-style failure (400/422): the error text is parsed for
     drift signals.
2. **Infer** — the error parser (`_infer_drift`) classifies the change into a
   drift type with a confidence score.
3. **Remember** — high-confidence drifts are persisted (deduplicated, so a
   repeated failure doesn't spam the store).
4. **Correct** — before the next call to the same endpoint, `apply_hints`
   rewrites the request body using the learned correction, and the tool
   output reports the applied fix.

## Drift taxonomy

| Drift type         | Error signal                                     | Confidence |
|--------------------|--------------------------------------------------|------------|
| `renamed_field`    | `unknown/unexpected field 'X', did you mean 'Y'` | high       |
| `renamed_field`    | `unknown/unexpected field 'X'` (no hint)         | medium     |
| `required_field`   | `'X' is required` / `missing field 'X'`          | high       |
| `type_change`      | `'X' should be an integer` / `expected int, got str` | medium  |
| `endpoint_removed` | `404` on an endpoint the store has seen working  | medium     |

**Why confidence matters:** only *high*-confidence corrections are applied
automatically to future requests. A rename the API itself suggested
(*"did you mean 'username'?"*) is a strong signal; a bare `unknown field`
without a hint is recorded and reported but never auto-applied, because
guessing a replacement could send wrong data. Required-field and type
changes are *reported* as hints the agent can act on.

## Explicit schema discovery

Beyond passive learning from failures, Ultron can fetch an API's canonical
schema on request: `learn_api_schema(base_url)` probes the conventional
OpenAPI discovery paths (`/openapi.json`, `/swagger.json`, `/v3/api-docs`,
`/api-docs`) and mines the spec for endpoint parameters and request-body
properties. A mined spec is the strongest evidence available — a field that
the spec says is required is recorded as a high-confidence hint even before
a single failed call.

## Usage prediction

`api_usage_hint(method, url, body)` is a registered tool (read-only, LOW
risk) that answers "what should I send to this endpoint?" before a call is
made:

- fields the API no longer accepts (renamed) → send the new name instead;
- fields that are required but missing from the body → add them;
- endpoints that have moved → prefer the new path.

The same logic runs *inside* `make_http_request`, so the loop closes even
when the agent never asks: a failed call today produces a corrected call
tomorrow, with the output showing exactly what was changed and why.

## Storage

SQLite at `~/.ultron` project data dir (`.ultron_api_schema.db`):

- `endpoints` — per (base_url, method, path): observed request/response
  keys, last-seen time, call count, and whether a spec confirmed it.
- `specs` — per base_url: fetched OpenAPI document (JSON).
- `drifts` — per (base_url, method, path, type, field): the correction,
  the evidence string, and confidence. Unique on the 5-tuple so repeated
  identical failures don't duplicate rows; on conflict the stored record is
  refreshed only when the new signal is stronger or different (a fresh
  "did you mean" replaces an older one, a bare rejection upgrades to a
  hinted one) so corrections can self-correct over time.

Successful calls *merge* (union) newly observed request fields into the
endpoint's stored set — a single call never clobbers spec-mined or
previously-seen fields, so the learned picture only grows richer.

`forget_api(base_url)` clears everything learned about one API.

## Security

- `learn_api_schema`, `api_usage_hint`, `get_api_knowledge`, `forget_api`
  are classified **LOW** — they are local, read-only introspection plus a
  URL-safety-gated GET of an OpenAPI document. `forget_api` only deletes
  Ultron's own learned data, never user files.
- The guardrails URL scan now covers these actions: a non-https /
  non-localhost target is denied before any network I/O (the OpenAPI fetch
  re-checks with `check_url_safety` regardless).
- Learning never breaks a request: every schema-store write is wrapped so a
  database hiccup degrades to "no learning" rather than a failed HTTP call.

## Agent wiring

- New detector `detect_api_schema_intent` runs **before** the HTTP detector
  (step 4.83), so *"learn the api schema for http://localhost:8000"* routes
  to schema learning instead of a GET. Phrases:
  - `learn / fetch / discover the api schema for <url|domain>` → learn
  - `what do you know about the api <url>` / `what apis do you know` → knowledge
  - `api usage hints for <url>` → hints
  - `forget / clear the api schema for <url>` → forget
- When a URL is missing, a one-turn clarification asks "which API?" (the
  same pattern file reads use). "What apis do you know" needs no URL and
  lists everything learned.
- The four new tools are registered and gated like every other tool; the
  ReAct agent gets them for free via the generic path.

## Edge cases handled

- Repeated failures of the same kind record one drift, not thousands.
- A drift learned for `POST /users` never leaks onto `GET /users/1` (keyed
  by method + path).
- Non-JSON bodies and HTML error pages are skipped by the parser.
- Querystrings are stripped before keying, so `?page=2` doesn't fork the
  endpoint identity.
- URL-safety is enforced both in the guardrails and inside the OpenAPI
  fetcher, so no outbound request can target a plain-http non-local host.

## Future work

- Feed drifts into the system prompt so the model *starts* from corrected
  expectations instead of discovering them on the first 400.
- TTL/aging for drifts so an API that reverts its schema forgets the stale
  correction.
- Schema diffing across spec versions (`/openapi.json` + `If-None-Match` /
  version stamps) to detect changes proactively rather than on failure.
