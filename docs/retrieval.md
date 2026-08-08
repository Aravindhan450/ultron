# Unified Retrieval

> Design doc for Ultron's retrieval orchestrator: one interface that decides
> the best way to get information from the external world, so the agent never
> has to guess between search / page-fetch / HTTP GET / connectivity check.

## Motivation

Ultron has three networking tools — `search_web` (DuckDuckGo), `fetch_page_text`
(fetch + parse a page), and `make_http_request` (raw GET/POST/… to an API) —
each behind its own regex detector. That split forces a tool-selection step on
every request:

- *"Check if this website is online and read its main headlines"* — is this a
  fetch, a GET, or two separate calls?
- *"Is example.com reachable?"* — there was no connectivity tool at all, so the
  agent would have had to fetch the page and squint at the status text.
- Choosing `fetch_page_text` when the user wanted a raw API response (or vice
  versa) is a common, wasteful error.

A **unified retrieval interface** removes the guessing: one tool takes the
request, plans the best strategy from context, and — when a request implies
several things (connection status *and* content) — composes them into a single
call. Tool selection becomes a deterministic plan, not a model guess.

## Design

### The tools

```
retrieve(request: str, url: str | None = None) -> str   # orchestrator
check_connectivity(url: str) -> str                      # HEAD/GET status check
```

`retrieve` is the single entry point. It reads *intent markers* out of the
request text and produces a plan (a small list of strategies), runs each one,
and returns a labelled combined report:

| Request contains…            | Plan                                   |
|------------------------------|----------------------------------------|
| connectivity marker*         | `[check_connectivity]`                 |
| content marker†              | `[fetch_page_text]`                    |
| connectivity + content       | `[check_connectivity, fetch_page_text]`|
| a URL but no markers         | `[fetch_page_text]`                    |
| neither (a bare query)       | `[search_web]`                         |

\* *connectivity markers:* `online`, `offline`, `reachable`, `live`,
`connectivity`, `status of`, `is it up/down/working`, `up right now`.
† *content markers:* `read`, `fetch`, `scrape`, `parse`, `headline(s)`,
`main content`, `content of`, `article`.

### `check_connectivity`

Answers "is this site up?" in one line:

```
https://example.com is online (200 OK, 342ms)
```

Implementation: an `httpx` HEAD request with a 10s timeout; if the server
rejects HEAD (transport-level failure), fall back to a GET that only reads the
status line. Reports the status code, reason phrase, and measured latency, or
an honest `unreachable` / `did not respond` message. Uses the same URL-safety
gate as every other network tool.

### The orchestrator — `retrieve`

1. Extract a URL (`http(s)://…` or a bare domain, which is normalized to
   `https://` — so *"check if example.com is online"* just works).
2. Score the request text against the connectivity and content markers.
3. Build the plan (table above).
4. Execute each strategy and return a labelled report, e.g.:

   ```
   [connectivity] https://news.example.com is online (200 OK, 210ms)

   [page] Welcome — Headline: ... (readable text)
   ```

   If a strategy needs a URL that wasn't given, it returns an explicit
   "please include a URL" error instead of silently guessing.

### Security model

The orchestrator is read-only by construction, but it still goes through the
security boundary exactly like every other tool:

- **Classification** — `retrieve` and `check_connectivity` are LOW risk
  (read-only). URL safety is enforced by the guardrails, not by the tier.
- **Guardrails** — an unsafe URL (non-`https`, non-localhost) in a `retrieve`
  or `check_connectivity` target **denies the call outright** before any
  request fires, reusing the existing URL rule. For `retrieve`, the URL check
  only runs when the target actually contains a URL — a bare search query is
  not a URL and must not be blocked as one.
- **Agent gate** — `handle_retrieve` calls `check_action("retrieve", url)`;
  deny → blocked message, allow → execute. Both simple and ReAct agents route
  `retrieve` through the generic gated path, so an unsafe URL can never reach
  the network.

### Detector — `detect_retrieval_intent`

Placed *before* the HTTP / web-search / fetch detectors in the simple agent, so
a request like *"check if example.com is online and read its headlines"* is
claimed by the orchestrator instead of being partially matched by a single
tool detector. It returns `{"request": …, "url": … | None}`:

- URL present + a connectivity or content marker → retrieve with that URL.
- No URL + connectivity marker → a *clarification turn*: the agent asks which
  website, remembers `category="retrieve"`, and on the next reply extracts the
  URL/domain and runs the orchestrator. (Same pattern as the existing
  file-read clarification flow.)
- No markers (plain search, POST/PUT API calls, bare "get") → `None`, so
  search and state-changing HTTP keep their current behaviour.

The LLM fallback paths can also call `retrieve`/`check_connectivity` directly;
the generic gated tool-execution path (simple + ReAct) applies the same
boundary checks.

## CLI flow

1. User: *"check if news.example.com is online and read its main headlines"*
2. `detect_retrieval_intent` → `{"request": …, "url": "https://news.example.com"}`
3. `handle_retrieve` → boundary verdict `allow` (LOW, safe URL)
4. `retrieve` plans `[check_connectivity, fetch_page_text]` and returns one
   combined report.

## Edge cases

- **Bare domains** — normalized to `https://` for connectivity checks. A bare
  domain only becomes a URL when it looks like a website: web-ish TLDs are
  accepted, file paths (`config.yaml`, `notes.txt`), email addresses
  (`a@b.co`) and non-http schemes (`ftp://…`) are not. Availability requests
  may still name non-web-TLD hosts (`my-site.local`) and are handled by the
  detector's connectivity path.
- **No URL given** — connectivity requests ask for the URL on the next turn
  (one-turn clarification); fetch requests without a URL return an explicit
  error rather than guessing.
- **Search stays search** — `search for X` (no URL, no markers) is untouched
  and keeps flowing to the web-search detector.
- **API calls stay API calls** — `post to http://localhost:8000` (no retrieval
  markers) is untouched and keeps its POST confirmation flow.
- **Unsafe URLs** — guardrails deny before any network I/O.

## Future work

- **N-result search + fetch**: *"search for X and read the top result"* —
  plan `[search_web, fetch_page_text]` chained on the first result's URL.
- LLM-assisted planning when markers are ambiguous (deterministic first, model
  second — same philosophy as the rest of the codebase).
- Streaming: show each strategy's output as it completes.
- Apply the same orchestration idea to local retrieval (files + memory).
