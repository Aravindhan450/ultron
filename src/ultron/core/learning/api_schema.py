"""ultron.core.learning.api_schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real-time API schema inference.

When Ultron talks to an HTTP API it learns the API's shape from two
sources and uses what it learns to predict correct usage:

1. **Observed interactions** — every ``make_http_request`` call records the
   request/response field shapes of the endpoint. A validation-style 4xx
   (400/422) is parsed for drift signals ("unknown field X, did you mean Y",
   "X is required", type mismatches); a 404 on an endpoint that previously
   worked is recorded as a removal.

2. **Explicit specs** — ``learn_api_schema`` fetches an OpenAPI document
   from the conventional discovery paths and mines it for endpoint
   parameters and request-body properties.

The learned knowledge feeds usage prediction: high-confidence corrections
are applied automatically to future calls (field renames), and everything
else is surfaced as hints the agent can act on. See docs/api-schema-inference.md.

Persistence is SQLite (``.ultron_api_schema.db`` in the project data dir).
Learning is strictly best-effort: every store write is wrapped so a
database hiccup degrades to "no learning" and never breaks an HTTP call.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

import httpx

from ultron.core.tools.builtin.http_client import check_url_safety
from ultron.core.tools.paths import ALLOWED_BASE_DIR

# Path to the schema-knowledge database. Tests may repoint this at a temp
# file before calling any function; every connection re-creates the schema
# idempotently, so no explicit re-init is required.
SCHEMA_DB_PATH = ALLOWED_BASE_DIR / ".ultron_api_schema.db"

# OpenAPI discovery paths probed in order by learn_api_schema().
OPENAPI_DISCOVERY_PATHS = (
    "/openapi.json",
    "/swagger.json",
    "/v3/api-docs",
    "/api-docs",
)

# HTTP statuses that carry schema-validation signals.
_VALIDATION_STATUSES = {400, 422}

# Drift types produced by the inference engine.
RENAMED_FIELD = "renamed_field"
REQUIRED_FIELD = "required_field"
TYPE_CHANGE = "type_change"
ENDPOINT_REMOVED = "endpoint_removed"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _split_url(url: str) -> tuple[str, str]:
    """
    Splits a URL into (base_url, path).

    base_url is the scheme://netloc origin used to key learned knowledge;
    path is the resource path with the query string removed so
    ``?page=2`` never forks an endpoint's identity.
    """
    clean = (url or "").strip()
    if "://" not in clean:
        clean = "https://" + clean
    try:
        scheme, rest = clean.split("://", 1)
        netloc, _, remainder = rest.partition("/")
        path = "/" + remainder
        # Strip the query string from the path.
        path = path.split("?", 1)[0].rstrip("/") or "/"
        return f"{scheme}://{netloc}", path
    except ValueError:
        return clean, "/"


def _extract_json_keys(payload: Any) -> list[str]:
    """
    Returns the top-level field names of a JSON payload.

    For a list payload (common in REST collections) the keys of the first
    element are used, so ``[{"id": 1, "name": "x"}]`` yields
    ``["id", "name"]``.
    """
    if isinstance(payload, dict):
        return sorted(str(k) for k in payload)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return sorted(str(k) for k in payload[0])
    return []


def _try_json(text: str) -> Any | None:
    """Parses *text* as JSON, returning None when it isn't JSON."""
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Drift inference
# ---------------------------------------------------------------------------

# Rename: the API names the field it no longer accepts.
_RENAMED_PATTERNS = (
    # "unknown field 'user', did you mean 'username'?" (Go/FastAPI/etc.)
    re.compile(
        r"(?:unknown|unexpected|invalid|unrecognized)\s+"
        r"(?:field|property|parameter|attribute|key)\s*"
        r"['\"]?(?P<field>[\w.-]+)['\"]?\s*,?\s*"
        r"did\s+you\s+mean\s*['\"]?(?P<correction>[\w.-]+)['\"]?",
        re.IGNORECASE,
    ),
    # "additional properties are not allowed 'x' ... did you mean 'y'?"
    re.compile(
        r"additional\s+properties\s+(?:are\s+)?not\s+allowed\s*"
        r"['\"]?(?P<field>[\w.-]+)['\"]?\s*,?\s*"
        r"did\s+you\s+mean\s*['\"]?(?P<correction>[\w.-]+)['\"]?",
        re.IGNORECASE,
    ),
    # Bare unknown-field without a suggestion.
    re.compile(
        r"(?:unknown|unexpected|unrecognized|invalid)\s+"
        r"(?:field|property|parameter|attribute|key)\s*"
        r"['\"]?(?P<field>[\w.-]+)['\"]?",
        re.IGNORECASE,
    ),
    # jsonschema / pydantic phrasing.
    re.compile(
        r"(?:additional\s+properties\s+(?:are\s+)?not\s+allowed|"
        r"extra\s+(?:fields|inputs)\s+not\s+permitted)\s*"
        r"['\"]?(?P<field>[\w.-]+)['\"]?",
        re.IGNORECASE,
    ),
)

# Required: the API names a field the request must include.
_REQUIRED_PATTERNS = (
    re.compile(
        r"(?:field|property|parameter|attribute|key)\s+"
        r"['\"]?(?P<field>[\w.-]+)['\"]?\s+is\s+required",
        re.IGNORECASE,
    ),
    re.compile(
        r"missing\s+(?:field|property|parameter|attribute|key)\s*"
        r"['\"]?(?P<field>[\w.-]+)['\"]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:required|mandatory)\s+(?:field|property|parameter|attribute|key)\s*"
        r"['\"]?(?P<field>[\w.-]+)['\"]?",
        re.IGNORECASE,
    ),
    # Rails-style: "param is missing or the value is empty: user"
    re.compile(r"param\s+is\s+missing\s+or\s+the\s+value\s+is\s+empty\s*:\s*(?P<field>\w+)", re.IGNORECASE),
)

# Type mismatch: the API expected a different type for a named field.
_TYPE_PATTERNS = (
    re.compile(
        r"(?:field|property|parameter|attribute|key)\s*"
        r"['\"]?(?P<field>[\w.-]+)['\"]?\s+"
        r"(?:should|must|has\s+to|needs\s+to|expected\s+to)\s+be\s+"
        r"(?:an?\s+)?(?:valid\s+)?(?P<type>\w+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"expected\s+(?:an?\s+)?(?P<type>\w+)\s*(?:type)?\s*,?\s+"
        r"(?:but\s+)?(?:got|received)\s+(?:an?\s+)?\w+"
        r"(?:\s+for\s+(?:field\s+|property\s+)?['\"]?(?P<field>[\w.-]+)['\"]?)?",
        re.IGNORECASE,
    ),
)


def _infer_drift(
    method: str,
    request_keys: list[str],
    status: int,
    response_text: str,
) -> dict[str, str] | None:
    """
    Parses an error response for schema-drift signals.

    Returns a drift record dict or None when the response carries no
    recognizable signal. The record keys: drift_type, field, correction
    (optional), evidence, confidence.
    """
    if status not in _VALIDATION_STATUSES:
        return None
    text = response_text or ""

    # --- renamed field -------------------------------------------------
    for pattern in _RENAMED_PATTERNS[:2]:
        match = pattern.search(text)
        if match:
            return {
                "drift_type": RENAMED_FIELD,
                "field": match.group("field"),
                "correction": match.group("correction"),
                "evidence": match.group(0)[:120],
                "confidence": "high",
            }
    # Rename without an explicit hint: medium confidence, no correction.
    for pattern in _RENAMED_PATTERNS[2:]:
        match = pattern.search(text)
        if match:
            return {
                "drift_type": RENAMED_FIELD,
                "field": match.group("field"),
                "correction": "",
                "evidence": match.group(0)[:120],
                "confidence": "medium",
            }

    # --- required field ------------------------------------------------
    for pattern in _REQUIRED_PATTERNS:
        match = pattern.search(text)
        if match:
            return {
                "drift_type": REQUIRED_FIELD,
                "field": match.group("field"),
                "correction": "",
                "evidence": match.group(0)[:120],
                "confidence": "high",
            }

    # --- type change ---------------------------------------------------
    for pattern in _TYPE_PATTERNS:
        match = pattern.search(text)
        if match:
            return {
                "drift_type": TYPE_CHANGE,
                "field": match.groupdict().get("field") or "",
                "correction": match.groupdict().get("type") or "",
                "evidence": match.group(0)[:120],
                "confidence": "medium",
            }

    # --- generic validation body-key mismatch --------------------------
    # Last resort: the error mentions a body key next to a rejection word.
    # Only fires when the request actually carried the named field, so the
    # signal is grounded in what was sent.
    if request_keys:
        for key in request_keys:
            in_error = re.search(rf"['\"]{re.escape(key)}['\"]", text)
            rejected = re.search(
                r"\b(?:unknown|unexpected|not\s+allowed|not\s+permitted|invalid)\b",
                text,
                re.IGNORECASE,
            )
            if in_error and rejected:
                return {
                        "drift_type": RENAMED_FIELD,
                        "field": key,
                        "correction": "",
                        "evidence": text.strip()[:120],
                        "confidence": "medium",
                    }
    return None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS endpoints (
    base_url   TEXT NOT NULL,
    method     TEXT NOT NULL,
    path       TEXT NOT NULL,
    request_keys  TEXT NOT NULL DEFAULT '[]',
    response_keys TEXT NOT NULL DEFAULT '[]',
    spec       INTEGER NOT NULL DEFAULT 0,
    last_seen  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    call_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (base_url, method, path)
);
CREATE TABLE IF NOT EXISTS specs (
    base_url TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS drifts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_url TEXT NOT NULL,
    method   TEXT NOT NULL,
    path     TEXT NOT NULL,
    drift_type TEXT NOT NULL,
    field    TEXT NOT NULL,
    correction TEXT NOT NULL DEFAULT '',
    evidence TEXT NOT NULL DEFAULT '',
    confidence TEXT NOT NULL DEFAULT 'medium',
    detected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (base_url, method, path, drift_type, field)
);
"""


def _connect() -> sqlite3.Connection:
    """Opens the knowledge DB, creating the schema idempotently."""
    conn = sqlite3.connect(str(SCHEMA_DB_PATH))
    conn.executescript(_SCHEMA_SQL)
    return conn


def _get_drifts(base_url: str, method: str, path: str) -> list[dict[str, str]]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT drift_type, field, correction, confidence, evidence "
                "FROM drifts WHERE base_url=? AND method=? AND path=?",
                (base_url, method, path),
            ).fetchall()
        return [
            {
                "drift_type": r[0],
                "field": r[1],
                "correction": r[2],
                "confidence": r[3],
                "evidence": r[4],
            }
            for r in rows
        ]
    except (sqlite3.Error, OSError):
        return []


def _record_drift(drift: dict[str, str], base_url: str, method: str, path: str) -> dict[str, str] | None:
    """
    Persists a drift record; returns it only when the row was inserted or
    meaningfully updated.

    On conflict the stored record is refreshed only when the new signal is
    stronger or different (a fresh did-you-mean suggestion replaces an older
    one, a bare rejection upgrades to a hinted one). Identical re-detections
    are ignored so repeated failures don't re-announce themselves.
    """
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "INSERT INTO drifts "
                "(base_url, method, path, drift_type, field, correction, evidence, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(base_url, method, path, drift_type, field) DO UPDATE SET "
                "correction=excluded.correction, evidence=excluded.evidence, "
                "confidence=excluded.confidence, detected_at=CURRENT_TIMESTAMP "
                "WHERE excluded.correction != drifts.correction "
                "OR excluded.confidence != drifts.confidence",
                (
                    base_url,
                    method,
                    path,
                    drift["drift_type"],
                    drift["field"],
                    drift.get("correction", ""),
                    drift.get("evidence", ""),
                    drift.get("confidence", "medium"),
                ),
            )
            conn.commit()
            if cursor.rowcount == 1:
                return drift
        return None
    except (sqlite3.Error, OSError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_interaction(
    method: str,
    url: str,
    request_body: Any,
    response_status: int,
    response_body: str,
) -> dict[str, str] | None:
    """
    Learns from one HTTP interaction.

    On success (2xx) the endpoint's observed request/response field shapes
    are stored. On a validation failure (400/422) or a 404 the error text is
    parsed for drift and any new drift is persisted.

    Returns the freshly-recorded drift dict (so the caller can surface it),
    or None when nothing new was learned.
    """
    base_url, path = _split_url(url)
    method_upper = (method or "GET").upper()
    request_keys = _extract_json_keys(request_body) if isinstance(request_body, dict) else []
    parsed_response = _try_json(response_body)
    response_keys = _extract_json_keys(parsed_response)

    try:
        if 200 <= response_status < 300:
            with _connect() as conn:
                # Merge (union) request fields rather than replacing them: a
                # single observed call must not clobber spec-mined or
                # previously-seen fields. Response fields keep the latest
                # observation, which is the freshest picture of the reply.
                existing = conn.execute(
                    "SELECT request_keys FROM endpoints "
                    "WHERE base_url=? AND method=? AND path=?",
                    (base_url, method_upper, path),
                ).fetchone()
                merged_keys = request_keys
                if existing:
                    try:
                        prev = json.loads(existing[0])
                    except (json.JSONDecodeError, TypeError):
                        prev = []
                    merged_keys = sorted(set(prev) | set(request_keys))
                conn.execute(
                    "INSERT INTO endpoints (base_url, method, path, request_keys, response_keys, call_count) "
                    "VALUES (?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT(base_url, method, path) DO UPDATE SET "
                    "request_keys=?, response_keys=?, call_count=call_count+1, "
                    "last_seen=CURRENT_TIMESTAMP",
                    (
                        base_url,
                        method_upper,
                        path,
                        json.dumps(merged_keys),
                        json.dumps(response_keys),
                        json.dumps(merged_keys),
                        json.dumps(response_keys),
                    ),
                )
                conn.commit()
            return None

        drift = _infer_drift(method_upper, request_keys, response_status, response_body)

        # A 404 on an endpoint we have seen succeed is recorded as a removal.
        if not drift and response_status == 404:
            with _connect() as conn:
                seen = conn.execute(
                    "SELECT call_count FROM endpoints WHERE base_url=? AND method=? AND path=?",
                    (base_url, method_upper, path),
                ).fetchone()
            if seen and seen[0] > 0:
                drift = {
                    "drift_type": ENDPOINT_REMOVED,
                    "field": path,
                    "correction": "",
                    "evidence": "HTTP 404 after previously successful calls",
                    "confidence": "medium",
                }
        if not drift:
            return None
        return _record_drift(drift, base_url, method_upper, path)
    except (sqlite3.Error, OSError):
        return None


def apply_hints(method: str, url: str, body: dict) -> tuple[dict | None, list[str]]:
    """
    Applies learned usage prediction to a pending request body.

    Returns ``(corrected_body, notes)``:

    - ``corrected_body`` is a NEW dict with high-confidence renames applied,
      or None when nothing needed correcting (caller keeps the original).
    - ``notes`` is a list of human-readable strings describing what was
      corrected and what required fields are still missing.

    Only high-confidence renames are applied automatically; every other drift
    type is surfaced as a note the caller can relay.
    """
    base_url, path = _split_url(url)
    method_upper = (method or "GET").upper()
    corrected: dict | None = None
    notes: list[str] = []

    for drift in _get_drifts(base_url, method_upper, path):
        field = drift["field"]
        if drift["drift_type"] == RENAMED_FIELD and drift["confidence"] == "high" and drift["correction"]:
            if field in body:
                replacement = drift["correction"]
                if replacement in body:
                    # Never clobber a value the caller already sent under the
                    # correct name — surface the collision instead.
                    notes.append(
                        f"field '{field}' is rejected; the body already sends '{replacement}'"
                    )
                else:
                    if corrected is None:
                        corrected = dict(body)
                    corrected[replacement] = corrected.pop(field)
                    notes.append(
                        f"field '{field}' was renamed to '{replacement}' — sending '{replacement}'"
                    )
        elif drift["drift_type"] == REQUIRED_FIELD and drift["confidence"] == "high":
            if field not in body:
                notes.append(f"field '{field}' is required by this endpoint")
        elif drift["drift_type"] == ENDPOINT_REMOVED:
            notes.append("this endpoint returned 404 recently — it may have moved")

    return corrected, notes


def learn_api_schema(base_url: str) -> str:
    """
    Fetches an OpenAPI document from the conventional discovery paths and
    mines it for endpoint parameter/body knowledge.

    URL safety is enforced before any network I/O (localhost or https only).
    """
    clean = (base_url or "").strip().rstrip("/")
    if not clean:
        return "Error: learn_api_schema needs a base URL (e.g. http://localhost:8000)."
    if "://" not in clean:
        clean = "https://" + clean

    safety_error = check_url_safety(clean)
    if safety_error:
        return f"Error: {safety_error}"

    base, _ = _split_url(clean)
    if base.endswith("://"):
        return f"Error: '{clean}' is not a valid base URL (missing host)."

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            spec_payload = None
            fetched_from = None
            for candidate in OPENAPI_DISCOVERY_PATHS:
                probe = f"{base}{candidate}"
                response = client.get(probe)
                if response.status_code != 200:
                    continue
                payload = _try_json(response.text)
                if payload and isinstance(payload, dict) and ("paths" in payload or "swagger" in payload):
                    spec_payload = payload
                    fetched_from = probe
                    break
    except (httpx.RequestError, httpx.TimeoutException, httpx.HTTPError, OSError) as exc:
        return f"Error: could not reach {base} ({exc})."

    if spec_payload is None:
        return (
            f"No OpenAPI document found at {base} — tried "
            + ", ".join(f"{base}{p}" for p in OPENAPI_DISCOVERY_PATHS)
            + ". The API may not publish a spec; usage is still learned from live calls."
        )

    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO specs (base_url, spec_json) VALUES (?, ?) "
                "ON CONFLICT(base_url) DO UPDATE SET spec_json=?, fetched_at=CURRENT_TIMESTAMP",
                (base, json.dumps(spec_payload), json.dumps(spec_payload)),
            )
            mined = _mine_spec(conn, base, spec_payload)
            conn.commit()
    except (sqlite3.Error, OSError):
        return f"Error: could not store the schema for {base}."

    return (
        f"Learned the OpenAPI schema from {fetched_from}: {mined} endpoint"
        f"{'s' if mined != 1 else ''} indexed for {base}."
    )


def _mine_spec(conn: sqlite3.Connection, base_url: str, spec: dict) -> int:
    """
    Mines an OpenAPI document for per-endpoint parameter and request-body
    knowledge, upserting it into the endpoints table. Returns the number of
    endpoints indexed.
    """
    schemas = (spec.get("components") or {}).get("schemas") or {}

    def resolve_properties(ref_schema: dict | None) -> list[str]:
        """Resolves $ref schemas one level deep into a property-key list."""
        if not ref_schema:
            return []
        if "$ref" in ref_schema:
            ref_name = str(ref_schema["$ref"]).rsplit("/", 1)[-1]
            ref_schema = schemas.get(ref_name, {})
        return sorted(str(k) for k in (ref_schema.get("properties") or {}))

    mined = 0
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        for method in ("get", "post", "put", "delete", "patch"):
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            # Only query/path parameters are learned as request fields —
            # headers describe transport, not payload shape.
            params = [
                str(p.get("name", ""))
                for p in (op.get("parameters") or [])
                if isinstance(p, dict)
                and p.get("name")
                and p.get("in") in ("query", "path")
            ]
            body_props: list[str] = []
            request_body = op.get("requestBody") or {}
            content = (request_body.get("content") or {}).get("application/json") or {}
            if isinstance(content, dict):
                body_props = resolve_properties(content.get("schema"))
            merged = sorted(set(params) | set(body_props))
            conn.execute(
                "INSERT INTO endpoints (base_url, method, path, request_keys, response_keys, spec, call_count) "
                "VALUES (?, ?, ?, ?, ?, 1, 0) "
                "ON CONFLICT(base_url, method, path) DO UPDATE SET "
                "request_keys=?, spec=1",
                (
                    base_url,
                    method.upper(),
                    path,
                    json.dumps(merged),
                    "[]",
                    json.dumps(merged),
                ),
            )
            mined += 1
    return mined


def get_api_knowledge(base_url: str = "") -> str:
    """
    Summarizes what Ultron has learned about one API (or all APIs when no
    URL is given).
    """
    try:
        with _connect() as conn:
            if base_url:
                base, _ = _split_url(base_url)
                rows = conn.execute(
                    "SELECT method, path, request_keys, response_keys, spec, call_count "
                    "FROM endpoints WHERE base_url=? ORDER BY path, method",
                    (base,),
                ).fetchall()
                drift_rows = conn.execute(
                    "SELECT drift_type, field, correction, confidence "
                    "FROM drifts WHERE base_url=? ORDER BY detected_at DESC",
                    (base,),
                ).fetchall()
            else:
                rows = []
                drift_rows = []
                api_rows = conn.execute(
                    "SELECT DISTINCT base_url FROM endpoints UNION SELECT base_url FROM specs ORDER BY base_url"
                ).fetchall()
                if not api_rows:
                    return "No API schemas learned yet. Make an API call or run 'learn the api schema for <url>'."

        if base_url and not rows:
            return f"No knowledge stored for {base} yet."

        lines: list[str] = []
        if base_url:
            lines.append(f"API knowledge for {base}:")
            for method, path, req_keys, resp_keys, spec, count in rows:
                spec_tag = " (from spec)" if spec else ""
                req_fields = ", ".join(json.loads(req_keys)) if json.loads(req_keys) else "none"
                resp_fields = ", ".join(json.loads(resp_keys)) if json.loads(resp_keys) else "none"
                lines.append(
                    f"- {method} {path} — calls: {count}{spec_tag}; "
                    f"request fields: {req_fields}; response fields: {resp_fields}"
                )
            if drift_rows:
                lines.append("\nDetected schema changes:")
                for drift_type, field, correction, confidence in drift_rows:
                    fix = f" → {correction}" if correction else ""
                    lines.append(f"- {drift_type} '{field}'{fix} (confidence: {confidence})")
        else:
            lines.append("Known APIs:")
            for (api,) in api_rows:
                lines.append(f"- {api}")

        return "\n".join(lines)
    except (sqlite3.Error, OSError):
        return "Error: could not read the schema knowledge store."


def forget_api(base_url: str) -> str:
    """
    Clears everything Ultron has learned about one API.
    """
    base, _ = _split_url(base_url)
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM endpoints WHERE base_url=?", (base,))
            conn.execute("DELETE FROM specs WHERE base_url=?", (base,))
            conn.execute("DELETE FROM drifts WHERE base_url=?", (base,))
            conn.commit()
        return f"Forgotten all learned schema knowledge for {base}."
    except (sqlite3.Error, OSError):
        return f"Error: could not clear schema knowledge for {base}."


# ---------------------------------------------------------------------------
# Registered tools
# ---------------------------------------------------------------------------

def api_usage_hint(method: str, url: str, body: str | None = None) -> str:
    """
    Returns the learned usage prediction for a pending API call.

    Shows what field renames will be applied automatically, which required
    fields are missing, and any endpoint-removal warnings. Body may be a
    JSON string to check against the learned hints.
    """
    base_url, path = _split_url(url)
    method_upper = (method or "GET").upper()
    drifts = _get_drifts(base_url, method_upper, path)

    lines: list[str] = []
    parsed_body = _try_json(body) if body else None

    if not drifts:
        return (
            f"No usage hints yet for {method_upper} {path} — nothing has been "
            "learned about this endpoint. Try the request first, or run "
            "learn_api_schema on the API."
        )

    lines.append(f"Usage hints for {method_upper} {path}:")
    for drift in drifts:
        if drift["drift_type"] == RENAMED_FIELD:
            if drift["correction"]:
                lines.append(
                    f"- field '{drift['field']}' is rejected; send '{drift['correction']}' instead"
                    + (" (applied automatically)" if drift["confidence"] == "high" else "")
                )
            else:
                lines.append(f"- field '{drift['field']}' is rejected (new name unknown)")
        elif drift["drift_type"] == REQUIRED_FIELD:
            lines.append(f"- required field: '{drift['field']}'")
        elif drift["drift_type"] == TYPE_CHANGE:
            target = f" as {drift['correction']}" if drift["correction"] else ""
            lines.append(f"- field '{drift['field']}' has a type change{target}")
        elif drift["drift_type"] == ENDPOINT_REMOVED:
            lines.append("- this endpoint recently returned 404 — it may have moved")

    if isinstance(parsed_body, dict):
        # apply_hints' notes already describe the rename/required situation
        # concretely — no separate "would rewrite" line needed.
        _corrected, notes = apply_hints(method_upper, url, parsed_body)
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)
