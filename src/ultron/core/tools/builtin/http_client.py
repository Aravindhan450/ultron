import json

import httpx


def check_url_safety(url: str) -> str | None:
    """
    Helper function to verify URL safety.

    Safety Rule:
    Only allows requests to localhost ("http://localhost", "http://127.0.0.1")
    or encrypted endpoints ("https://"). Plain unencrypted "http://" to non-localhost
    hosts is blocked to avoid accidentally exposing sensitive data to third parties.

    Returns None if safe, or an error string if blocked.
    """
    clean_url = url.strip()
    is_localhost = clean_url.startswith(("http://localhost", "http://127.0.0.1"))
    is_https = clean_url.startswith("https://")

    if not (is_localhost or is_https):
        return "Error: only localhost or https URLs are allowed."
    return None


def make_http_request(method: str, url: str, body: str | None = None) -> str:
    """
    Makes an HTTP request (GET, POST, PUT, DELETE) to a specified URL and
    returns the formatted response status and body.

    Parameters:
      - method: HTTP method (e.g. "GET", "POST", "PUT", "DELETE")
      - url: Target URL string
      - body: Optional string payload to send for POST/PUT requests

    Safety Rule:
      Only allows requests to localhost ("http://localhost", "http://127.0.0.1")
      or encrypted endpoints ("https://"). Plain unencrypted "http://" to non-localhost
      hosts is blocked to avoid accidentally exposing sensitive data to third parties.
    """
    # Normalize the method name to uppercase (e.g. "get" -> "GET")
    method_upper = method.strip().upper()
    valid_methods = {"GET", "POST", "PUT", "DELETE"}

    if method_upper not in valid_methods:
        return f"Error: unsupported HTTP method '{method}'. Allowed methods: {', '.join(sorted(valid_methods))}."

    clean_url = url.strip()

    # --- Safety Check ---
    safety_error = check_url_safety(clean_url)
    if safety_error:
        return safety_error

    # Prepare request body/JSON payload if provided for POST or PUT
    json_data = None
    data_content = None

    if body and method_upper in {"POST", "PUT"}:
        # Try parsing the body string as JSON first.
        # If valid, httpx will automatically serialize it and set 'Content-Type: application/json'.
        # If it's not valid JSON, send it as raw plain text instead.
        try:
            json_data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            data_content = body

    # --- Schema-aware usage prediction -----------------------------------
    # Before sending, apply learned high-confidence corrections (e.g. a field
    # the API renamed) so a schema change we already detected never needs to
    # be explained again. Learning is best-effort: any failure here simply
    # leaves the request unchanged.
    applied_notes: list[str] = []
    try:
        if isinstance(json_data, dict):
            from ultron.core.learning.api_schema import apply_hints
            corrected, hints = apply_hints(method_upper, clean_url, json_data)
            if corrected is not None:
                json_data = corrected
            applied_notes = hints
    except Exception:  # noqa: BLE001 — schema hints must never break a request
        applied_notes = []

    try:
        # Use httpx.Client with a strict 10-second timeout to prevent hanging forever
        with httpx.Client(timeout=10.0) as client:
            response = client.request(
                method=method_upper,
                url=clean_url,
                json=json_data,
                content=data_content,
            )
    except httpx.TimeoutException:
        return "Error: request timed out after 10 seconds."
    except httpx.RequestError as exc:
        return f"Error: connection or request failed ({exc})."
    except (httpx.HTTPError, UnicodeDecodeError) as exc:
        return f"Error: unexpected error during HTTP request ({exc})."

    # --- Real-time schema learning ---------------------------------------
    # Record the interaction so the store can learn endpoint shapes and
    # detect drift from validation errors. Returns the drift dict only when
    # this call newly detected a schema change.
    new_drift: dict | None = None
    try:
        from ultron.core.learning.api_schema import record_interaction
        new_drift = record_interaction(
            method_upper, clean_url, json_data, response.status_code, response.text
        )
    except Exception:  # noqa: BLE001 — learning must never break a request
        new_drift = None

    # Try pretty-printing the response body if it returns JSON format
    try:
        parsed_json = response.json()
        formatted_body = json.dumps(parsed_json, indent=2)
    except ValueError:
        # If the response isn't JSON (e.g. HTML or plain text), use raw text
        formatted_body = response.text

    # Truncate response body to 2000 characters if it's too long
    if len(formatted_body) > 2000:
        formatted_body = formatted_body[:2000] + "\n... [response truncated to 2000 characters]"

    output = f"Status: {response.status_code} {response.reason_phrase}\n\nResponse:\n{formatted_body}"

    # --- Schema-change notes ---------------------------------------------
    if new_drift:
        field = new_drift.get("field", "")
        correction = new_drift.get("correction", "")
        drift_type = new_drift.get("drift_type", "schema change")
        if correction:
            hint = f"send '{correction}' instead of '{field}'"
        elif field:
            hint = f"'{field}' is affected"
        else:
            hint = "usage prediction updated"
        output += (
            f"\n\n[api schema] Detected a {drift_type}: {hint}. "
            "This is now remembered — future calls to this endpoint will adapt automatically."
        )
    for note in applied_notes:
        output += f"\n[api schema] {note}"

    return output
