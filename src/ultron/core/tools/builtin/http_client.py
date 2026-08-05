import json
import httpx


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
    # We check if the URL starts with secure https or a local development address.
    # Plain "http://" to an external IP/domain sends data unencrypted across the internet,
    # so we prevent the tool from executing those requests.
    is_localhost = clean_url.startswith("http://localhost") or clean_url.startswith("http://127.0.0.1")
    is_https = clean_url.startswith("https://")

    if not (is_localhost or is_https):
        return "Error: only localhost or https URLs are allowed."

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

    try:
        # Use httpx.Client with a strict 10-second timeout to prevent hanging forever
        with httpx.Client(timeout=10.0) as client:
            response = client.request(
                method=method_upper,
                url=clean_url,
                json=json_data,
                content=data_content,
            )

        # Try pretty-printing the response body if it returns JSON format
        try:
            parsed_json = response.json()
            formatted_body = json.dumps(parsed_json, indent=2)
        except Exception:
            # If the response isn't JSON (e.g. HTML or plain text), use raw text
            formatted_body = response.text

        # Truncate response body to 2000 characters if it's too long
        if len(formatted_body) > 2000:
            formatted_body = formatted_body[:2000] + "\n... [response truncated to 2000 characters]"

        # Format status code & phrase (e.g. "200 OK") along with the response body
        return f"Status: {response.status_code} {response.reason_phrase}\n\nResponse:\n{formatted_body}"

    except httpx.TimeoutException:
        return "Error: request timed out after 10 seconds."
    except httpx.RequestError as exc:
        return f"Error: connection or request failed ({exc})."
    except Exception as exc:
        return f"Error: unexpected error during HTTP request ({exc})."
