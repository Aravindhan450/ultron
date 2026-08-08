import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from ultron.core.tools.builtin.http_client import check_url_safety


def search_web(query: str, max_results: int = 3) -> str:
    """
    Searches the web using DuckDuckGo and returns formatted search results.

    Parameters:
      - query: The search query string (e.g., "latest news on Python 3.12")
      - max_results: Maximum number of search results to return (default: 3)

    Returns:
      A clean formatted string listing each result's snippet followed by Title and Link lines.
    """
    try:
        # Initialize DuckDuckGo search client
        with DDGS() as ddgs:
            # Perform text search with specified result limit
            raw_results = list(ddgs.text(query, max_results=max_results))

        if not raw_results:
            return f"No search results found for '{query}'."

        formatted_output = []
        for i, res in enumerate(raw_results, start=1):
            title = res.get("title", "No Title")
            url = res.get("href", "No URL")
            snippet = res.get("body", "No snippet available.")
            block = f"{i}. {snippet}\n   **Title:** {title}\n   **Link:** {url}"
            formatted_output.append(block)

        return "\n\n".join(formatted_output)

    except (DDGSException, httpx.HTTPError, OSError, ValueError) as exc:
        return f"Error: search failed ({exc})."


def fetch_page_text(url: str) -> str:
    """
    Fetches an HTML web page, strips HTML tags/scripts, and returns readable plain text.

    Parameters:
      - url: The web page URL to fetch (e.g. "https://example.com")

    Returns:
      Extracted plain text up to 3000 characters.

    Safety Rule:
      Reuses the same URL safety check as http_client.py (only allows localhost or https URLs).
    """
    clean_url = url.strip()

    # Re-use URL safety check from http_client tool
    safety_error = check_url_safety(clean_url)
    if safety_error:
        return safety_error

    try:
        # Fetch the web page content with a 10 second timeout
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(clean_url)
            response.raise_for_status()

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags so their raw code doesn't clutter the extracted text
        for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            element.decompose()

        # Get plain text from the HTML body
        text = soup.get_text(separator=" ")

        # Clean up multiple whitespace/newlines into single spaces
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # Truncate result if longer than 3000 characters
        if len(clean_text) > 3000:
            return clean_text[:3000] + "\n...[truncated]"

        return clean_text if clean_text else "No readable text content found on page."

    except httpx.TimeoutException:
        return "Error: request timed out after 10 seconds."
    except httpx.HTTPStatusError as exc:
        return f"Error: HTTP request failed with status code {exc.response.status_code}."
    except httpx.RequestError as exc:
        return f"Error: connection or request failed ({exc})."
    except (httpx.HTTPError, UnicodeDecodeError, ValueError) as exc:
        return f"Error: failed to fetch or parse web page ({exc})."
