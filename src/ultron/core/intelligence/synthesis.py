"""
ultron.core.intelligence.synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Synthesizes raw tool observations into polished, human-readable
Markdown answers for conversational UI delivery.
"""

from __future__ import annotations

import re

from ultron.core.intelligence.prompt_assembly import (
    build_response_guidance,
    polish_response,
)
from ultron.core.logging import get_logger

logger = get_logger(__name__)


def strip_internal_thought(text: str) -> str:
    """
    Strips internal ReAct 'Thought: ...' reasoning blocks from final conversational
    answers if the model emitted them alongside a final answer.
    """
    cleaned = (text or "").strip()
    match = re.match(
        r"^Thought:\s*.*?\n\s*(?:Final Answer:\s*|Answer:\s*)?(.*)$",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    return cleaned


async def synthesize_observation(
    user_request: str,
    observation: str,
    engine,
    tool_name: str = "tool",
) -> str:
    """
    Synthesizes raw tool observation into a clean, human-readable natural-language
    Markdown response answering the user's request.
    """
    # If the user explicitly requested raw output, respect that
    req_lower = user_request.lower()
    if any(k in req_lower for k in ("raw output", "raw results", "raw search", "unformatted", "exact payload")):
        return str(observation)

    if not observation or not str(observation).strip():
        return f"No results or data were returned by the tool for '{user_request}'."

    messages = [
        {
            "role": "system",
            "content": (
                "You are ULTRON, a helpful and precise local AI assistant.\n"
                "You have been provided with observation data gathered by an internal tool to answer a user's request.\n\n"
                "INSTRUCTIONS:\n"
                "1. Synthesize the observation into a clean, structured, and informative Markdown response.\n"
                "2. Directly answer the user's question or fulfill their request.\n"
                "3. Do NOT output raw JSON, raw dictionary blocks, or unformatted snippet dumps.\n"
                "4. Highlight key details, specifications, or takeaways using bullet points or sections.\n"
                "5. Include relevant sources (titles and URLs) at the end if present in the observation.\n"
                + build_response_guidance()
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Request: {user_request}\n\n"
                f"Tool Observation ({tool_name}):\n{observation}\n\n"
                "Synthesized Final Response:"
            ),
        },
    ]

    if engine is not None:
        try:
            raw = await engine.generate(messages)
            if raw and raw.strip():
                return polish_response(strip_internal_thought(raw))
        except Exception as exc:  # noqa: BLE001 — fallback to unformatted observation on model failure
            logger.warning("Failed to synthesize tool observation with model: %s", exc)

    # Fallback if model synthesis failed or produced nothing
    return polish_response(
        f"### Findings for **{user_request}**\n\n{observation}"
    )
