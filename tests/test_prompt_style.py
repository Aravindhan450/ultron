"""
Tests for the shared response-style guidance and light response polish.

build_response_guidance is appended to every system prompt so replies are
well-mannered and structured; polish_response tidies the model's raw output
without rewriting its words.
"""

from ultron.core.intelligence.prompt_assembly import (
    build_response_guidance,
    polish_response,
)


def test_guidance_covers_manners_and_structure():
    g = build_response_guidance()
    assert "polite" in g
    assert "RESPONSE STYLE" in g
    assert "Markdown" in g
    assert "concise" in g
    assert "Never invent" in g
    assert g.endswith("\n")


def test_guidance_stable_and_wired_into_system_prompts():
    # The guidance must appear in every system-prompt construction site.
    from ultron.core.agents.react import build_system_prompt
    from ultron.main import _BASE_SYSTEM_PROMPT

    assert "RESPONSE STYLE" in _BASE_SYSTEM_PROMPT
    assert "RESPONSE STYLE" in build_system_prompt()


def test_polish_strips_surrounding_whitespace():
    assert polish_response("  hello  \n") == "hello"


def test_polish_collapses_blank_line_runs():
    raw = "line one\n\n\n\nline two"
    assert polish_response(raw) == "line one\n\nline two"


def test_polish_preserves_content():
    assert polish_response("Hello there!") == "Hello there!"


def test_polish_empty_fallback():
    result = polish_response("   \n  ")
    assert "couldn't generate" in result
    assert result.strip()


def test_polish_none_fallback():
    assert "couldn't generate" in polish_response(None)
