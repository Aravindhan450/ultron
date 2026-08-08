"""
Tests for the width-adaptive chat session UI (``ultron.ui.session``).

The toolbar / rprompt / prompt builders are pure functions of the terminal
width, so they are exercised directly at several widths to prove the
interface re-flows — without ever overflowing — when the window is resized.
"""

from ultron.core.state import CLIState
from ultron.ui.session import build_prompt_html, build_rprompt, build_toolbar

try:
    from wcwidth import wcswidth
except ImportError:  # pragma: no cover - wcwidth ships with prompt_toolkit

    def wcswidth(text):  # type: ignore[no-redef]
        return len(text)


def _state(**overrides):
    defaults = {
        "active_model": "llama3.2",
        "current_dir": "~/projects/ultron",
        "version": "v0.1.0",
        "status": "Ready",
    }
    defaults.update(overrides)
    return CLIState(**defaults)


def _text(fragments):
    return "".join(text for _, text in fragments)


def _display_width(text):
    width = wcswidth(text)
    return width if width >= 0 else len(text)


def _frag_width(fragments):
    return sum(_display_width(text) for _, text in fragments)


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------


def test_toolbar_wide_contains_all_segments():
    frags = build_toolbar(_state(), "react", "interactive", width=140)
    text = _text(frags)
    assert "Ultron" in text
    assert "llama3.2" in text
    assert "react" in text
    assert "~/projects/ultron" in text
    assert "interactive" in text
    assert "Ready" in text
    assert "Esc cancel" in text


def test_toolbar_narrow_drops_decorative_segments():
    frags = build_toolbar(_state(), "simple", "strict", width=40)
    text = _text(frags)
    assert "Ultron" not in text
    assert "~/projects/ultron" not in text
    assert "strict" not in text
    assert "Esc cancel" not in text
    # the essentials survive
    assert "llama3.2" in text
    assert "simple" in text


def test_toolbar_medium_keeps_hints_but_not_cwd():
    frags = build_toolbar(_state(), "react", "interactive", width=80)
    text = _text(frags)
    assert "Esc cancel" in text
    assert "~/projects/ultron" not in text  # cwd needs >= 100 cols
    assert "interactive" not in text  # security chip needs >= 88 cols


def test_toolbar_never_overflows_any_width():
    for width in (30, 50, 70, 90, 120, 200):
        frags = build_toolbar(_state(), "react", "interactive", width=width)
        assert _frag_width(frags) <= width, f"toolbar overflow at width {width}"


def test_toolbar_never_overflows_at_exact_hint_boundary():
    # For this state, content is 78 columns and the hints string is 34, so the
    # hints fit exactly at width 112. A pad of exactly 0 must drop the hints
    # instead of pushing the row one cell past the width (regression guard).
    for width in (112, 113):
        frags = build_toolbar(_state(), "react", "interactive", width=width)
        assert _frag_width(frags) <= width, f"toolbar overflow at width {width}"


def test_toolbar_reflects_thinking_status():
    frags = build_toolbar(_state(status="Thinking..."), "react", "", width=120)
    assert "Thinking" in _text(frags)


# ---------------------------------------------------------------------------
# rprompt
# ---------------------------------------------------------------------------


def test_rprompt_hidden_on_narrow():
    assert build_rprompt("strict", width=40) == []


def test_rprompt_shows_mode_when_wide():
    text = _text(build_rprompt("strict", width=80))
    assert "strict" in text


# ---------------------------------------------------------------------------
# Prompt markup
# ---------------------------------------------------------------------------


def test_prompt_wide_contains_model_and_agent():
    prompt = build_prompt_html("llama3.2", "react", width=120)
    assert "llama3.2" in prompt
    assert "react" in prompt


def test_prompt_narrow_is_bare_arrow():
    prompt = build_prompt_html("llama3.2", "react", width=40)
    assert "llama3.2" not in prompt
    assert "❯" in prompt


# ---------------------------------------------------------------------------
# Theme adaptivity helpers
# ---------------------------------------------------------------------------


def test_banner_mode_thresholds():
    from ultron.ui.theme import _banner_mode

    assert _banner_mode(140) == "side"
    assert _banner_mode(80) == "stack"
    assert _banner_mode(40) == "text"


def test_panel_padding_thresholds():
    from ultron.ui.theme import _panel_padding

    assert _panel_padding(120) == (0, 2)
    assert _panel_padding(40) == (0, 0)
