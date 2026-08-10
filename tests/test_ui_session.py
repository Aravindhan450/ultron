"""
Tests for the width-adaptive chat session UI (``ultron.ui.session``).

The toolbar / prompt builders are pure functions of the terminal width, so
they are exercised directly at several widths to prove the interface
re-flows — without ever overflowing — when the window is resized.
"""

from ultron.core.state import CLIState
from ultron.ui.session import build_prompt_html, build_toolbar

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
    assert "llama3.2" in text
    assert "react" in text
    assert "~/projects/ultron" in text
    assert "interactive" in text
    assert "Ready" in text
    assert "Esc cancel" in text


def test_toolbar_narrow_drops_decorative_segments():
    frags = build_toolbar(_state(), "simple", "strict", width=40)
    text = _text(frags)
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
    # Find the exact width at which the right-aligned hints first fit, then
    # prove the row never overflows at, one below, and one above it (a pad
    # of exactly 0 must drop the hints instead of pushing the row one cell
    # past the width — regression guard).
    state = _state()
    boundary = None
    for width in range(60, 200):
        if "Esc cancel" in _text(build_toolbar(state, "react", "interactive", width=width)):
            boundary = width
            break
    assert boundary is not None
    for width in (boundary - 1, boundary, boundary + 1):
        frags = build_toolbar(state, "react", "interactive", width=width)
        assert _frag_width(frags) <= width, f"toolbar overflow at width {width}"


def test_toolbar_reflects_thinking_status():
    frags = build_toolbar(_state(status="Thinking..."), "react", "", width=120)
    assert "Thinking" in _text(frags)


def test_toolbar_status_dot_tracks_status():
    frags = build_toolbar(_state(status="Thinking..."), "react", "", width=120)
    assert "●" in _text(frags)


# ---------------------------------------------------------------------------
# Prompt markup
# ---------------------------------------------------------------------------


def test_prompt_is_bare_marker():
    # Claude Code style: the prompt line is only the ❯ marker — model and
    # agent context live in the bottom toolbar, never the input line.
    prompt = build_prompt_html("llama3.2", "react", width=120)
    assert "❯" in prompt
    assert "llama3.2" not in prompt
    assert "react" not in prompt


def test_prompt_narrow_is_bare_arrow():
    prompt = build_prompt_html("llama3.2", "react", width=40)
    assert "llama3.2" not in prompt
    assert "❯" in prompt


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------


def test_render_help_table_lists_commands(monkeypatch):
    # Borderless list: every command row present, no panel boxes anywhere.
    import io

    from rich.console import Console

    from ultron.ui import theme

    buf = io.StringIO()
    fake = Console(file=buf, width=120, force_terminal=False)
    monkeypatch.setattr(theme, "console", fake)
    theme.UI.render_help_table()
    out = buf.getvalue()
    assert "Available Commands" in out
    assert "/help" in out
    assert "/memory" in out
    assert "/exit, /quit" in out
    assert "╭" not in out and "╰" not in out  # no boxed panels


def test_help_table_spans_exact_console_width(monkeypatch):
    # Every table row must render at exactly the console width (expand=True),
    # uniform with all other panels — and re-flows on resize.
    import io

    from rich.console import Console

    from ultron.ui import theme

    for width in (60, 100, 140):
        buf = io.StringIO()
        # force_terminal=False keeps ANSI escapes out of the capture; the
        # expand=True padding is a table-renderer behavior independent of
        # terminal detection.
        fake = Console(file=buf, width=width, force_terminal=False)
        monkeypatch.setattr(theme, "console", fake)
        theme.UI.render_help_table()
        # Table rows (and their wrapped continuation lines) start with the
        # two-space indent; the "Available Commands" header line is exempt.
        rows = [ln for ln in buf.getvalue().splitlines() if ln.startswith("  ")]
        assert rows, f"no rows rendered at width {width}"
        assert all(len(ln) == width for ln in rows), (
            f"table rows not exactly console width at {width}: "
            f"{sorted({len(ln) for ln in rows})}"
        )


def test_banner_contains_logo(monkeypatch):
    # The ASCII ULTRON wordmark must be back in the startup banner.
    import io

    from rich.console import Console

    from ultron.ui import theme

    buf = io.StringIO()
    fake = Console(file=buf, width=130, force_terminal=False)
    monkeypatch.setattr(theme, "console", fake)
    theme.UI.render_banner(model="llama3.2", cwd_short="~/ultron")
    out = buf.getvalue()
    assert "██╗" in out  # block-drawing logo characters
    assert "Ultron AI" in out


def test_banner_narrow_falls_back_to_text(monkeypatch):
    # Below the stack threshold the banner must not emit ASCII art (it would
    # overflow a very narrow viewport) — a compact wordmark instead.
    import io

    from rich.console import Console

    from ultron.ui import theme

    buf = io.StringIO()
    fake = Console(file=buf, width=40, force_terminal=False)
    monkeypatch.setattr(theme, "console", fake)
    theme.UI.render_banner(model="llama3.2", cwd_short="~/ultron")
    out = buf.getvalue()
    assert "██╗" not in out
    assert "Ultron AI" in out


def test_shared_console_highlight_disabled():
    # rich's auto-highlight paints plain strings (numbers/paths inside model
    # names, versions, cwd) in unrelated colours — the shared console must
    # have it off so the muted palette stays authoritative.
    from ultron.ui.theme import console

    # rich stores the constructor flag as ``_highlight`` (``highlight`` is a
    # per-call ConsoleOptions override, not an instance attribute).
    assert console._highlight is False


def test_response_is_boxed_with_orange_border(monkeypatch):
    # Assistant replies render inside a quiet panel whose border is the brand
    # orange #FB6C00 — restored per the molten-flame palette.
    import io

    from rich.console import Console

    from ultron.ui import theme

    buf = io.StringIO()
    fake = Console(file=buf, width=100, force_terminal=False)
    monkeypatch.setattr(theme, "console", fake)
    theme.UI.render_response("Hello **world**.")
    out = buf.getvalue()
    assert "╭" in out and "╰" in out  # a boxed panel wraps the reply
    assert "Hello" in out and "world" in out  # markdown content survives


def test_response_border_is_brand_orange_truecolor(monkeypatch):
    # With terminal emulation on, the response panel border must carry the
    # exact truecolor of #FB6C00 (R=251 G=108 B=0).
    import io

    from rich.console import Console

    from ultron.ui import theme

    buf = io.StringIO()
    fake = Console(file=buf, width=100, force_terminal=True)
    monkeypatch.setattr(theme, "console", fake)
    theme.UI.render_response("Hello.")
    out = buf.getvalue()
    assert "\x1b[38;2;251;108;0m" in out, "response border is not #FB6C00"


def test_response_panel_header_is_just_the_wordmark(monkeypatch):
    # The response box header carries only the ULTRON wordmark — no model,
    # version or timestamp repeating context already established in the
    # startup banner.
    import io

    from rich.console import Console

    from ultron.ui import theme

    buf = io.StringIO()
    fake = Console(file=buf, width=100, force_terminal=False)
    monkeypatch.setattr(theme, "console", fake)
    theme.UI.render_response("Hello.")
    lines = buf.getvalue().splitlines()
    top = lines[0]
    assert "ULTRON" in top, "response header must carry the ULTRON wordmark"
    assert "╭" in top  # it sits on the panel's top border
    # Nothing else in the header: no model name, no "vX.Y.Z" version, no
    # clock timestamp (a colon would appear in HH:MM:SS).
    for forbidden in ("gemma", "llama", "v0.", "Ultron AI", ":"):
        assert forbidden not in top, f"response header repeats {forbidden!r}"


def test_banner_logo_uses_flame_palette_gradient(monkeypatch):
    # The 6 ASCII logo lines carry the 4 brand colours as the exact gradient:
    # #E73F1E, #E73F1E, #FB6C00, #FB6C00, #F9B637, #FFDD9C — each of the
    # deeper colours used twice so the flame transition reads smooth.
    import io
    import re

    from rich.console import Console

    from ultron.ui import theme

    buf = io.StringIO()
    fake = Console(file=buf, width=130, force_terminal=True)
    monkeypatch.setattr(theme, "console", fake)
    theme.UI.render_banner(model="llama3.2", cwd_short="~/ultron")

    # rich folds "bold" into the same SGR as the colour (\x1b[1;38;2;R;G;Bm),
    # so extract the RGB params per logo line (the lines made of block glyphs).
    logo_colors = []
    for line in buf.getvalue().splitlines():
        if "█" in line or "═" in line:
            match = re.search(r"38;2;(\d+);(\d+);(\d+)", line)
            assert match, f"logo line has no colour: {line!r}"
            logo_colors.append(
                f"#{int(match.group(1)):02X}{int(match.group(2)):02X}{int(match.group(3)):02X}"
            )
    assert logo_colors == [
        "#E73F1E", "#E73F1E",
        "#FB6C00", "#FB6C00",
        "#F9B637",
        "#FFDD9C",
    ], f"unexpected logo gradient: {logo_colors}"


def test_chat_session_prompt_async_has_no_bottom_toolbar(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from ultron.ui.session import ChatSession

    mock_session = MagicMock()
    mock_session.prompt_async = AsyncMock(return_value="hello")
    state = _state()
    chat_session = ChatSession(state=state, session=mock_session, agent_tag=lambda: "simple")

    import asyncio
    res = asyncio.run(chat_session.prompt_async())
    assert res == "hello"
    mock_session.prompt_async.assert_called_once()
    kwargs = mock_session.prompt_async.call_args.kwargs
    assert "bottom_toolbar" not in kwargs

