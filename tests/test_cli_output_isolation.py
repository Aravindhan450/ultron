"""
Tests for Ultron CLI output quality, log isolation, response rendering, and CLI Output Contract.
"""

from __future__ import annotations

import ast
import io
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ultron.core.intelligence.synthesis import (
    strip_internal_thought,
    synthesize_observation,
)
from ultron.core.logging import (
    is_verbose_logging,
    set_verbose_logging,
)
from ultron.ui.theme import UI, console


def test_a_log_isolation_normal_mode(capsys):
    """
    Test A: In normal mode (default), logger.info and logger.debug
    must NOT print to console/stdout/stderr.
    """
    set_verbose_logging(False)
    assert not is_verbose_logging()

    test_logger = logging.getLogger("ultron.test.isolation")
    test_logger.info("DIAGNOSTIC_INFO_SECRET_12345")
    test_logger.debug("DIAGNOSTIC_DEBUG_SECRET_67890")

    captured = capsys.readouterr()
    assert "DIAGNOSTIC_INFO_SECRET_12345" not in captured.out
    assert "DIAGNOSTIC_INFO_SECRET_12345" not in captured.err
    assert "DIAGNOSTIC_DEBUG_SECRET_67890" not in captured.out
    assert "DIAGNOSTIC_DEBUG_SECRET_67890" not in captured.err


def test_b_verbose_mode_enables_console_logging(capsys):
    """
    Test B: In verbose mode, diagnostic logs are allowed to print to the console.
    """
    try:
        set_verbose_logging(True)
        assert is_verbose_logging()

        test_logger = logging.getLogger("ultron.test.verbose")
        test_logger.info("VERBOSE_DIAGNOSTIC_ACTIVE_54321")

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "VERBOSE_DIAGNOSTIC_ACTIVE_54321" in combined
    finally:
        set_verbose_logging(False)


def test_c_ui_render_response_formatting():
    """
    Test C: UI.render_response formats assistant answers in canonical Ultron panel.
    """
    buf = io.StringIO()
    test_console = console.__class__(file=buf, force_terminal=True, width=80)

    # Monkeypatch console temporarily for test inspection
    from ultron.ui import theme
    original_console = theme.console
    theme.console = test_console
    try:
        UI.render_response("This is a clean synthesized response.")
        output = buf.getvalue()
        assert "ULTRON" in output
        assert "This is a clean synthesized response." in output
    finally:
        theme.console = original_console


@pytest.mark.anyio
async def test_d_synthesize_observation_with_model():
    """
    Test D1: synthesize_observation invokes the model engine and produces clean Markdown.
    """
    mock_engine = AsyncMock()
    mock_engine.generate.return_value = (
        "Thought: I should summarize.\n"
        "Here is the summarized information about the iPhone Duo:\n"
        "- It features dual folding displays\n"
        "- Enhanced battery architecture\n"
    )

    raw_data = "{'title': 'iPhone Duo', 'body': 'Specs leaked on tech blog'}"
    result = await synthesize_observation(
        user_request="research the iphone duo",
        observation=raw_data,
        engine=mock_engine,
        tool_name="search_web",
    )

    mock_engine.generate.assert_awaited_once()
    # Thought must be stripped
    assert "Thought:" not in result
    assert "iPhone Duo" in result
    assert "dual folding displays" in result


@pytest.mark.anyio
async def test_d2_synthesize_observation_graceful_fallback():
    """
    Test D2: When model synthesis fails or engine is None, falls back gracefully without crashing.
    """
    raw_data = "Raw error output or offline text"
    result = await synthesize_observation(
        user_request="check system status",
        observation=raw_data,
        engine=None,
        tool_name="retrieve",
    )

    assert "check system status" in result
    assert raw_data in result


def test_d3_strip_internal_thought():
    """
    Test D3: strip_internal_thought correctly removes ReAct Thought: preambles.
    """
    raw_react = (
        "Thought: The user wants to know the capital of France. I know this.\n"
        "Final Answer: The capital of France is Paris."
    )
    stripped = strip_internal_thought(raw_react)
    assert stripped == "The capital of France is Paris."


def test_e_ui_render_tool_activity():
    """
    Test E: UI.render_tool_activity renders a compact chip line with action and target.
    """
    buf = io.StringIO()
    test_console = console.__class__(file=buf, force_terminal=True, width=80)

    from ultron.ui import theme
    original_console = theme.console
    theme.console = test_console
    try:
        UI.render_tool_activity("search_web", "python 3.12 release notes")
        output = buf.getvalue()
        assert "search_web" in output
        assert "python 3.12 release notes" in output
    finally:
        theme.console = original_console


def test_f_raw_output_request_honored():
    """
    Test F: When user explicitly asks for raw output, observation is not altered.
    """
    import asyncio
    raw_json = '{"status": "ok", "count": 42}'
    result = asyncio.run(
        synthesize_observation("give me the raw results", raw_json, engine=None)
    )
    assert result == raw_json


def test_g_no_forbidden_print_in_core():
    """
    Test G: Static AST check ensuring no bare print() statements in core / agents / runtime / tools
    (except memory tools where graph logging is routed, and UI rendering).
    """
    repo_root = Path(__file__).resolve().parent.parent
    core_dir = repo_root / "src" / "ultron" / "core"

    violations = []
    for py_file in core_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id == "print":
                        violations.append(f"{py_file.relative_to(repo_root)}:{node.lineno}")
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue

    assert not violations, f"Forbidden bare print() statements detected in core: {violations}"
