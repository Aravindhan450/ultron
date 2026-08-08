"""
Tests for resource constraint awareness.

The monitor measures every command run (wall/CPU time, peak RSS), records
history for future forecasts, and escalates forecast-heavy commands to
confirmation before they run. Forecasts come from static command-family
patterns plus measured history. The run-history DB is repointed at a temp
file for every test.
"""

import asyncio

import pytest

from ultron.core.agents import security as agent_security
from ultron.core.agents.simple import (
    SimpleAgent,
    _generic_target_content,
    detect_resource_intent,
    handle_parallel,
)
from ultron.core.tools import resource_monitor as rm
from ultron.core.tools.builtin.command_runner import run_command, run_parallel
from ultron.security import SecurityBoundary
from ultron.security.models import RiskTier


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    """Every test uses its own run-history database."""
    monkeypatch.setattr(rm, "RESOURCES_DB_PATH", tmp_path / "resources.db")


@pytest.fixture
def interactive_mode(monkeypatch):
    monkeypatch.setattr(
        agent_security, "_boundary", SecurityBoundary(mode="interactive")
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# System snapshot
# ---------------------------------------------------------------------------


def test_check_resources_reports_system():
    result = rm.check_resources()
    assert "System resources" in result
    assert "CPU" in result


def test_system_snapshot_shape():
    snap = rm.system_snapshot()
    assert "cpu_count" in snap
    assert "load" in snap
    assert "memory_total_gb" in snap


# ---------------------------------------------------------------------------
# Forecast engine (static patterns)
# ---------------------------------------------------------------------------


def test_forecast_pip_install_heavy():
    fc = rm.forecast_command("pip install requests")
    assert fc["severity"] == rm.HEAVY
    assert fc["duration_s"] is not None
    assert fc["peak_mb"] is not None
    assert any("pattern" in r for r in fc["reasons"])


def test_forecast_find_root_critical():
    assert rm.forecast_command("find /")["severity"] == rm.CRITICAL


def test_forecast_pytest_moderate():
    assert rm.forecast_command("pytest -v")["severity"] == rm.MODERATE


def test_forecast_light_no_pattern():
    fc = rm.forecast_command("ls -la")
    assert fc["severity"] == rm.LIGHT
    assert fc["duration_s"] is None


def test_forecast_warning_only_above_light():
    warning = rm.forecast_warning("pip install x")
    assert warning is not None
    assert "heavy" in warning
    # The warning text carries no label prefix — callers add it, so notes
    # never read "[resources] note: [resources] ...".
    assert not warning.startswith("[resources]")
    assert rm.forecast_warning("ls -la") is None


def test_forecast_script_run_requires_first_token():
    # The interpreter must be the command; a bare mention is not a script run.
    assert rm.forecast_command("echo python is fun")["severity"] == rm.LIGHT
    assert rm.forecast_command("python3 app.py")["severity"] == rm.MODERATE
    assert rm.forecast_command("sudo python3 app.py")["severity"] == rm.MODERATE


def test_forecast_npm_i_heavy():
    assert rm.forecast_command("npm i")["severity"] == rm.HEAVY
    assert rm.forecast_command("pnpm i")["severity"] == rm.HEAVY


def test_forecast_severity_helper():
    assert rm.forecast_severity("find /") == rm.CRITICAL
    assert rm.forecast_severity("ls") == rm.LIGHT


# ---------------------------------------------------------------------------
# History learning
# ---------------------------------------------------------------------------


def test_forecast_learns_from_history():
    rm.record_run("sleep 1", duration=1.5, peak_mb=300.0, cpu_seconds=0.1)
    fc = rm.forecast_command("sleep 1")
    assert fc["severity"] == rm.MODERATE  # small but non-trivial measured run
    assert any("last run took" in r for r in fc["reasons"])
    assert fc["duration_s"] == 1.5


def test_forecast_history_overrides_static():
    # A measured slow run upgrades the static 'find' profile via reality.
    rm.record_run("find .", duration=400.0, peak_mb=500.0)
    fc = rm.forecast_command("find .")
    assert fc["severity"] == rm.CRITICAL
    assert fc["duration_s"] == 400.0


def test_history_scoped_to_family():
    rm.record_run("pip install x", duration=300.0)
    # 'ls' must not inherit pip's profile.
    assert rm.forecast_command("ls -la")["duration_s"] is None


def test_history_family_isolates_pkg_subcommands():
    # One heavy 'pip install' must not taint 'pip list' (same tool, different
    # subcommand) — families now include the verb for package managers.
    rm.record_run("pip install requests", duration=300.0)
    fc = rm.forecast_command("pip list")
    assert fc["severity"] == rm.LIGHT
    assert fc["duration_s"] is None
    # The same subcommand still inherits its own (now heavier) history.
    assert rm.forecast_command("pip install other")["severity"] in {
        rm.HEAVY,
        rm.CRITICAL,
    }


# ---------------------------------------------------------------------------
# Measured run reporting
# ---------------------------------------------------------------------------


def test_run_command_reports_resources():
    result = run_command("echo hello")
    assert "Exit code: 0" in result
    assert "[resources] finished in" in result


def test_run_command_failure_reports_resources():
    result = run_command("exit 3")
    assert "Exit code: 3" in result
    assert "[resources] finished in" in result


def test_run_parallel_reports_batch_resources():
    result = run_parallel(["echo hi", "echo world"])
    assert "[resources] batch: 2 commands in" in result
    assert "[resources] finished in" in result


def test_runs_are_recorded():
    run_command("echo hello")
    runs = rm._recent_runs("echo")
    assert runs, "measured run should be stored for future forecasts"


# ---------------------------------------------------------------------------
# Registry + security
# ---------------------------------------------------------------------------


def test_registry_registers_resource_tools():
    from ultron.core.tools.registry import get_tool

    assert get_tool("check_resources") is not None
    assert get_tool("resource_forecast") is not None


def test_classify_resource_tools_low():
    boundary = SecurityBoundary(mode="interactive")
    assert boundary.classify_action("check_resources", "") == RiskTier.LOW
    assert boundary.classify_action("resource_forecast", "pip install x") == RiskTier.LOW


def test_generic_target_content_resource_tools():
    target, content = _generic_target_content(
        "resource_forecast", {"command": "pip install x"}
    )
    assert target == "pip install x"
    assert content is None
    target, content = _generic_target_content("check_resources", {})
    assert target == ""
    assert content is None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def test_detect_resource_check_phrases():
    for text in (
        "check system resources",
        "how is my system",
        "how much memory is free",
        "cpu usage",
        "system load",
    ):
        m = detect_resource_intent(text)
        assert m is not None and m["action"] == "check", text


def test_detect_resource_forecast_phrases():
    m = detect_resource_intent("resource forecast for pip install")
    assert m == {"action": "forecast", "command": "pip install"}
    m = detect_resource_intent("how heavy is find /")
    assert m["action"] == "forecast"
    assert m["command"] == "find /"
    m = detect_resource_intent("will this be heavy")
    assert m["action"] == "forecast"
    assert m["command"] is None  # needs a command clarification


@pytest.mark.parametrize(
    "text",
    [
        "search for python news",
        "run ls",
        "read the config file",
        "check my code",  # lint phrasing, not resources
        "what do you know about databases",
        "post to http://localhost:8000 with body {}",
        "remember that Paris is in France",
    ],
)
def test_detect_resource_negative(text):
    assert detect_resource_intent(text) is None


# ---------------------------------------------------------------------------
# Agent escalation (proactive warnings)
# ---------------------------------------------------------------------------


def test_agent_escalates_heavy_command_to_confirmation(interactive_mode):
    agent = SimpleAgent(None)
    msg = _run(agent.run("run find /"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"
    assert "[resources] ⚠" in msg.content


def test_agent_forecast_heavy_batch_escalates(interactive_mode):
    commands = [f"echo cmd{i}" for i in range(9)]  # large batch
    msg = handle_parallel(commands)
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_parallel"
    assert "[resources] ⚠" in msg.content
    assert "9 commands" in msg.content


def test_agent_moderate_command_runs_with_note(interactive_mode, monkeypatch):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(reg.TOOLS, "run_command", lambda command: "fake output")
    agent = SimpleAgent(None)
    msg = _run(agent.run("run pytest -v"))
    # pytest is LOW + moderate forecast: auto-runs, note appended.
    assert msg.pending_action is None
    assert "fake output" in msg.content
    assert "[resources]" in msg.content
    assert "note:" in msg.content


def test_agent_heavy_escalation_single_prefix(interactive_mode):
    agent = SimpleAgent(None)
    msg = _run(agent.run("run find /"))
    assert msg.pending_action is not None
    assert msg.content.count("[resources]") == 1


def test_agent_permissive_heavy_warns_without_prompting(monkeypatch):
    monkeypatch.setattr(
        agent_security, "_boundary", SecurityBoundary(mode="permissive")
    )
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(reg.TOOLS, "run_command", lambda command: "fake output")
    agent = SimpleAgent(None)
    msg = _run(agent.run("run find /"))
    # Permissive mode promises no prompts: heavy commands run with the
    # resource warning attached instead of escalating to confirmation.
    assert msg.pending_action is None
    assert "fake output" in msg.content
    assert "[resources] ⚠" in msg.content


def test_agent_light_command_runs_cleanly(interactive_mode, monkeypatch):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(reg.TOOLS, "run_command", lambda command: "ok")
    agent = SimpleAgent(None)
    msg = _run(agent.run("run ls"))
    assert msg.pending_action is None
    assert msg.content == "ok"


def test_agent_check_resources(monkeypatch, interactive_mode):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(reg.TOOLS, "check_resources", lambda: "CPU: 8 cores")
    agent = SimpleAgent(None)
    msg = _run(agent.run("check system resources"))
    assert msg.pending_action is None
    assert "CPU: 8 cores" in msg.content


def test_agent_resource_forecast(monkeypatch, interactive_mode):
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(
        reg.TOOLS, "resource_forecast", lambda command: f"forecast for {command}"
    )
    agent = SimpleAgent(None)
    msg = _run(agent.run("resource forecast for pip install"))
    assert "forecast for pip install" in msg.content


def test_execute_plan_annotates_heavy_step(interactive_mode, monkeypatch):
    from ultron.core.agents.simple import execute_plan
    from ultron.core.tools import registry as reg

    monkeypatch.setitem(reg.TOOLS, "run_command", lambda command: "fake done")
    results = _run(execute_plan([{"action": "run_command", "command": "find /"}]))
    joined = "\n".join(results)
    assert "⚠" in joined
    assert "fake done" in joined
