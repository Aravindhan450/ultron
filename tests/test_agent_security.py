"""
Tests for the security boundary wired into the simple agent's tool paths.

Every tool call the simple agent makes (direct execution, pending-action
confirmation, or multi-step plan step) is routed through boundary.check()
before anything runs. These tests exercise the three verdicts:

- deny    → hard block: no PendingAction is offered, nothing executes
- confirm → PendingAction so the CLI can ask the user first
- allow   → auto-execution (read-only / LOW-risk actions, or permissive mode)
"""

import asyncio

from ultron.core.agents.simple import SimpleAgent, execute_plan
from ultron.core.types import Role


class FakeEngine:
    """Scripted engine — no real LLM or network access needed."""

    def __init__(self, responses=()):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# allow → auto-execution (LOW risk)
# ---------------------------------------------------------------------------


def test_read_file_auto_executes(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    note = tmp_path / "notes.txt"
    note.write_text("hello world", encoding="utf-8")

    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run(f"read {note}"))
    assert msg.pending_action is None
    assert "hello world" in msg.content


def test_readonly_command_auto_executes():
    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run("run ls"))
    assert msg.pending_action is None
    assert msg.content and "Error" not in msg.content


# ---------------------------------------------------------------------------
# confirm → PendingAction (HIGH / CRITICAL risk)
# ---------------------------------------------------------------------------


def test_write_file_requires_confirmation(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run("create new.txt with content hi"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "write_file"
    assert not (tmp_path / "new.txt").exists()  # nothing written yet


def test_dangerous_command_still_confirms():
    # Dangerous patterns escalate to CRITICAL but the user keeps the final
    # say — the command is offered for confirmation, never auto-run.
    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run("run rm -rf /"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"
    assert msg.pending_action.target == "rm -rf /"


def test_state_changing_command_requires_confirmation():
    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run("run mkdir newdir"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"


# ---------------------------------------------------------------------------
# deny → hard block (guardrails)
# ---------------------------------------------------------------------------


def test_secret_write_blocked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run("create leak.txt with content AKIA1234567890ABCDEF"))
    assert msg.pending_action is None  # never offered for confirmation
    assert "Blocked by security" in msg.content
    assert not (tmp_path / "leak.txt").exists()


def test_unsafe_url_fetch_blocked():
    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run("fetch this page http://example.com"))
    assert msg.pending_action is None
    assert "Blocked by security" in msg.content


def test_path_escape_read_blocked(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")

    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run(f"read {outside}"))
    assert msg.pending_action is None
    assert "Blocked by security" in msg.content


def test_command_with_secret_blocked():
    # Even a read-only command is denied when the command string itself would
    # carry a credential (e.g. `grep <aws-key> file`).
    agent = SimpleAgent(FakeEngine())
    msg = _run(agent.run("run grep AKIA1234567890ABCDEF config.txt"))
    assert msg.pending_action is None
    assert "Blocked by security" in msg.content


def test_llm_fallback_read_file_gated(tmp_path, monkeypatch):
    # The legacy TOOL_CALL: read_file fallback path is gated too — a path-
    # escape read never executes there.
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")

    engine = FakeEngine(["none", f"TOOL_CALL: read_file: {outside}"])
    agent = SimpleAgent(engine)
    msg = _run(agent.run("read the file"))
    assert msg.pending_action is None
    assert "Blocked by security" in msg.content
    assert "outside_secret.txt" not in msg.content  # content never leaked


# ---------------------------------------------------------------------------
# multi-step plans: deny hard-stops; confirm runs unless strict mode
# ---------------------------------------------------------------------------


def test_plan_deny_step_blocks_and_stops():
    steps = [
        {"action": "write_file", "filename": "leak.txt", "content": "AKIA1234567890ABCDEF"},
        {"action": "read_file", "filename": "notes.txt"},
    ]
    results = asyncio.run(execute_plan(steps))
    assert "BLOCKED by security" in results[0]
    assert len(results) == 1  # plan stopped; the later step never ran


def test_plan_confirm_step_runs_in_interactive(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    steps = [{"action": "write_file", "filename": "ok.txt", "content": "hi"}]
    results = asyncio.run(execute_plan(steps))
    assert (tmp_path / "ok.txt").read_text() == "hi"
    assert "Step 1 (write_file)" in results[0]


def test_plan_confirm_step_skipped_in_strict(tmp_path, monkeypatch):
    import ultron.core.agents.security as sec
    from ultron.core.tools import paths
    from ultron.security import SecurityBoundary

    monkeypatch.setattr(sec, "_boundary", SecurityBoundary(mode="strict"))
    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    steps = [{"action": "write_file", "filename": "ok.txt", "content": "hi"}]
    results = asyncio.run(execute_plan(steps))
    assert "skipped" in results[0]
    assert not (tmp_path / "ok.txt").exists()


def test_multistep_user_request_gated(tmp_path, monkeypatch):
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    engine = FakeEngine(['[{"action": "write_file", "filename": "a.txt", "content": "hello"}]'])
    agent = SimpleAgent(engine)
    msg = _run(agent.run("create a.txt with hello, then read it back"))
    assert (tmp_path / "a.txt").read_text() == "hello"
    assert msg.role == Role.ASSISTANT
