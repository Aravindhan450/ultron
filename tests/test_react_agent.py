"""
Unit tests for the ReAct agent's reasoning loop.

The loop is driven by a scripted FakeEngine so no real LLM or network access
is required. Read-only tool execution is exercised against real tools
(read_file on a temp file) to verify Observations flow back into the loop.

Note: the state-modifying tool paths are tested only as far as the
PendingAction confirmation boundary — actually executing them requires the
interactive CLI confirmation flow, which is out of scope here.
"""

import asyncio
import json

from ultron.core.agents.react import ReActAgent, extract_tool_call
from ultron.core.types import Role


class FakeEngine:
    """
    Minimal engine stand-in that returns scripted responses in order and
    records every message list it receives.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""

    async def stream(self, messages, **kwargs):
        yield ""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# extract_tool_call
# ---------------------------------------------------------------------------


def test_extract_tool_call_fenced_and_bare():
    fenced = (
        'Thought: I need to read the file.\n'
        '```json\n{"tool": "read_file", "arguments": {"file_path": "a.txt"}}\n```'
    )
    assert extract_tool_call(fenced) == {
        "tool": "read_file",
        "arguments": {"file_path": "a.txt"},
    }

    bare = '{"tool": "run_command", "arguments": {"command": "ls"}}'
    assert extract_tool_call(bare)["tool"] == "run_command"


def test_extract_tool_call_returns_none_for_plain_text():
    assert extract_tool_call("Hello! Here is my final answer.") is None
    assert extract_tool_call("") is None
    assert extract_tool_call('```json\n{"note": "no tool key"}\n```') is None


def test_extract_tool_call_handles_nested_braces():
    # The body string contains braces; naive non-greedy matching would stop
    # at the first '}' and produce unparseable JSON.
    text = (
        '```json\n'
        '{"tool": "make_http_request", '
        '"arguments": {"method": "POST", "url": "https://x.test", '
        '"body": "{\\"a\\": 1}"}}\n'
        '```'
    )
    call = extract_tool_call(text)
    assert call is not None
    assert call["tool"] == "make_http_request"


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------


def test_final_answer_returned_directly():
    engine = FakeEngine(["Hello there!"])
    agent = ReActAgent(engine)
    msg = _run(agent.run("hi"))
    assert msg.role == Role.ASSISTANT
    assert msg.content == "Hello there!"
    assert len(engine.calls) == 1  # no tool round-trips


def test_tool_loop_read_file_then_answer(tmp_path, monkeypatch):
    # Allow read_file to access the pytest temp dir.
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)

    note = tmp_path / "notes.txt"
    note.write_text("hello world", encoding="utf-8")

    engine = FakeEngine(
        [
            f'```json\n{{"tool": "read_file", "arguments": {{"file_path": "{note}"}}}}\n```',
            "The file says: hello world",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("read the notes file"))
    assert msg.content == "The file says: hello world"

    # The second LLM call must contain the tool observation from the first.
    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert "hello world" in observations[0]["content"]
    assert observations[0]["name"] == "read_file"


def test_run_command_requires_confirmation():
    # State-changing commands (HIGH tier) go through the confirmation gate.
    engine = FakeEngine(
        ['```json\n{"tool": "run_command", "arguments": {"command": "mkdir newdir"}}\n```']
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("make a directory"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "run_command"
    assert msg.pending_action.target == "mkdir newdir"
    assert len(engine.calls) == 1  # loop stops, no silent execution


def test_readonly_command_executes_directly():
    # Read-only commands (LOW tier) are auto-allowed by the boundary and run
    # inside the loop; the output feeds back as an Observation.
    engine = FakeEngine(
        ['```json\n{"tool": "run_command", "arguments": {"command": "ls"}}\n```']
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("list files"))
    assert msg.pending_action is None
    assert len(engine.calls) == 2  # tool call + final answer
    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert len(observations[0]["content"]) > 0  # real `ls` output


def test_secret_write_is_blocked(tmp_path, monkeypatch):
    # Guardrail hard block: content carrying a credential is denied before it
    # is even offered for confirmation, and nothing touches the filesystem.
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    engine = FakeEngine(
        [
            (
                '```json\n{"tool": "write_file", "arguments": '
                '{"filename": "leak.txt", "content": "aws key AKIA1234567890ABCDEF"}}\n```'
            )
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("write a file"))
    assert msg.pending_action is None  # never offered for confirmation
    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert "Blocked by security" in observations[0]["content"]
    assert not (tmp_path / "leak.txt").exists()  # nothing written


def test_write_file_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)
    engine = FakeEngine(
        [
            (
                '```json\n{"tool": "write_file", "arguments": '
                '{"filename": "new.txt", "content": "hi"}}\n```'
            )
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("create new.txt with content hi"))
    assert msg.pending_action is not None
    assert msg.pending_action.action_type == "write_file"
    assert msg.pending_action.target == "new.txt"
    assert msg.pending_action.content == "hi"
    # Nothing may be written to disk before the user confirms.
    assert not (tmp_path / "new.txt").exists()


def test_tool_loop_run_tool_batch_then_answer(tmp_path, monkeypatch):
    # ReAct drives run_tool_batch as one tool call; the synthesized report
    # flows back as an Observation so the model can answer from it.
    from ultron.core.tools import paths

    monkeypatch.setattr(paths, "ALLOWED_BASE_DIR", tmp_path)

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("alpha content", encoding="utf-8")
    b.write_text("beta content", encoding="utf-8")

    calls_json = json.dumps(
        [
            {"tool": "read_file", "arguments": {"file_path": str(a)}},
            {"tool": "read_file", "arguments": {"file_path": str(b)}},
        ]
    )
    tool_block = json.dumps(
        {"tool": "run_tool_batch", "arguments": {"calls_json": calls_json}}
    )
    engine = FakeEngine(
        [
            f"```json\n{tool_block}\n```",
            "The files say alpha and beta.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("read both files"))
    assert msg.content == "The files say alpha and beta."

    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert observations[0]["name"] == "run_tool_batch"
    assert "Parallel batch" in observations[0]["content"]
    assert "alpha content" in observations[0]["content"]
    assert "beta content" in observations[0]["content"]


def test_run_tool_batch_routes_directly_without_outer_url_scan(monkeypatch):
    # A valid batch whose member carries an http:// URL must NOT be blocked by
    # an outer content scan — run_tool_batch routes directly and the per-member
    # gate decides (check_connectivity on an http URL would be a member-level
    # guardrail call, not a blanket batch block).
    from ultron.core.intelligence import parallel_tools as pt

    captured = {}
    real_execute = pt.execute_batch

    def spy_execute(calls):
        captured["calls"] = calls
        return real_execute(calls)

    monkeypatch.setattr(pt, "execute_batch", spy_execute)

    calls_json = json.dumps(
        [
            {"tool": "check_connectivity", "arguments": {"url": "http://example.com"}},
            {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        ]
    )
    tool_block = json.dumps(
        {"tool": "run_tool_batch", "arguments": {"calls_json": calls_json}}
    )
    engine = FakeEngine(
        [
            f"```json\n{tool_block}\n```",
            "done",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("check the site and read readme"))
    assert "done" in msg.content
    # The batch reached the executor with both members intact.
    assert captured["calls"][0]["tool"] == "check_connectivity"
    assert captured["calls"][1]["tool"] == "read_file"


def test_run_tool_batch_missing_calls_json_is_observation():
    engine = FakeEngine(
        ['```json\n{"tool": "run_tool_batch", "arguments": {}}\n```', "ok"]
    )
    agent = ReActAgent(engine)
    _run(agent.run("batch something"))
    # The error is fed back as an Observation so the loop can recover.
    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert "calls_json" in observations[0]["content"]


def test_run_tool_batch_accepts_calls_list_argument():
    # The model may emit a structured list under 'calls' instead of a JSON
    # string under 'calls_json' — both spellings must work.
    calls_list = [
        {"tool": "read_file", "arguments": {"file_path": "README.md"}},
        {"tool": "read_file", "arguments": {"file_path": "pyproject.toml"}},
    ]
    tool_block = json.dumps(
        {"tool": "run_tool_batch", "arguments": {"calls": calls_list}}
    )
    engine = FakeEngine(
        [
            f"```json\n{tool_block}\n```",
            "done",
        ]
    )
    agent = ReActAgent(engine)
    _run(agent.run("read readme and pyproject"))
    # The synthesized batch report flows back as an Observation.
    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert "Parallel batch" in observations[0]["content"]
    assert "README.md" in observations[0]["content"]


def test_max_iterations_stops_loop():
    tool_json = '```json\n{"tool": "search_memories", "arguments": {"keyword": "zzz"}}\n```'
    engine = FakeEngine([tool_json] * 5)
    agent = ReActAgent(engine, max_iterations=3)
    msg = _run(agent.run("what do you remember?"))
    assert "maximum" in msg.content.lower()
    assert len(engine.calls) == 3


def test_unknown_tool_becomes_observation():
    engine = FakeEngine(
        [
            '```json\n{"tool": "does_not_exist", "arguments": {}}\n```',
            "Final answer here.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("do a thing"))
    assert msg.content == "Final answer here."
    second_call = engine.calls[1]
    assert any("does_not_exist" in m.get("content", "") for m in second_call)


def test_malformed_arguments_do_not_crash():
    # The model can emit a non-dict "arguments" value; the agent must coerce
    # it and feed back an error Observation rather than crash the loop.
    engine = FakeEngine(
        [
            '```json\n{"tool": "run_command", "arguments": ["ls"]}\n```',
            "Final answer.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("run something"))
    assert msg.content == "Final answer."
    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert "command" in observations[0]["content"]  # error observation


def test_readonly_http_result_continues_loop(monkeypatch):
    # A read-only tool that returns a plain ChatMessage (like handle_http's
    # GET branch) must feed back as an Observation, not terminate the loop.
    import ultron.core.agents.react as react_mod
    from ultron.core.types import ChatMessage as CM

    def fake_http(method, url, body=None):
        return CM(role=Role.ASSISTANT, content="Status: 200 OK")

    monkeypatch.setattr(react_mod, "handle_http", fake_http)
    engine = FakeEngine(
        [
            (
                '```json\n{"tool": "make_http_request", "arguments": '
                '{"method": "GET", "url": "https://example.com"}}\n```'
            ),
            "The API is healthy.",
        ]
    )
    agent = ReActAgent(engine)
    msg = _run(agent.run("check the API"))
    assert msg.content == "The API is healthy."
    second_call = engine.calls[1]
    observations = [m for m in second_call if m.get("role") == "tool"]
    assert len(observations) == 1
    assert "Status: 200 OK" in observations[0]["content"]


def test_system_prompt_injected_with_tool_schema():
    engine = FakeEngine(["answer"])
    agent = ReActAgent(engine)
    _run(agent.run("hello"))
    first_call = engine.calls[0]
    assert first_call[0]["role"] == "system"
    assert "Available Tools" in first_call[0]["content"]
    assert "read_file" in first_call[0]["content"]
    assert "run_command" in first_call[0]["content"]
