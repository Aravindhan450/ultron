"""Tests for the multi-step intent detector and the task planner.

``plan_task`` is exercised against a scripted fake engine — no live model needed.
"""


import asyncio
import json

import httpx
import pytest

from ultron.core.agents.simple import detect_multistep_intent, plan_task


class FakeEngine:
    """Scripted engine stub: returns a canned response, never calls a model."""

    def __init__(self, response: str = "", error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[list] = []

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.response


# ---------------------------------------------------------------------------
# detect_multistep_intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_input",
    [
        "create world.txt, write hello world in it, then read it back",
        "read file A and then write file B",
        "run pytest then open the report",
        "save the draft, then remember the deadline",
    ],
)
def test_detect_multistep_intent_true(user_input):
    """Compound requests (sequence marker + 2+ action verbs) are detected."""
    assert detect_multistep_intent(user_input) is True


@pytest.mark.parametrize(
    "user_input",
    [
        "read test.txt",
        "remember that I use FastAPI",
        "write a note",
        "show the diff",
        "read it then",  # sequence marker but only one action verb
    ],
)
def test_detect_multistep_intent_false(user_input):
    """Simple requests, or a sequence marker with a single verb, are not."""
    assert detect_multistep_intent(user_input) is False


# ---------------------------------------------------------------------------
# plan_task (mocked engine)
# ---------------------------------------------------------------------------


def test_plan_task_parses_valid_json():
    steps = [
        {"action": "write_file", "filename": "world.txt", "content": "hello world"},
        {"action": "read_file", "filename": "world.txt"},
    ]
    engine = FakeEngine(response=json.dumps(steps))
    result = asyncio.run(
        plan_task(
            "create world.txt, write hello world in it, then read it back",
            engine,
        )
    )
    assert result == steps
    # The planning prompt was actually sent to the engine, with the request.
    assert len(engine.calls) == 1
    sent_prompt = engine.calls[0][0]["content"]
    assert "create world.txt" in sent_prompt


def test_plan_task_strips_markdown_fences():
    engine = FakeEngine(
        response='```json\n[{"action": "run_command", "command": "pytest -v"}]\n```'
    )
    result = asyncio.run(plan_task("run tests", engine))
    assert result == [{"action": "run_command", "command": "pytest -v"}]


def test_plan_task_non_json_returns_none():
    engine = FakeEngine(response="sure, I'll do that")
    assert asyncio.run(plan_task("do things", engine)) is None


def test_plan_task_non_list_returns_none():
    engine = FakeEngine(response='{"action": "read_file", "filename": "a.txt"}')
    assert asyncio.run(plan_task("read a file", engine)) is None


def test_plan_task_empty_response_returns_none():
    assert asyncio.run(plan_task("do things", FakeEngine(response=""))) is None


def test_plan_task_filters_unknown_actions():
    response = json.dumps(
        [
            {"action": "read_file", "filename": "a.txt"},
            {"action": "rm_rf", "command": "rm -rf /"},
            "not a dict",
        ]
    )
    result = asyncio.run(plan_task("mixed request", FakeEngine(response=response)))
    assert result == [{"action": "read_file", "filename": "a.txt"}]


def test_plan_task_no_valid_steps_returns_none():
    engine = FakeEngine(response=json.dumps([{"action": "rm_rf", "command": "rm -rf /"}]))
    assert asyncio.run(plan_task("boom", engine)) is None


@pytest.mark.parametrize(
    "error",
    [httpx.HTTPError("boom"), OSError("boom"), ValueError("boom")],
)
def test_plan_task_engine_error_returns_none(error):
    """Engine failures (HTTP, I/O, parse) fall back silently to None."""
    assert asyncio.run(plan_task("do things", FakeEngine(error=error))) is None
