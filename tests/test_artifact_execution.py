"""
Tests for Real Artifact Creation & Execution Layer.
"""

from __future__ import annotations

import asyncio

import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.intelligence.task_classification import (
    classify_task_deterministic,
)
from ultron.core.intelligence.task_planning import (
    build_artifact_plan,
)
from ultron.core.tools.paths import set_active_project_dir
from ultron.core.types import (
    TaskState,
    TaskType,
)


@pytest.fixture(autouse=True)
def reset_active_project_dir_fixture():
    set_active_project_dir(None)
    yield
    set_active_project_dir(None)


class ScriptedEngine:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, messages) -> str:
        self.calls.append(messages)
        return self.responses.pop(0) if self.responses else ""


def _run(coro):
    return asyncio.run(coro)


def test_artifact_classification_distinguishes_explanation_vs_building():
    c_info = classify_task_deterministic("Explain how to build a Python expense tracker.")
    assert c_info.task_type == TaskType.INFORMATIONAL

    c_guide = classify_task_deterministic("Guide on how to implement authentication.")
    assert c_guide.task_type == TaskType.INFORMATIONAL

    c_build = classify_task_deterministic("Build a Python expense tracker application.")
    assert c_build.task_type == TaskType.SOFTWARE_ENGINEERING

    c_create = classify_task_deterministic("Create a TodoList application in TodoList folder.")
    assert c_create.task_type == TaskType.SOFTWARE_ENGINEERING


def test_build_artifact_plan_structure():
    plan = build_artifact_plan("Build an expense tracker")
    assert len(plan.steps) == 3
    assert "implement" in plan.steps[0].description.lower()
    assert "execute" in plan.steps[1].description.lower()
    assert "verify" in plan.steps[2].description.lower()
    assert plan.steps[1].dependencies == [1]
    assert plan.steps[2].dependencies == [1, 2]


def test_react_coding_gate_circuit_breaks_repeated_failures():
    engine = ScriptedEngine(["OK"])
    agent = ReActAgent(engine)

    task = TaskState(goal="Debug missing file")
    # Simulate repeated failure of read_file on nonexistent file
    task.record_tool_execution(tool_name="read_file", target="missing.py", success=False, detail="File not found")
    task.record_tool_execution(tool_name="read_file", target="missing.py", success=False, detail="File not found")

    gate_result = agent._coding_gate(task, "read_file", {"file_path": "missing.py"})
    assert gate_result is not None
    assert "has already failed 2 times" in gate_result
    assert "Do not retry this exact operation" in gate_result


def test_react_verification_requires_plan_terminality():
    task = TaskState(goal="Build app")
    plan = build_artifact_plan("Build app")
    task.attach_plan(plan)

    assert not plan.is_satisfied()
    assert task.is_complete() is False
