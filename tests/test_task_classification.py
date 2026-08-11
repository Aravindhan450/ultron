"""Tests for the general task classification layer (Fix #2)."""

from __future__ import annotations

import asyncio

from ultron.core.intelligence.task_classification import (
    classify_task,
    classify_task_deterministic,
    extract_goal,
)
from ultron.core.types import TaskClassification, TaskType


class FakeEngine:
    """Minimal fake LLM engine returning canned responses."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages) -> str:
        self.calls += 1
        return self.responses.pop(0)


def _run(coro):
    return asyncio.run(coro)


def classify(request: str) -> TaskClassification:
    return classify_task_deterministic(request)


# ---------------------------------------------------------------------------
# Task-type scenarios (deterministic, no LLM)
# ---------------------------------------------------------------------------


def test_informational_task():
    result = classify("Explain dependency injection.")
    assert result.task_type is TaskType.INFORMATIONAL
    assert result.requires_actions is False
    assert result.goal.endswith("?")


def test_simple_action():
    result = classify("List all running processes.")
    assert result.task_type is TaskType.SIMPLE_ACTION


def test_file_operation():
    result = classify("List all Python files.")
    assert result.task_type in {TaskType.FILE_OPERATION, TaskType.SIMPLE_ACTION}


def test_generic_multi_step_task():
    result = classify("Find all Python files, count their lines, and save the result.")
    assert result.task_type is TaskType.MULTI_STEP


def test_feature_implementation():
    result = classify("Create a FastAPI backend.")
    assert result.task_type is TaskType.SOFTWARE_ENGINEERING


def test_new_project_creation():
    result = classify("Create a TodoList application in a separate folder TodoList")
    assert result.task_type is TaskType.SOFTWARE_ENGINEERING


def test_debugging_task():
    result = classify("Fix the failing login tests.")
    assert result.task_type in {TaskType.DEBUGGING, TaskType.SOFTWARE_ENGINEERING}


def test_debugging_crash():
    result = classify("Find why the server crashes on startup.")
    assert result.task_type is TaskType.DEBUGGING


def test_refactoring_task():
    result = classify("Refactor this authentication service.")
    assert result.task_type is TaskType.SOFTWARE_ENGINEERING


def test_code_review_task():
    result = classify("Review this repository for security problems.")
    assert result.task_type is TaskType.CODE_REVIEW


def test_repository_analysis():
    result = classify("Analyze this repository and explain how authentication works.")
    assert result.task_type is TaskType.RESEARCH


def test_dependency_upgrade():
    result = classify("Upgrade this project from React 18 to React 19.")
    assert result.task_type is TaskType.SOFTWARE_ENGINEERING


def test_existing_project_modification():
    result = classify("Add a database migration to the existing project.")
    assert result.task_type in {TaskType.DATA_OPERATION, TaskType.SOFTWARE_ENGINEERING}


def test_system_operation():
    result = classify("Configure Redis for this project.")
    assert result.task_type in {
        TaskType.SYSTEM_OPERATION,
        TaskType.CONFIGURATION,
        TaskType.SOFTWARE_ENGINEERING,
    }


def test_configuration_task():
    result = classify("Configure the .env file for this project.")
    assert result.task_type is TaskType.CONFIGURATION


def test_filename_nouns_do_not_become_engineering():
    assert classify("create a file app.txt, write hello into it, then read it").task_type \
        is TaskType.MULTI_STEP
    assert classify("create a directory called TestDir").task_type in {
        TaskType.FILE_OPERATION,
        TaskType.SIMPLE_ACTION,
    }


# ---------------------------------------------------------------------------
# Clarification
# ---------------------------------------------------------------------------


def test_clarification_required_when_no_target():
    result = classify("Deploy this.")
    assert result.task_type is TaskType.SYSTEM_OPERATION
    assert result.clarification_required is True
    assert result.clarification_questions


def test_no_clarification_for_specific_deploy():
    result = classify("Deploy the API to production.")
    assert result.task_type is TaskType.SYSTEM_OPERATION
    assert result.clarification_required is False


# ---------------------------------------------------------------------------
# Goal understanding (goal is separate from any tool call)
# ---------------------------------------------------------------------------


def test_goal_extraction_rewrites_fix_request():
    assert extract_goal("Fix the failing authentication tests.") == (
        "Make the authentication tests pass."
    )


def test_goal_is_not_a_tool_call():
    result = classify("Fix the failing login tests.")
    assert not result.goal.lower().startswith("run ")
    assert "pytest" not in result.goal.lower()


def test_goal_strips_polite_prefixes():
    assert extract_goal("Can you please create a FastAPI backend?") == (
        "Create a FastAPI backend."
    )


# ---------------------------------------------------------------------------
# Purity and LLM fallback path
# ---------------------------------------------------------------------------


def test_classifier_is_pure_and_deterministic():
    result = classify("Create a TodoList application")
    assert isinstance(result, TaskClassification)
    assert result.task_type in TaskType


def test_llm_path_used_only_for_ambiguous_requests():
    engine = FakeEngine(
        [
            (
                '{"task_type": "debugging", "goal": "Diagnose and resolve the '
                'situation.", "clarification_required": false, '
                '"clarification_questions": []}'
            )
        ]
    )
    result = _run(classify_task("handle the situation", engine))
    assert result.task_type is TaskType.DEBUGGING
    assert result.goal == "Diagnose and resolve the situation."
    assert engine.calls == 1


def test_llm_not_called_when_deterministic_matches():
    engine = FakeEngine([])
    result = _run(classify_task("Create a FastAPI backend.", engine))
    assert result.task_type is TaskType.SOFTWARE_ENGINEERING
    assert engine.calls == 0


def test_llm_garbage_falls_back_to_deterministic():
    engine = FakeEngine(["this is not json"])
    result = _run(classify_task("handle the situation", engine))
    assert result.task_type is TaskType.INFORMATIONAL


def test_llm_invalid_task_type_ignored():
    engine = FakeEngine(['{"task_type": "do_stuff", "goal": "x"}'])
    result = _run(classify_task("handle the situation", engine))
    assert result.task_type is TaskType.INFORMATIONAL


def test_task_type_requires_actions_flag():
    assert TaskType.INFORMATIONAL.requires_actions is False
    assert TaskType.SOFTWARE_ENGINEERING.requires_actions is True
    assert TaskType.DEBUGGING.requires_actions is True
