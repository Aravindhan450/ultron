from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ultron.core.intelligence.model_catalog import ModelRole, get_default_catalog
from ultron.core.intelligence.model_lifecycle import (
    LifecycleState,
    ModelHandle,
    ModelLifecycleManager,
)
from ultron.core.intelligence.model_router import (
    ComplexityLevel,
    ContextSize,
    MemoryPressure,
    ModelRouter,
    RoutingRequest,
    TaskRoutingState,
)
from ultron.core.runtime.result import RunResult
from ultron.core.runtime.runtime import AgentRuntime
from ultron.core.types import (
    ChatMessage,
    PendingAction,
    Role,
    TaskError,
    TaskState,
    TaskStatus,
    TaskType,
)


@pytest.fixture
def catalog():
    return get_default_catalog()


@pytest.fixture
def router(catalog):
    return ModelRouter(catalog)


@pytest.fixture
def lifecycle_mgr(catalog):
    mgr = ModelLifecycleManager()
    mgr.ensure_loaded = MagicMock(
        return_value=ModelHandle(
            model_spec=catalog.get_coding(),
            endpoint_url="http://mock:8080",
            state=LifecycleState.LOADED,
        )
    )
    return mgr


@pytest.fixture
def runtime(router, lifecycle_mgr):
    return AgentRuntime(router=router, lifecycle_manager=lifecycle_mgr)


def test_transition_to_repair_state_lifecycle():
    """Verify transition_to_repair brings TASK_FAILED back to TASK_RUNNING while preserving errors."""
    task = TaskState(goal="fix error in script", task_type=TaskType.SOFTWARE_ENGINEERING)
    task.record_failure("Execution crashed with ZeroDivisionError")

    assert task.status == TaskStatus.TASK_FAILED
    assert len(task.errors) == 1
    assert "ZeroDivisionError" in task.errors[0].message

    task.transition_to_repair()
    assert task.status == TaskStatus.TASK_RUNNING
    assert len(task.errors) == 1

    # Should allow resume and wait_for_confirmation without raising ValueError
    task.wait_for_confirmation()
    assert task.status == TaskStatus.WAITING_CONFIRMATION

    task.resume()
    assert task.status == TaskStatus.TASK_RUNNING


def test_routing_request_repair_vs_escalation(runtime):
    """Verify AgentRuntime generates REPAIR for <=2 errors and ESCALATION for >2 errors."""
    # 1 error -> REPAIR
    task = TaskState(goal="fix bug in calculator.py", task_type=TaskType.SOFTWARE_ENGINEERING)
    task.errors.append(TaskError(message="Syntax error at line 1"))
    req = runtime._build_routing_request(task.goal, task)
    assert req.task_state == TaskRoutingState.REPAIR
    assert req.coding is True

    # 2 errors -> REPAIR
    task.errors.append(TaskError(message="ImportError: module not found"))
    req2 = runtime._build_routing_request(task.goal, task)
    assert req2.task_state == TaskRoutingState.REPAIR
    assert req2.coding is True

    # 3 errors -> ESCALATION
    task.errors.append(TaskError(message="RecursionError: maximum depth exceeded"))
    req3 = runtime._build_routing_request(task.goal, task)
    assert req3.task_state == TaskRoutingState.ESCALATION
    assert req3.coding is True


def test_model_router_repair_selects_coding_specialist(router):
    """Verify ModelRouter selects CODING specialist during REPAIR on coding tasks."""
    req = RoutingRequest(
        task_description="Fix crash in error.py",
        complexity=ComplexityLevel.SIMPLE,
        coding=True,
        context_size=ContextSize.NORMAL,
        task_state=TaskRoutingState.REPAIR,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.CODING
    assert decision.selected_model.model_id == "qwen2.5-coder-7b-instruct"
    assert "Repair state on coding task; specialist preferred." in decision.reason


def test_model_router_escalation_selects_primary(router):
    """Verify ModelRouter selects PRIMARY during ESCALATION even on coding tasks."""
    req = RoutingRequest(
        task_description="Persistent crash in error.py",
        complexity=ComplexityLevel.COMPLEX,
        coding=True,
        context_size=ContextSize.NORMAL,
        task_state=TaskRoutingState.ESCALATION,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.PRIMARY
    assert decision.selected_model.model_id == "qwen3-8b"
    assert "Escalation state overrides coding specialist preference" in decision.reason


@pytest.mark.anyio
async def test_repair_runtime_reentry_workflow(runtime):
    """Simulate tool failure triggering autonomous REPAIR re-entry via AgentRuntime."""
    with patch.object(runtime.router, "route", wraps=runtime.router.route) as spy_route, \
         patch.object(runtime.lifecycle_manager, "ensure_loaded", wraps=runtime.lifecycle_manager.ensure_loaded) as spy_load:

        mock_react_agent = MagicMock()
        mock_react_agent.engine = MagicMock()
        mock_react_agent.run = AsyncMock(return_value=ChatMessage(
            role=Role.ASSISTANT,
            content="I am fixing the error by replacing print(1/0) with print(1).",
            pending_action=PendingAction(
                action_type="write_file",
                target="error.py",
                content="print(1)",
            ),
        ))

        # Initial failed execution on a plan
        task = TaskState(
            goal="Create script error.py, execute, fix when it crashes",
            task_type=TaskType.SOFTWARE_ENGINEERING,
        )
        task.record_failure("Step 2 (run_command): FAILED after 3 attempts. Last error: ZeroDivisionError")
        task.transition_to_repair()

        run_res: RunResult = await runtime.execute(
            mock_react_agent,
            task.goal,
            [],
            task=task,
        )

        # 1. Router must be consulted
        spy_route.assert_called_once()
        routing_request = spy_route.call_args[0][0]
        assert routing_request.task_state == TaskRoutingState.REPAIR
        assert routing_request.coding is True

        # 2. Lifecycle manager must load CODING model
        spy_load.assert_called_once()
        loaded_model = spy_load.call_args[0][0]
        assert loaded_model.role == ModelRole.CODING
        assert loaded_model.model_id == "qwen2.5-coder-7b-instruct"

        # 3. Agent must return next repair action
        assert run_res.message.pending_action is not None
        assert run_res.message.pending_action.action_type == "write_file"
        assert run_res.message.pending_action.target == "error.py"
