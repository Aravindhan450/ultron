import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from ultron.main import continue_task_after_confirmation
from ultron.core.runtime.runtime import AgentRuntime
from ultron.core.intelligence.model_router import ModelRouter, RoutingRequest, RoutingDecision
from ultron.core.intelligence.model_catalog import get_default_catalog, ModelRole
from ultron.core.intelligence.model_lifecycle import ModelLifecycleManager, ModelHandle, LifecycleState
from ultron.core.types import TaskState, TaskType, TaskError, TaskStatus, ChatMessage, Role

class DummyEngine:
    def __init__(self):
        self.base_url = "http://dummy"
    def set_model(self, model):
        pass

class DummyAgent:
    def __init__(self):
        self.engine = DummyEngine()
    async def run(self, *args, **kwargs):
        return ChatMessage(role=Role.ASSISTANT, content="mocked response")

@pytest.fixture
def test_setup():
    catalog = get_default_catalog()
    router = ModelRouter(catalog)
    manager = ModelLifecycleManager()
    
    manager.ensure_loaded = MagicMock(return_value=ModelHandle(
        model_spec=catalog.get_fast(),
        endpoint_url="http://mock:8080",
        state=LifecycleState.LOADED
    ))
    
    runtime = AgentRuntime(router=router, lifecycle_manager=manager)
    return runtime, catalog

@pytest.mark.anyio
async def test_real_repair_continuation_re_enters_routing(test_setup):
    runtime, catalog = test_setup
    
    # Spy on runtime.execute and router.route
    with patch.object(runtime, 'execute', wraps=runtime.execute) as spy_execute, \
         patch.object(runtime.router, 'route', wraps=runtime.router.route) as spy_route:
        
        agent = DummyAgent()
        
        # A task that has failed once, putting it in REPAIR state conceptually
        task = TaskState(goal="fix the bug", task_type=TaskType.SOFTWARE_ENGINEERING)
        task.errors.append(TaskError(message="Build failed"))
        task.status = TaskStatus.TASK_RUNNING
        
        # The confirmation is granted, so we call the real continuation function
        # passing the runtime exactly like main.py does
        result_msg = await continue_task_after_confirmation(
            agent=agent,
            task=task,
            result="Tool output: error fixed",
            history=[],
            session=None,
            runtime=runtime
        )
        
        # 1. Assert AgentRuntime was re-entered!
        spy_execute.assert_awaited_once()
        
        # 2. Assert the ModelRouter was consulted during this continuation!
        spy_route.assert_called_once()
        
        # 3. Assert the ModelRouter made the expected REPAIR decision (CODING role for software engineering)
        routing_request = spy_route.call_args[0][0]
        assert routing_request.coding is True
        
        # 4. Check the loaded model
        called_model = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
        assert called_model.role == ModelRole.CODING
        
        assert result_msg.content == "mocked response"

@pytest.mark.anyio
async def test_real_escalation_continuation_re_enters_routing(test_setup):
    runtime, catalog = test_setup
    
    # Spy on runtime.execute and router.route
    with patch.object(runtime, 'execute', wraps=runtime.execute) as spy_execute, \
         patch.object(runtime.router, 'route', wraps=runtime.router.route) as spy_route:
        
        agent = DummyAgent()
        
        # A task that has failed multiple times, putting it in ESCALATION state
        task = TaskState(goal="fix the bug", task_type=TaskType.SOFTWARE_ENGINEERING)
        task.errors.extend([
            TaskError(message="Build failed 1"),
            TaskError(message="Build failed 2"),
            TaskError(message="Build failed 3"),
        ])
        task.status = TaskStatus.TASK_RUNNING
        
        # The confirmation is granted, so we call the real continuation function
        result_msg = await continue_task_after_confirmation(
            agent=agent,
            task=task,
            result="Tool output: error still not fixed",
            history=[],
            session=None,
            runtime=runtime
        )
        
        # 1. Assert AgentRuntime was re-entered!
        spy_execute.assert_awaited_once()
        
        # 2. Assert the ModelRouter was consulted during this continuation!
        spy_route.assert_called_once()
        
        # 3. Assert the routing request correctly identified Escalation
        routing_request = spy_route.call_args[0][0]
        from ultron.core.intelligence.model_router import TaskRoutingState
        assert routing_request.task_state == TaskRoutingState.ESCALATION
        
        # 4. Check the loaded model is PRIMARY because of Escalation rule
        called_model = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
        assert called_model.role == ModelRole.PRIMARY

