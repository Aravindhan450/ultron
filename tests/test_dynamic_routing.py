import pytest
from unittest.mock import MagicMock, AsyncMock
from ultron.core.runtime.runtime import AgentRuntime
from ultron.core.intelligence.model_router import ModelRouter, RoutingDecision
from ultron.core.intelligence.model_catalog import get_default_catalog, ModelRole
from ultron.core.intelligence.model_lifecycle import ModelLifecycleManager, ModelHandle
from ultron.core.types import TaskState, TaskType, TaskPlan
from ultron.core.agents.base import BaseAgent
from ultron.core.agents.react import ReActAgent
from ultron.security.boundary import SecurityBoundary

class DummyEngine:
    def __init__(self):
        self.base_url = "http://dummy"
    def set_model(self, model):
        pass

from ultron.core.types import ChatMessage, Role
class DummyAgent(BaseAgent):
    def __init__(self):
        super().__init__(engine=DummyEngine())
    async def run(self, user_input, history=None, **kwargs):
        return ChatMessage(role=Role.ASSISTANT, content="mocked response")

@pytest.fixture
def catalog():
    return get_default_catalog()

@pytest.fixture
def router(catalog):
    return ModelRouter(catalog)

@pytest.fixture
def lifecycle_manager():
    from ultron.core.intelligence.model_lifecycle import LifecycleState
    manager = ModelLifecycleManager()
    # Mock ensure_loaded so we don't actually spawn processes
    manager.ensure_loaded = MagicMock(return_value=ModelHandle(
        model_spec=get_default_catalog().get_fast(),
        endpoint_url="http://mock:8080",
        state=LifecycleState.LOADED
    ))
    return manager

@pytest.fixture
def runtime(router, lifecycle_manager):
    return AgentRuntime(router=router, lifecycle_manager=lifecycle_manager)

@pytest.mark.anyio
async def test_a_simple_task_routing(runtime):
    agent = DummyAgent()
    await runtime.execute(agent, "hello")
    # No task -> Simple -> FAST
    called_model = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
    assert called_model.role == ModelRole.FAST
    assert agent.engine.base_url == "http://mock:8080"

@pytest.mark.anyio
async def test_b_coding_task_routing(runtime):
    agent = DummyAgent()
    task = TaskState(goal="write python", task_type=TaskType.SOFTWARE_ENGINEERING)
    await runtime.execute(agent, "write python", task=task)
    called_model = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
    assert called_model.role == ModelRole.CODING

@pytest.mark.anyio
async def test_c_complex_task_routing(runtime):
    from ultron.core.types import PlanStep
    agent = DummyAgent()
    plan = TaskPlan(
        goal="do lots of things",
        task_type=TaskType.MULTI_STEP,
        steps=[PlanStep(id=i, description="step", expected_outcome="outcome") for i in range(8)]
    )
    task = TaskState(goal="do lots of things", plan=plan)
    await runtime.execute(agent, "do lots of things", task=task)
    called_model = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
    assert called_model.role == ModelRole.PRIMARY

@pytest.mark.anyio
async def test_d_lifecycle_invocation(runtime):
    agent = DummyAgent()
    await runtime.execute(agent, "hello")
    runtime.lifecycle_manager.ensure_loaded.assert_called_once()

@pytest.mark.anyio
async def test_e_model_switch(runtime):
    agent = DummyAgent()
    # First task: simple
    await runtime.execute(agent, "hello")
    model_1 = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
    assert model_1.role == ModelRole.FAST
    
    # Second task: coding
    task2 = TaskState(goal="code", task_type=TaskType.SOFTWARE_ENGINEERING)
    await runtime.execute(agent, "code", task=task2)
    model_2 = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
    assert model_2.role == ModelRole.CODING

@pytest.mark.anyio
async def test_f_routing_authority(catalog, lifecycle_manager):
    from ultron.core.intelligence.model_router import ConfidenceLevel, RoutingRequest
    router = ModelRouter(catalog)
    router.route = MagicMock(return_value=RoutingDecision(
        selected_model=catalog.get_coding(),
        reason="Mocked",
        confidence=ConfidenceLevel.HIGH,
        fallback_model=catalog.get_primary(),
        signals={}
    ))
    runtime = AgentRuntime(router=router, lifecycle_manager=lifecycle_manager)
    agent = DummyAgent()
    await runtime.execute(agent, "hello")
    called_model = runtime.lifecycle_manager.ensure_loaded.call_args[0][0]
    assert called_model.role == ModelRole.CODING

@pytest.mark.anyio
async def test_g_react_compatibility(runtime):
    engine = DummyEngine()
    agent = ReActAgent(engine=engine)
    # Prevent actual LLM call
    agent.run = AsyncMock(return_value=ChatMessage(role=Role.ASSISTANT, content="hi"))
    await runtime.execute(agent, "hello")
    assert agent.engine.base_url == "http://mock:8080"
