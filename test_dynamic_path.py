import asyncio
from ultron.core.runtime.runtime import AgentRuntime
from ultron.core.intelligence.model_catalog import get_default_catalog
from ultron.core.intelligence.model_router import ModelRouter
from ultron.core.intelligence.model_lifecycle import ModelLifecycleManager
from ultron.core.agents.simple import SimpleAgent
from ultron.core.engine.llama_cpp import LlamaCppEngine
from ultron.core.types import TaskState, TaskType, TaskPlan

async def run_real_test():
    catalog = get_default_catalog()
    router = ModelRouter(catalog)
    manager = ModelLifecycleManager()
    runtime = AgentRuntime(router=router, lifecycle_manager=manager)
    
    # We will use SimpleAgent for these tests so it just returns the response without tools
    engine = LlamaCppEngine(base_url="http://dummy")
    agent = SimpleAgent(engine=engine)
    
    try:
        # Test 1: General (FAST)
        # Note: Gemma Q8_K_M is missing in the previous report. 
        # But wait, if I run it, it will fail due to missing file.
        # Let's run it anyway, expect failure, or run Coder first.
        print("=== Test 1: Coding ===")
        task1 = TaskState(goal="Write a Python function that returns the sum of two integers. Return only the function.", task_type=TaskType.SOFTWARE_ENGINEERING)
        res1 = await runtime.execute(agent, "Write a Python function that returns the sum of two integers. Return only the function.", task=task1)
        print("Result 1:", res1.message.content if res1.message else res1.error)

        print("\n=== Test 2: Reasoning ===")
        plan = TaskPlan(goal="Reasoning", task_type=TaskType.MULTI_STEP, steps=[1]*8) # Complex
        task2 = TaskState(goal="A train travels 60 km in 45 minutes. What is its average speed in km/h? Return only the number.", plan=plan)
        res2 = await runtime.execute(agent, "A train travels 60 km in 45 minutes. What is its average speed in km/h? Return only the number.", task=task2)
        print("Result 2:", res2.message.content if res2.message else res2.error)

        print("\n=== Test 3: General ===")
        # Expect this to fail due to missing Q8_K_M
        res3 = await runtime.execute(agent, "Respond with exactly: ULTRON_DYNAMIC_ROUTING_OK")
        print("Result 3:", res3.message.content if res3.message else res3.error)
        
    finally:
        manager.shutdown()

if __name__ == "__main__":
    asyncio.run(run_real_test())
