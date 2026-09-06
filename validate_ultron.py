import asyncio
from pathlib import Path
from ultron.core.intelligence.model_catalog import get_default_catalog
from ultron.core.intelligence.model_router import ModelRouter, RoutingRequest, ComplexityLevel, ContextSize, TaskRoutingState, MemoryPressure
from ultron.core.intelligence.model_lifecycle import ModelLifecycleManager, LifecycleState
from ultron.core.engine.llama_cpp import LlamaCppEngine

async def run_inference(endpoint_url, prompt):
    engine = LlamaCppEngine(base_url=endpoint_url)
    return await engine.generate([{"role": "user", "content": prompt}])

async def main():
    print("--- ULTRON VALIDATION SCRIPT ---")
    
    catalog = get_default_catalog()
    router = ModelRouter(catalog)
    manager = ModelLifecycleManager()
    
    print("\n--- TEST 1: REAL GEMMA LOAD ---")
    gemma_spec = catalog.get("gemma-3-4b-it")
    print(f"Loading {gemma_spec.model_id}...")
    try:
        handle1 = manager.ensure_loaded(gemma_spec)
        print(f"State: {handle1.state.value}, Endpoint: {handle1.endpoint_url}")
        response = await run_inference(handle1.endpoint_url, "Respond with exactly: ULTRON_GEMMA_OK")
        print(f"Inference response: {response}")
    except Exception as e:
        print(f"Gemma load failed: {e}")
    
    print("\n--- TEST 2: GEMMA RELEASE ---")
    manager.release(gemma_spec.model_id)
    print(f"Gemma state after release: {manager.get_status(gemma_spec.model_id).value}")
    
    print("\n--- TEST 3: REAL CODER LOAD ---")
    coder_spec = catalog.get("qwen2.5-coder-7b-instruct")
    print(f"Loading {coder_spec.model_id}...")
    try:
        handle2 = manager.ensure_loaded(coder_spec)
        print(f"State: {handle2.state.value}, Endpoint: {handle2.endpoint_url}")
        response = await run_inference(handle2.endpoint_url, "Write a Python function that returns the sum of two integers. Return only the function.")
        print(f"Inference response:\n{response}")
    except Exception as e:
        print(f"Coder load failed: {e}")
    
    print("\n--- TEST 4: CODER RELEASE ---")
    manager.release(coder_spec.model_id)
    print(f"Coder state after release: {manager.get_status(coder_spec.model_id).value}")

    print("\n--- TEST 5: REAL QWEN3 LOAD ---")
    qwen3_spec = catalog.get("qwen3-8b")
    print(f"Loading {qwen3_spec.model_id}...")
    try:
        handle3 = manager.ensure_loaded(qwen3_spec)
        print(f"State: {handle3.state.value}, Endpoint: {handle3.endpoint_url}")
        response = await run_inference(handle3.endpoint_url, "A train travels 60 km in 45 minutes. What is its average speed in km/h? Return only the number.")
        print(f"Inference response: {response}")
    except Exception as e:
        print(f"Qwen3 load failed: {e}")

    print("\n--- TEST 6: QWEN3 RELEASE ---")
    manager.release(qwen3_spec.model_id)
    print(f"Qwen3 state after release: {manager.get_status(qwen3_spec.model_id).value}")

    print("\n--- TEST 7: ROUTER + LIFECYCLE INTEGRATION ---")
    tests = [
        ("Simple general task", RoutingRequest(task_description="Say hello", complexity=ComplexityLevel.SIMPLE, coding=False, context_size=ContextSize.LIGHT, task_state=TaskRoutingState.INITIAL, memory_pressure=MemoryPressure.LOW)),
        ("Complex general task", RoutingRequest(task_description="Analyze architecture", complexity=ComplexityLevel.COMPLEX, coding=False, context_size=ContextSize.HEAVY, task_state=TaskRoutingState.INITIAL, memory_pressure=MemoryPressure.LOW)),
        ("Coding task", RoutingRequest(task_description="Write python", complexity=ComplexityLevel.SIMPLE, coding=True, context_size=ContextSize.LIGHT, task_state=TaskRoutingState.INITIAL, memory_pressure=MemoryPressure.LOW)),
        ("Coding repair", RoutingRequest(task_description="Fix syntax", complexity=ComplexityLevel.SIMPLE, coding=True, context_size=ContextSize.LIGHT, task_state=TaskRoutingState.REPAIR, memory_pressure=MemoryPressure.LOW)),
        ("Coding escalation", RoutingRequest(task_description="Fix deep bug", complexity=ComplexityLevel.COMPLEX, coding=True, context_size=ContextSize.HEAVY, task_state=TaskRoutingState.ESCALATION, memory_pressure=MemoryPressure.LOW)),
    ]
    
    for name, req in tests:
        decision = router.route(req)
        print(f"{name} -> Routed to: {decision.selected_model.model_id}")
        
    manager.shutdown()
    
if __name__ == "__main__":
    asyncio.run(main())
