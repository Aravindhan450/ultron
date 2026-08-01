import asyncio
from ultron.core.engine import get_engine
from ultron.core.agents.simple import plan_task, detect_multistep_intent

async def main():
    # Test 1: the detector — should catch compound requests, ignore simple ones
    print("=== Testing detect_multistep_intent ===")
    print(detect_multistep_intent("create world.txt, write hello world in it, then read it back"))  # expect True
    print(detect_multistep_intent("read test.txt"))  # expect False
    print(detect_multistep_intent("remember that I use FastAPI"))  # expect False

    # Test 2: the planner — needs a real engine instance
    print("\n=== Testing plan_task ===")
    engine = get_engine()
    steps = await plan_task(
        "create a file named world.txt, write hello world in it, then read it back",
        engine
    )
    print(steps)

asyncio.run(main())