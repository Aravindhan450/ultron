import asyncio

from ultron.core.agents.simple import SimpleAgent, handle_debug

# Scenario 1: a real failing command, diagnosed with environmental state.
msg = handle_debug(command="python -c \"import numpy\" 2>/dev/null; python -c 'import pandas'")
print("=== Scenario 1: command run + diagnosis ===")
print(msg.content[:1800])
print()

# Scenario 2: pasted error text (no execution) with an expectation.
msg2 = handle_debug(
    error=(
        "Traceback (most recent call last):\n"
        "  File \"app.py\", line 4, in <module>\n"
        "    import pandas\n"
        "ModuleNotFoundError: No module named 'pandas'\n"
    ),
    expected="app runs and prints 5 rows",
)
print("=== Scenario 2: pasted traceback ===")
print(msg2.content[:1500])
print()

# Scenario 3: agent dispatch for "why is my script failing".
agent = SimpleAgent(engine=None)
msg3 = asyncio.run(agent.run("why is my test failing"))
print("=== Scenario 3: agent dispatch ===")
print(msg3.content[:600])
