with open("src/ultron/main.py", "r") as f:
    content = f.read()

content = content.replace(
    "async def async_chat(agent_type: str = \"simple\"):",
    "async def async_chat(agent_type: str = \"simple\", no_server: bool = False):"
)
content = content.replace(
    "lifecycle_manager = ModelLifecycleManager()",
    "lifecycle_manager = ModelLifecycleManager(no_server=no_server)"
)
content = content.replace(
    "await async_chat(agent_type=agent)",
    "await async_chat(agent_type=agent, no_server=no_server)"
)

with open("src/ultron/main.py", "w") as f:
    f.write(content)
