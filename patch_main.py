with open("src/ultron/main.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("async def async_chat("):
        lines[i] = line.replace('agent_type: str = "simple"', 'agent_type: str = "simple", no_server: bool = False')
    if "lifecycle_manager = ModelLifecycleManager()" in line:
        lines[i] = line.replace("ModelLifecycleManager()", "ModelLifecycleManager(no_server=no_server)")
    if "await async_chat(agent_type=agent)" in line:
        lines[i] = line.replace("await async_chat(agent_type=agent)", "await async_chat(agent_type=agent, no_server=effective_no_server)")
    
    if "async def _run_chat_session():" in line:
        # We need to insert effective_no_server before this
        # Find where it is
        pass

# Instead of complex search/replace, let's just use simple replaces on the whole string where possible
with open("src/ultron/main.py", "r") as f:
    content = f.read()

content = content.replace("async def async_chat(agent_type: str = \"simple\"):", "async def async_chat(agent_type: str = \"simple\", no_server: bool = False):")
content = content.replace("lifecycle_manager = ModelLifecycleManager()", "lifecycle_manager = ModelLifecycleManager(no_server=no_server)")

# The try/finally around while True:
old_while = "    reflow.start()\n\n    while True:\n        try:"
new_while = "    reflow.start()\n\n    try:\n        while True:\n            try:"
content = content.replace(old_while, new_while)

# Indent everything inside while True until the end of async_chat
# Since this is tricky, I'll just write a script that processes lines inside the block
