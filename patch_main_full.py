with open("src/ultron/main.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_async_chat = False
while_idx = -1
for i, line in enumerate(lines):
    if line.startswith("async def async_chat("):
        line = line.replace('agent_type: str = "simple"', 'agent_type: str = "simple", no_server: bool = False')
    if "lifecycle_manager = ModelLifecycleManager()" in line:
        line = line.replace("ModelLifecycleManager()", "ModelLifecycleManager(no_server=no_server)")
    
    if line.strip() == "while True:" and lines[i-2].strip() == "reflow.start()":
        new_lines.append("    try:\n")
        new_lines.append("        while True:\n")
        while_idx = i
        continue
    
    if while_idx != -1:
        if line.startswith("@app.command()"):
            while new_lines[-1].strip() == "":
                new_lines.pop()
            
            if "lifecycle_manager.shutdown()" in new_lines[-1]:
                new_lines.pop()
                if "Phase 3.5: Cleanup" in new_lines[-1]:
                    new_lines.pop()
            
            new_lines.append("\n    finally:\n        # Phase 3.5: Cleanup dynamically loaded models\n        lifecycle_manager.shutdown()\n\n")
            while_idx = -1
        else:
            if line.strip():
                new_lines.append("    " + line)
            else:
                new_lines.append("\n")
            continue

    if line.startswith("    async def _run_chat_session():"):
        new_lines.append("    import os\n")
        new_lines.append('    env_no_server = os.environ.get("ULTRON_NO_SERVER", "").lower() in ("1", "true", "yes")\n')
        new_lines.append("    effective_no_server = no_server or env_no_server\n\n")
        new_lines.append(line)
        continue
    
    if "await async_chat(agent_type=agent)" in line:
        line = line.replace("await async_chat(agent_type=agent)", "await async_chat(agent_type=agent, no_server=effective_no_server)")

    new_lines.append(line)

with open("src/ultron/main.py", "w") as f:
    f.writelines(new_lines)
