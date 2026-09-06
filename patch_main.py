import re

with open("src/ultron/main.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_async_chat = False
while_true_started = False

for line in lines:
    if line.startswith("    memory_session = SessionMemory()"):
        new_lines.append(line)
        new_lines.append("\n    # Dynamic Runtime Integration (Phase 3.5)\n")
        new_lines.append("    from ultron.core.intelligence.model_catalog import get_default_catalog\n")
        new_lines.append("    from ultron.core.intelligence.model_router import ModelRouter\n")
        new_lines.append("    from ultron.core.intelligence.model_lifecycle import ModelLifecycleManager\n\n")
        new_lines.append("    catalog = get_default_catalog()\n")
        new_lines.append("    model_router = ModelRouter(catalog)\n")
        new_lines.append("    lifecycle_manager = ModelLifecycleManager()\n")
        continue

    if line.strip() == "runtime = AgentRuntime()":
        new_lines.append("            runtime = AgentRuntime(router=model_router, lifecycle_manager=lifecycle_manager)\n")
        continue
        
    if line.startswith("    async def _run_chat_session():"):
        new_lines.append(line)
        new_lines.append("        await async_chat(agent_type=agent)\n\n")
        continue

    # Skip the old _run_chat_session body
    if line.strip() == "server_manager = LlamaServerManager()":
        continue
    if line.strip() == "server_started = False":
        continue
    if "if not skip_server and not server_manager.check_endpoint_occupied():" in line:
        continue
    if "UI.render_status(\"Starting local llama-server...\", status=\"info\")" in line:
        continue
    if "server_manager.start()" in line:
        continue
    if "server_started = True" in line:
        continue
    if "UI.render_status(\"llama-server is ready.\", status=\"success\")" in line:
        continue
    if "except Exception as exc:  # noqa: BLE001" in line:
        continue
    if "logger.error(\"Failed to start llama-server: %s\", exc)" in line:
        continue
    if "UI.render_error(str(exc), title=\"Server Startup Error\")" in line:
        continue
    if "return" in line and "UI.render_error" in new_lines[-1]: # tricky
        pass 
    if "await async_chat(agent_type=agent)" in line and "finally:" not in line:
        continue # handled above
    if line.strip() == "finally:":
        continue
    if line.strip() == "if server_started:":
        continue
    if line.strip() == "UI.render_status(\"Stopping local llama-server...\", status=\"info\")":
        continue
    if line.strip() == "server_manager.stop()":
        continue

    # Find the end of async_chat
    if line.startswith("@app.command()"):
        # Insert finally block before the next command
        new_lines.append("\n    # Phase 3.5: Shutdown lifecycle manager\n")
        new_lines.append("    lifecycle_manager.shutdown()\n\n")
    
    new_lines.append(line)

# Let's just write this to main_patched.py and compare
with open("src/ultron/main_patched.py", "w") as f:
    f.writelines(new_lines)
