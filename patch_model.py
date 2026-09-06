import re

with open("src/ultron/main.py", "r") as f:
    content = f.read()

start_str = '    if clean_cmd == "/model" or clean_cmd.startswith("/model "):'
end_str = '    if clean_cmd == "/agent" or clean_cmd.startswith("/agent "):'

new_model_cmd = """    if clean_cmd == "/model" or clean_cmd.startswith("/model "):
        from ultron.core.config import settings
        
        console.print(f"[bold {ACCENT}]🧠 Dynamic Model Architecture (Phase 3.5):[/bold {ACCENT}]")
        
        if _runtime and hasattr(_runtime, 'lifecycle_manager'):
            lm = _runtime.lifecycle_manager
            active = lm._active_spec
            status = lm.get_status(active.model_id).value if active else "None"
            
            console.print(f"  [{MUTED}]Active Server Model:[/{MUTED}] [bold {TEXT}]{active.model_id if active else 'None'} ({status})[/bold {TEXT}]")
            console.print(f"  [{MUTED}]Backend:[/{MUTED}]             [bold {TEXT}]llama.cpp (llama-server)[/bold {TEXT}]")
            if not lm.no_server:
                console.print(f"  [{MUTED}]Dynamic Routing:[/{MUTED}]   [bold {GREEN}]ENABLED[/bold {GREEN}]")
            else:
                console.print(f"  [{MUTED}]Dynamic Routing:[/{MUTED}]   [bold {YELLOW}]NO_SERVER[/bold {YELLOW}] (Bypassing subprocess)")
        else:
            console.print(f"  [{MUTED}]Active Server Model:[/{MUTED}] [bold {TEXT}]Unknown[/bold {TEXT}]")

        parts = clean_cmd.split(maxsplit=1)
        if len(parts) > 1:
            requested_model = parts[1].strip()
            console.print(
                f"\\n[bold {YELLOW}]Notice:[/bold {YELLOW}] Manual model switching via /model <name> is disabled in Phase 3.5.\\n"
                f"Model selection is now deterministically managed by the [bold {TEXT}]ModelRouter[/bold {TEXT}].\\n"
            )
        else:
            console.print(
                f"\\n[{MUTED}]Note: Model switching is dynamically managed by the AgentRuntime based on task complexity.[/{MUTED}]\\n"
            )
        return True, False
"""

start_idx = content.find(start_str)
end_idx = content.find(end_str)
content = content[:start_idx] + new_model_cmd + "\n\n" + content[end_idx:]

with open("src/ultron/main.py", "w") as f:
    f.write(content)
