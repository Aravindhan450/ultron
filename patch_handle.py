with open("src/ultron/main.py", "r") as f:
    content = f.read()

content = content.replace(
    "                    handled, should_exit = await handle_slash_command(\n                        trimmed_input,\n                        console,\n                        history,\n                        agent,\n                        memory_session,\n                        state,\n                        reflow,\n                    )",
    "                    handled, should_exit = await handle_slash_command(\n                        trimmed_input,\n                        console,\n                        history,\n                        agent,\n                        memory_session,\n                        state,\n                        reflow,\n                        _runtime=runtime,\n                    )"
)

with open("src/ultron/main.py", "w") as f:
    f.write(content)
