with open("src/ultron/main.py", "r") as f:
    content = f.read()

content = content.replace(
    "    session=None,\n) -> ChatMessage:",
    "    session=None,\n    runtime=None,\n) -> ChatMessage:"
)

content = content.replace(
    "    task.last_observation = result\n    return await agent.run(task.goal, history, task=task, session=session)",
    "    task.last_observation = result\n    if runtime:\n        run_res = await runtime.execute(agent, task.goal, history, task=task, session=session)\n        return run_res.message\n    return await agent.run(task.goal, history, task=task, session=session)"
)

content = content.replace(
    "                    response_msg = await continue_task_after_confirmation(\n                        agent, task, result, history, session=memory_session\n                    )",
    "                    response_msg = await continue_task_after_confirmation(\n                        agent, task, result, history, session=memory_session, runtime=runtime\n                    )"
)

with open("src/ultron/main.py", "w") as f:
    f.write(content)
