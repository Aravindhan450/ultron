"""
MITLRunner: Autonomous execution runner and trace recorder for Model-in-the-Loop validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tests.model_in_loop.harness.sandbox import MITLSandbox
from ultron.core.agents.react import ReActAgent
from ultron.core.coding.context import CodeContext
from ultron.core.context import ContextBudgetConfig, RepositoryContextManager
from ultron.core.engine.base import BaseEngine
from ultron.core.memory import MemoryProvider
from ultron.core.runtime import AgentRuntime, RuntimeBudget
from ultron.core.types import ChatMessage, TaskState, TaskType
from ultron.main import (
    continue_task_after_confirmation,
    execute_pending_action,
)


@dataclass
class ToolInvocationRecord:
    iteration: int
    tool_name: str
    target: str
    arguments: dict[str, Any]
    success: bool
    result_preview: str


@dataclass
class ExecutionTrace:
    scenario_name: str
    prompt: str
    model_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    iterations: int = 0
    tools_invoked: list[ToolInvocationRecord] = field(default_factory=list)
    files_inspected: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tests_executed: list[str] = field(default_factory=list)
    final_response: str = ""
    is_completed: bool = False
    error: str | None = None

    def summary(self) -> str:
        tool_names = [t.tool_name for t in self.tools_invoked]
        return (
            f"Trace(scenario={self.scenario_name}, duration={self.duration:.2f}s, "
            f"iterations={self.iterations}, tools={tool_names}, "
            f"files_modified={self.files_modified}, completed={self.is_completed})"
        )


class MITLRunner:
    """
    Executes an autonomous LLM coding task through the real Ultron ReActAgent,
    AgentRuntime, and CodingExecutor pipeline against a sandbox.
    """

    def __init__(
        self,
        engine: BaseEngine,
        max_iterations: int = 15,
        timeout: float = 180.0,
    ) -> None:
        self.engine = engine
        self.max_iterations = max_iterations
        self.timeout = timeout

    async def run(
        self,
        scenario: Any,
        sandbox: MITLSandbox,
    ) -> ExecutionTrace:
        """
        Runs the model autonomously on the scenario and returns the detailed execution trace.
        """
        trace = ExecutionTrace(
            scenario_name=getattr(scenario, "name", "mitl_scenario"),
            prompt=scenario.prompt,
            model_name=getattr(self.engine, "model", "llama-cpp"),
            start_time=time.monotonic(),
        )

        provider = MemoryProvider()
        cm = RepositoryContextManager(
            workspace=sandbox.workspace,
            budget=ContextBudgetConfig(max_total_tokens=4000),
            memory_provider=provider,
        )
        runtime = AgentRuntime(
            context_manager=cm,
            default_budget=RuntimeBudget(max_iterations=self.max_iterations, timeout_seconds=self.timeout),
        )

        agent = ReActAgent(engine=self.engine, max_iterations=self.max_iterations)
        task = TaskState(goal=scenario.prompt, task_type=TaskType.DEBUGGING)
        code_ctx = CodeContext(workspace=sandbox.workspace)
        code_ctx.attach_task(task)
        task.code_context = code_ctx

        history: list[ChatMessage] = []

        try:
            # 1. Start agent turn via AgentRuntime
            run_result = await runtime.execute(
                agent=agent,
                user_input=scenario.prompt,
                history=history,
                task=task,
            )

            current_msg = run_result.message
            trace.iterations += 1

            deadline = time.monotonic() + self.timeout

            # 2. Autonomous Action -> Confirmation -> Continue loop
            while time.monotonic() < deadline and trace.iterations <= self.max_iterations:
                if current_msg is None or current_msg.pending_action is None:
                    # Agent completed turn or provided final natural language response
                    trace.is_completed = True
                    trace.final_response = current_msg.content if current_msg else ""
                    break

                action = current_msg.pending_action
                tool_name = action.action_type
                target = action.target or ""

                # Track file inspection vs modifications vs test runs
                if tool_name in ("read_file", "view_file", "find_files", "file_search"):
                    if target and target not in trace.files_inspected:
                        trace.files_inspected.append(target)
                elif tool_name in ("write_file", "create_file", "replace_file_content", "replace_in_file", "apply_edit"):
                    if target and target not in trace.files_modified:
                        trace.files_modified.append(target)
                elif tool_name == "run_command":
                    cmd = action.content or target
                    trace.tests_executed.append(cmd)

                # Execute the confirmed action inside the sandbox
                action_result = await execute_pending_action(action)
                success = not str(action_result).startswith(("Error", "Blocked by security"))

                trace.tools_invoked.append(
                    ToolInvocationRecord(
                        iteration=trace.iterations,
                        tool_name=tool_name,
                        target=target,
                        arguments={"content": (action.content or "")[:200]},
                        success=success,
                        result_preview=str(action_result)[:200],
                    )
                )

                # Continue agent execution with observation
                trace.iterations += 1
                current_msg = await continue_task_after_confirmation(
                    agent=agent,
                    task=task,
                    result=action_result,
                    history=history,
                )

            if current_msg and current_msg.content and not trace.final_response:
                trace.final_response = current_msg.content

        except Exception as exc:  # noqa: BLE001
            trace.error = str(exc)

        finally:
            trace.end_time = time.monotonic()
            trace.duration = trace.end_time - trace.start_time
            # Sync any internal tool executions from task history
            if task and task.execution_history:
                for ex in task.execution_history:
                    t_name = ex.tool_name
                    t_target = ex.target or ""
                    if t_name in ("read_file", "view_file", "find_files", "file_search", "list_directory"):
                        if t_target and t_target not in trace.files_inspected:
                            trace.files_inspected.append(t_target)
                    elif t_name in ("write_file", "create_file", "replace_file_content", "replace_in_file", "apply_edit"):
                        if t_target and t_target not in trace.files_modified:
                            trace.files_modified.append(t_target)
                    elif t_name == "run_command":
                        cmd = ex.target or ex.detail
                        if cmd not in trace.tests_executed:
                            trace.tests_executed.append(cmd)

                    # Ensure in tools_invoked
                    if not any(t.tool_name == t_name and t.target == t_target for t in trace.tools_invoked):
                        trace.tools_invoked.append(
                            ToolInvocationRecord(
                                iteration=len(trace.tools_invoked) + 1,
                                tool_name=t_name,
                                target=t_target,
                                arguments={},
                                success=ex.success,
                                result_preview=ex.detail[:200] if ex.detail else "",
                            )
                        )

            # Update files modified from sandbox actual status
            actual_modified = sandbox.get_modified_files()
            for f in actual_modified:
                if f not in trace.files_modified:
                    trace.files_modified.append(f)

        return trace
