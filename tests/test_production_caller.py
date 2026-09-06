import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.anyio
async def test_production_confirmation_path_passes_runtime():
    """
    Objective A: Prove that the production caller in async_chat passes the runtime
    to continue_task_after_confirmation.
    """
    from ultron.main import async_chat
    from ultron.core.types import ChatMessage, Role, PendingAction, TaskState, TaskType

    # Mock all the interactive/CLI parts so async_chat can run headlessly
    with patch("prompt_toolkit.PromptSession") as mock_prompt_session_cls, \
         patch("questionary.select") as mock_questionary, \
         patch("ultron.main.execute_pending_action", new_callable=AsyncMock) as mock_execute, \
         patch("ultron.main.continue_task_after_confirmation", new_callable=AsyncMock) as mock_continue, \
         patch("ultron.core.agents.get_agent") as mock_get_agent, \
         patch("ultron.core.intelligence.model_lifecycle.ModelLifecycleManager") as mock_lifecycle, \
         patch("ultron.core.intelligence.model_router.ModelRouter") as mock_router:

        # 1. Provide an agent that returns a pending action
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        
        # We need a task state so it hits the continue_task_after_confirmation branch
        mock_task = TaskState(goal="test", task_type=TaskType.SOFTWARE_ENGINEERING)
        
        mock_msg = ChatMessage(
            role=Role.ASSISTANT, 
            content="I will do something",
            pending_action=PendingAction(action_type="run_command", target="echo hi"),
            task_state=mock_task
        )
        mock_agent.run.return_value = mock_msg
        
        mock_get_agent.return_value = mock_agent

        # 2. Mock PromptSession to return a message, then raise KeyboardInterrupt to exit loop
        mock_prompt_instance = mock_prompt_session_cls.return_value
        mock_prompt_instance.prompt_async = AsyncMock(side_effect=["do something", KeyboardInterrupt()])

        # 3. Mock questionary to automatically say "Yes, allow"
        mock_questionary.return_value.ask_async = AsyncMock(return_value="Yes, allow")
        
        mock_execute.return_value = "Action succeeded"
        mock_continue.return_value = ChatMessage(role=Role.ASSISTANT, content="done")

        # Run the chat loop!
        await async_chat(agent_type="simple", no_server=True)

        # 4. Verify that continue_task_after_confirmation was called with the runtime
        mock_continue.assert_awaited_once()
        
        # Check kwargs of the call
        call_kwargs = mock_continue.call_args.kwargs
        assert "runtime" in call_kwargs, "Production caller MUST pass runtime= parameter!"
        assert call_kwargs["runtime"] is not None, "Production caller MUST pass the actual AgentRuntime instance!"
        
        # Verify it's an AgentRuntime
        from ultron.core.runtime.runtime import AgentRuntime
        assert isinstance(call_kwargs["runtime"], AgentRuntime), "Passed runtime must be an AgentRuntime instance!"
