from typing import Any
from ultron.core.agents.base import BaseAgent

class SimpleAgent(BaseAgent):
    """
    A simple agent that passes user input and history directly to the engine.
    """

    async def run(self, user_input: str, history: list[dict[str, Any]] | None = None) -> str:
        """
        Runs the agent conversation step.
        """
        messages = []
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_input})
        return await self.engine.generate(messages)
