from abc import ABC, abstractmethod
from typing import Any
from ultron.core.engine.base import BaseEngine

class BaseAgent(ABC):
    """
    Abstract base class for all Ultron agents, coordinating interaction with engines.
    """

    def __init__(self, engine: BaseEngine) -> None:
        self.engine = engine

    @abstractmethod
    async def run(self, user_input: str, history: list[dict[str, Any]] | None = None) -> str:
        """
        Execute the agent logic given user input and conversation history.
        """
        pass
