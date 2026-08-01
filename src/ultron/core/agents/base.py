from abc import ABC, abstractmethod
from ultron.core.engine.base import BaseEngine
from ultron.core.types import ChatMessage

class BaseAgent(ABC):
    """
    Abstract base class for all Ultron agents, coordinating interaction with engines.
    """

    def __init__(self, engine: BaseEngine) -> None:
        self.engine = engine

    @abstractmethod
    async def run(self, user_input: str, history: list[ChatMessage] | None = None) -> ChatMessage:
        """
        Execute the agent logic given user input and conversation history.
        """
        pass
