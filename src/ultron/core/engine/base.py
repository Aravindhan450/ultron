from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

class BaseEngine(ABC):
    """
    Abstract base class defining the interface for all Ultron LLM backends.
    """

    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """
        Generate a complete text response for a given list of chat messages.
        """
        pass

    @abstractmethod
    async def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        """
        Stream text response chunks for a given list of chat messages.
        """
        pass
