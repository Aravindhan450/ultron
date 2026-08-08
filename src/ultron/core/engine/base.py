from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class BaseEngine(ABC):
    """
    Abstract base class defining the interface for all Ultron LLM backends.
    """

    @abstractmethod
    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """
        Generate a complete text response for a given list of chat messages.
        """

    @abstractmethod
    async def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        """
        Stream text response chunks for a given list of chat messages.
        """
