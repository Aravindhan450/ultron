import json
from typing import Any, AsyncIterator
import httpx
from ultron.core.engine.base import BaseEngine
from ultron.core.logging import get_logger

logger = get_logger("ultron.engine.ollama")

class OllamaEngine(BaseEngine):
    """
    Concrete implementation of BaseEngine using the Ollama local HTTP API.
    """

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3") -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """
        Generate a complete response using Ollama's chat API.
        """
        model = kwargs.pop("model", self.default_model)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return str(data["message"]["content"])
            except httpx.HTTPError as e:
                logger.error(f"Ollama API request failed: {e}")
                raise

    async def stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        """
        Stream chat response chunks from Ollama's API.
        """
        model = kwargs.pop("model", self.default_model)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    async for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("message", {}).get("content", "")
                            if chunk:
                                yield chunk
            except httpx.HTTPError as e:
                logger.error(f"Ollama streaming API request failed: {e}")
                raise
