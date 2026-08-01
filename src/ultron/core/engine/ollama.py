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

    async def _handle_http_error(self, e: httpx.HTTPStatusError, model: str) -> None:
        """
        Helper method to provide context-rich error messages for HTTP status failures.
        """
        if e.response.status_code == 404:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                    if resp.status_code == 200:
                        tags = resp.json()
                        models = [m["name"] for m in tags.get("models", [])]
                        logger.error(
                            f"[bold red]Model '{model}' was not found in your local Ollama instance.[/bold red]\n"
                            f"Available local models: {', '.join(models)}\n"
                            f"To resolve, pull the model (e.g. `ollama pull {model}`) or configure "
                            f"ULTRON_MODEL in your environment/.env to use an available model."
                        )
                        return
            except Exception:
                pass
        logger.error(f"Ollama API request failed: {e}")

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
            except httpx.HTTPStatusError as e:
                await self._handle_http_error(e, model)
                raise
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
            except httpx.HTTPStatusError as e:
                await self._handle_http_error(e, model)
                raise
            except httpx.HTTPError as e:
                logger.error(f"Ollama streaming API request failed: {e}")
                raise
