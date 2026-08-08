import json
from collections.abc import AsyncIterator
from typing import Any

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

    @property
    def model(self) -> str:
        return self.default_model

    @model.setter
    def model(self, new_model: str) -> None:
        self.default_model = new_model

    def set_model(self, new_model: str) -> None:
        """Dynamically update the active model for LLM generation requests."""
        self.default_model = new_model

    async def supports_images(self, model: str | None = None) -> bool | None:
        """
        Returns whether the given model is multimodal (accepts image input).

        Queries Ollama's /api/show and checks the model's ``capabilities``.

        Returns None when the capability could not be determined (server
        unreachable, no capabilities field) — callers should treat that as
        "couldn't check" and let the real request surface any error, rather
        than showing the "no vision" hint for the wrong reason.
        """
        model = model or self.default_model
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/api/show",
                    json={"model": model},
                    timeout=10.0,
                )
                resp.raise_for_status()
                caps = resp.json().get("capabilities", [])
            except (httpx.HTTPError, OSError, ValueError):
                return None
        return "vision" in caps

    async def list_models(self) -> list[str]:
        """
        Fetch available local model names from Ollama.
        """
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
                if resp.status_code == 200:
                    tags = resp.json()
                    return [m["name"] for m in tags.get("models", [])]
            # ValueError covers JSONDecodeError from resp.json() on non-JSON bodies.
            except (httpx.HTTPError, ValueError) as e:
                logger.error(f"Failed to fetch model list from Ollama: {e}")
        return []

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
            # ValueError covers JSONDecodeError from resp.json() on non-JSON bodies.
            except (httpx.HTTPError, OSError, ValueError) as exc:
                logger.debug("Could not fetch available models for error hint: %s", exc)
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
