import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ultron.core.engine.base import BaseEngine
from ultron.core.logging import get_logger

logger = get_logger("ultron.engine.llama_cpp")


class LlamaCppEngine(BaseEngine):
    """
    Concrete implementation of BaseEngine using the llama-server HTTP API
    (OpenAI-compatible endpoints: /v1/chat/completions and /v1/models).
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        default_model: str = "default",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    @property
    def model(self) -> str:
        return self.default_model

    @model.setter
    def model(self, new_model: str) -> None:
        self.default_model = new_model

    def set_model(self, new_model: str) -> None:
        """Dynamically update the active model identifier."""
        self.default_model = new_model

    async def supports_images(self, model: str | None = None) -> bool | None:
        """
        Returns whether the active server instance/model supports vision/images.
        Checks llama-server /props endpoint for multimodal / vision capabilities.
        Returns None when unreachable or undetermined.
        """
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.base_url}/props", timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    modalities = data.get("modalities", {})
                    if isinstance(modalities, dict) and "vision" in modalities:
                        return bool(modalities.get("vision", False))
                    # Fallback for older llama.cpp builds
                    return bool(
                        data.get("has_clip", False)
                        or data.get("has_mmproj", False)
                        or data.get("multimodal", False)
                    )
            except (httpx.HTTPError, OSError, ValueError):
                return None
        return None


    async def get_active_model(self) -> str | None:
        """
        Queries llama-server /v1/models to determine the actual model currently loaded in memory.
        Returns the model id/path if reachable, or None if server is unavailable.
        """
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.base_url}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [
                        m.get("id", m.get("name", ""))
                        for m in data.get("data", [])
                        if m.get("id") or m.get("name")
                    ]
                    if models:
                        return models[0]
            except (httpx.HTTPError, OSError, ValueError):
                return None
        return None

    async def list_models(self) -> list[str]:
        """
        Fetch available model IDs from llama-server (/v1/models).
        """
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.base_url}/v1/models", timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    return [
                        m.get("id", m.get("name", ""))
                        for m in data.get("data", [])
                        if m.get("id") or m.get("name")
                    ]
            except (httpx.HTTPError, ValueError) as e:
                logger.error(f"Failed to fetch model list from llama-server: {e}")
        return []


    async def _handle_http_error(self, e: httpx.HTTPStatusError, model: str) -> None:
        """
        Provide structured diagnostics for HTTP status errors.
        """
        status = e.response.status_code
        logger.error(f"llama-server HTTP error ({status}) for model '{model}': {e}")

    def _prepare_payload(
        self, messages: list[dict[str, Any]], stream: bool, **kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        model = kwargs.pop("model", self.default_model)

        # Standardize messages for OpenAI chat completions format
        formatted_messages = []
        for msg in messages:
            formatted_msg = dict(msg)
            # Handle images: convert agent image data to OpenAI image_url content parts
            if formatted_msg.get("images"):

                content_parts = [{"type": "text", "text": str(formatted_msg.get("content", ""))}]
                for img_b64 in formatted_msg["images"]:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    })
                formatted_msg["content"] = content_parts
                formatted_msg.pop("images", None)
            formatted_messages.append(formatted_msg)

        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": stream,
            **kwargs,
        }
        return model, payload

    async def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """
        Generate a complete response using llama-server's /v1/chat/completions endpoint.
        """
        model, payload = self._prepare_payload(messages, stream=False, **kwargs)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("llama-server returned response with empty choices")

                message = choices[0].get("message", {})
                content = message.get("content")
                if content is None:
                    raise ValueError("llama-server returned message with no content")
                return str(content)
            except httpx.HTTPStatusError as e:
                await self._handle_http_error(e, model)
                raise
            except httpx.HTTPError as e:
                logger.error(f"llama-server connection/request failed: {e}")
                raise

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Stream chat response chunks from llama-server's /v1/chat/completions SSE stream.
        """
        model, payload = self._prepare_payload(messages, stream=True, **kwargs)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    chunk = delta.get("content", "")
                                    if chunk:
                                        yield chunk
                            except json.JSONDecodeError:
                                continue
            except httpx.HTTPStatusError as e:
                await self._handle_http_error(e, model)
                raise
            except httpx.HTTPError as e:
                logger.error(f"llama-server streaming request failed: {e}")
                raise

