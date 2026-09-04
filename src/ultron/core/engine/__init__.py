from ultron.core.config import settings
from ultron.core.engine.base import BaseEngine
from ultron.core.engine.llama_cpp import LlamaCppEngine
from ultron.core.engine.server import LlamaServerManager

__all__ = ["BaseEngine", "LlamaCppEngine", "LlamaServerManager", "get_engine"]




def get_engine(name: str | None = None) -> BaseEngine:
    """
    Factory function to get the appropriate LLM engine backend.

    Defaults to the LlamaCppEngine backend.
    """
    model_name = name or settings.model
    return LlamaCppEngine(
        base_url=settings.llama_cpp_base_url,
        default_model=model_name,
        timeout=settings.timeout,
    )
