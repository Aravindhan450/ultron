from ultron.core.config import settings
from ultron.core.engine.base import BaseEngine
from ultron.core.engine.ollama import OllamaEngine

def get_engine(name: str = None) -> BaseEngine:
    """
    Factory function to get the appropriate LLM engine backend.
    
    Currently supports and defaults to the Ollama backend.
    """
    model_name = name or settings.model
    # Since Ollama is the only implemented engine for now, we default to it.
    return OllamaEngine(default_model=model_name)
