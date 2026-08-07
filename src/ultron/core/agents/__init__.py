from ultron.core.agents.base import BaseAgent
from ultron.core.agents.react import ReActAgent
from ultron.core.agents.simple import SimpleAgent
from ultron.core.engine import get_engine

# Single source of truth for which agent types exist. The CLI (/agent command)
# and any other callers should derive their options from this.
SUPPORTED_AGENTS: tuple[str, ...] = ("simple", "react")


def get_agent(agent_type: str = "simple", engine=None) -> BaseAgent:
    """
    Factory function to initialize and retrieve an agent instance.

    Supported agent types (see ``SUPPORTED_AGENTS``):
      - "simple": deterministic detectors + single-shot LLM fallback
      - "react":  ReAct (Reason + Act) tool-use loop

    Args:
        agent_type: Which agent class to build.
        engine: Optional engine to reuse — preserves live model/backend state
            (e.g. a model chosen via /model). Defaults to a fresh engine from
            the active settings.
    """
    engine = engine or get_engine()
    if agent_type == "simple":
        return SimpleAgent(engine)
    if agent_type == "react":
        return ReActAgent(engine)
    raise ValueError(
        f"Unsupported agent type: {agent_type}. Available: {', '.join(SUPPORTED_AGENTS)}"
    )
