from ultron.core.agents.base import BaseAgent
from ultron.core.agents.simple import SimpleAgent
from ultron.core.engine import get_engine

def get_agent(agent_type: str = "simple") -> BaseAgent:
    """
    Factory function to initialize and retrieve an agent instance.
    """
    engine = get_engine()
    if agent_type == "simple":
        return SimpleAgent(engine)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
