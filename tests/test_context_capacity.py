import pytest
from ultron.core.intelligence.model_catalog import get_default_catalog
from ultron.core.agents.react import build_system_prompt

def test_context_capacity_sufficient_for_react_prompt():
    catalog = get_default_catalog()
    prompt = build_system_prompt()
    
    # A rough upper bound for tokens is len(prompt) / 3 or 4.
    # The prompt is ~25000 characters. 25000 / 3 = ~8300 tokens.
    # Thus, context length must be at least 12000.
    
    for spec in catalog.list_models():
        assert spec.recommended_context_length >= 16384, (
            f"Model {spec.model_id} has insufficient context length "
            f"({spec.recommended_context_length}) to accommodate the ReAct prompt."
        )

