"""Tests for ultron.core.intelligence.model_router.

Validates the Phase 2 Model Router: deterministic scoring, hard constraints,
soft preferences, and edge cases.
"""
from __future__ import annotations

import pytest

from ultron.core.intelligence.model_catalog import ModelRole, get_default_catalog
from ultron.core.intelligence.model_router import (
    ComplexityLevel,
    ConfidenceLevel,
    ContextSize,
    MemoryPressure,
    ModelRouter,
    RoutingRequest,
    TaskRoutingState,
)


@pytest.fixture
def catalog():
    return get_default_catalog()


@pytest.fixture
def router(catalog):
    return ModelRouter(catalog)


def test_basic_simple_non_coding(router):
    req = RoutingRequest(
        task_description="What is the weather?",
        complexity=ComplexityLevel.SIMPLE,
        coding=False,
        context_size=ContextSize.LIGHT,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.FAST
    assert decision.selected_model.model_id == "gemma-3-4b-it"
    assert decision.confidence == ConfidenceLevel.HIGH


def test_basic_moderate_non_coding(router):
    req = RoutingRequest(
        task_description="Explain recursion.",
        complexity=ComplexityLevel.MODERATE,
        coding=False,
        context_size=ContextSize.NORMAL,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    # MODERATE -> FAST or PRIMARY. Both are valid. The router prefers FAST slightly via ties or score.
    # In my logic, MODERATE gives +5 to both FAST and PRIMARY. FAST avoids -10 CODING penalty.
    # Let's ensure it doesn't pick CODING.
    assert decision.selected_model.role in (ModelRole.FAST, ModelRole.PRIMARY)


def test_basic_complex_non_coding(router):
    req = RoutingRequest(
        task_description="Analyze this architecture diagram deeply.",
        complexity=ComplexityLevel.COMPLEX,
        coding=False,
        context_size=ContextSize.HEAVY,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.PRIMARY
    assert decision.selected_model.model_id == "qwen3-8b"


def test_coding_simple(router):
    req = RoutingRequest(
        task_description="Fix this typo.",
        complexity=ComplexityLevel.SIMPLE,
        coding=True,
        context_size=ContextSize.LIGHT,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.CODING
    assert decision.selected_model.model_id == "qwen2.5-coder-7b-instruct"


def test_coding_moderate(router):
    req = RoutingRequest(
        task_description="Refactor the class.",
        complexity=ComplexityLevel.MODERATE,
        coding=True,
        context_size=ContextSize.NORMAL,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.CODING


def test_coding_complex(router):
    req = RoutingRequest(
        task_description="Rewrite the subsystem.",
        complexity=ComplexityLevel.COMPLEX,
        coding=True,
        context_size=ContextSize.HEAVY,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.CODING


def test_coding_escalation(router):
    req = RoutingRequest(
        task_description="Fix recursion error.",
        complexity=ComplexityLevel.COMPLEX,
        coding=True,
        context_size=ContextSize.HEAVY,
        task_state=TaskRoutingState.ESCALATION,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    # Escalation on coding should prefer PRIMARY
    assert decision.selected_model.role == ModelRole.PRIMARY


def test_high_memory_pressure_simple(router):
    req = RoutingRequest(
        task_description="Summarize this text.",
        complexity=ComplexityLevel.SIMPLE,
        coding=False,
        context_size=ContextSize.LIGHT,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.HIGH,
    )
    decision = router.route(req)
    assert decision.selected_model.role == ModelRole.FAST
    assert "High memory pressure" in decision.reason


def test_determinism(router):
    req = RoutingRequest(
        task_description="Determine logic.",
        complexity=ComplexityLevel.MODERATE,
        coding=True,
        context_size=ContextSize.HEAVY,
        task_state=TaskRoutingState.REPAIR,
        memory_pressure=MemoryPressure.MEDIUM,
    )
    
    d1 = router.route(req)
    d2 = router.route(req)
    d3 = router.route(req)
    
    assert d1.selected_model.model_id == d2.selected_model.model_id == d3.selected_model.model_id
    assert d1.reason == d2.reason == d3.reason


def test_explainability(router):
    req = RoutingRequest(
        task_description="Tell me a joke.",
        complexity=ComplexityLevel.SIMPLE,
        coding=False,
        context_size=ContextSize.LIGHT,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    decision = router.route(req)
    assert len(decision.reason) > 0
    assert "Simple task" in decision.reason


def test_hard_constraint_filtering(catalog):
    # Create a catalog with ONLY non-coding models, and one that doesn't have CODING.
    # Actually, we can test that if we remove the Coder model, the router falls back to PRIMARY.
    # Wait, the prompt says if a capability is required but no registered model supports it, fail clearly.
    # BUT for coding, the logic says "if a coding-capable model exists".
    
    # Let's mock a catalog
    from ultron.core.intelligence.model_catalog import (
        ModelCapability,
        ModelCatalog,
        ModelSpec,
    )
    
    m1 = ModelSpec(
        model_id="no-code",
        display_name="No Code",
        family="none",
        parameter_count="1B",
        quantization="Q4",
        filename="nocode.gguf",
        role=ModelRole.PRIMARY,
        capabilities=frozenset({ModelCapability.GENERAL}),
    )
    cat = ModelCatalog([m1])
    r = ModelRouter(cat)
    
    req = RoutingRequest(
        task_description="Code something.",
        complexity=ComplexityLevel.SIMPLE,
        coding=True, # Needs coding
        context_size=ContextSize.LIGHT,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    
    # Since no model supports coding, the hard constraint allows fallback to non-coding
    d = r.route(req)
    assert d.selected_model.model_id == "no-code"

def test_missing_model_empty_catalog():
    from ultron.core.intelligence.model_catalog import ModelCatalog
    cat = ModelCatalog([])
    r = ModelRouter(cat)
    
    req = RoutingRequest(
        task_description="Fail.",
        complexity=ComplexityLevel.SIMPLE,
        coding=False,
        context_size=ContextSize.LIGHT,
        task_state=TaskRoutingState.INITIAL,
        memory_pressure=MemoryPressure.LOW,
    )
    with pytest.raises(RuntimeError, match="Cannot route: ModelCatalog is completely empty"):
        r.route(req)
