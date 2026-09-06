"""ultron.core.intelligence.model_router
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model Router for Ultron's multi-model architecture.

Provides deterministic routing of tasks to the most appropriate registered
model based on complexity, coding requirements, task state, context size,
and memory pressure.

Architecture note:
    The router is a *pure decision component*. It evaluates signals and
    produces a `RoutingDecision`. It does NOT load models, unload models,
    or execute any agent logic.
"""
from __future__ import annotations

from enum import Enum, unique

from pydantic import BaseModel

from ultron.core.intelligence.model_catalog import (
    ModelCatalog,
    ModelRole,
    ModelSpec,
)


@unique
class ComplexityLevel(str, Enum):
    """Abstract classification of task complexity."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@unique
class ContextSize(str, Enum):
    """Abstract classification of required context length."""
    LIGHT = "light"
    NORMAL = "normal"
    HEAVY = "heavy"


@unique
class TaskRoutingState(str, Enum):
    """Lifecycle state of the task for routing decisions."""
    INITIAL = "initial"
    IN_PROGRESS = "in_progress"
    REPAIR = "repair"
    ESCALATION = "escalation"


@unique
class MemoryPressure(str, Enum):
    """Abstract classification of system memory constraints."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@unique
class ConfidenceLevel(str, Enum):
    """Qualitative confidence level of the routing decision."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RoutingRequest(BaseModel):
    """The incoming signals required to make a routing decision."""
    task_description: str
    complexity: ComplexityLevel
    coding: bool
    context_size: ContextSize
    task_state: TaskRoutingState
    memory_pressure: MemoryPressure


class RoutingDecision(BaseModel):
    """The deterministic output of the routing process."""
    selected_model: ModelSpec
    reason: str
    confidence: ConfidenceLevel
    fallback_model: ModelSpec
    signals: dict[str, str]


class ModelRouter:
    """Deterministic, typed router that selects models from a ModelCatalog."""

    def __init__(self, catalog: ModelCatalog) -> None:
        self.catalog = catalog

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Evaluate the request signals and return the optimal routing decision."""
        signals = {
            "complexity": request.complexity.value,
            "coding": str(request.coding),
            "context_size": request.context_size.value,
            "task_state": request.task_state.value,
            "memory_pressure": request.memory_pressure.value,
        }

        # Baseline fallback is always the primary model.
        try:
            fallback = self.catalog.get_primary()
        except KeyError as e:
            raise RuntimeError("Cannot route: Required PRIMARY fallback model is missing from catalog.") from e

        # -------------------------------------------------------------------
        # Explicit Routing Modes
        # -------------------------------------------------------------------
        if request.coding and request.task_state == TaskRoutingState.ESCALATION:
            return RoutingDecision(
                selected_model=fallback,
                reason="Escalation state overrides coding specialist preference; PRIMARY selected for deeper diagnosis.",
                confidence=ConfidenceLevel.HIGH,
                fallback_model=fallback,
                signals=signals,
            )

        # -------------------------------------------------------------------
        # Normal Routing Mode
        # -------------------------------------------------------------------
        candidates = self.catalog.list_models()
        if not candidates:
            raise RuntimeError("Cannot route: ModelCatalog is empty.")

        # Hard constraints filter
        viable_candidates = self._apply_hard_constraints(candidates, request)
        if not viable_candidates:
            raise ValueError("No registered model satisfies the hard capability constraints.")

        # Scoring
        scored_candidates = []
        for model in viable_candidates:
            score, reasons = self._score_model(model, request)
            scored_candidates.append((score, model, reasons))

        # Sort descending by score. In ties, Python's stable sort preserves original catalog order,
        # but to be deterministic regardless of order, we sort by score desc, then ID asc.
        scored_candidates.sort(key=lambda x: (-x[0], x[1].model_id))

        best_score, best_model, best_reasons = scored_candidates[0]
        
        # Determine confidence
        confidence = ConfidenceLevel.MEDIUM
        if best_score >= 10:
            confidence = ConfidenceLevel.HIGH
        elif best_score < 5:
            confidence = ConfidenceLevel.LOW

        reason_str = " | ".join(best_reasons)
        if not reason_str:
            reason_str = "Selected via default fallback scoring."

        return RoutingDecision(
            selected_model=best_model,
            reason=reason_str,
            confidence=confidence,
            fallback_model=fallback,
            signals=signals,
        )

    def _apply_hard_constraints(self, candidates: list[ModelSpec], request: RoutingRequest) -> list[ModelSpec]:
        """Filter out models that fundamentally cannot perform the task."""
        viable = []
        
        # If the task requires coding, we heavily prefer a model with CODING capability.
        has_coding_model = any(m.supports_coding for m in candidates)
        
        for m in candidates:
            if request.coding and has_coding_model and not m.supports_coding:
                continue
            viable.append(m)
            
        return viable

    def _score_model(self, model: ModelSpec, request: RoutingRequest) -> tuple[int, list[str]]:
        """
        Evaluate soft preferences and return a numeric score and human-readable reasons.
        Higher score is better.
        """
        score = 0
        reasons = []

        # 1. Coding Preference
        if request.coding:
            if model.supports_coding and model.role == ModelRole.CODING:
                score += 20
                reasons.append("Coding task; coding specialist has strongest capability match.")
            elif model.supports_coding:
                score += 10
                reasons.append("Coding task; model has coding capability.")
        else:
            # Non-coding tasks
            if model.role == ModelRole.CODING:
                score -= 10  # Avoid using coding specialist for non-coding tasks

        # 2. Complexity Preference
        if request.complexity == ComplexityLevel.SIMPLE:
            if model.role == ModelRole.FAST:
                score += 10
                reasons.append("Simple task; FAST model preferred for latency.")
        elif request.complexity == ComplexityLevel.COMPLEX:
            if model.role == ModelRole.PRIMARY:
                score += 10
                reasons.append("Complex task; PRIMARY model preferred for deep reasoning.")
        elif request.complexity == ComplexityLevel.MODERATE and model.role in (ModelRole.FAST, ModelRole.PRIMARY):
            score += 5
            reasons.append(f"Moderate task; {model.role.value} model is suitable.")

        # 3. Context Preference
        if request.context_size == ContextSize.HEAVY:
            if model.role == ModelRole.PRIMARY:
                score += 5
                reasons.append("Heavy context; PRIMARY model handles large context well.")
        elif request.context_size == ContextSize.LIGHT and model.role == ModelRole.FAST:
            score += 5
            reasons.append("Light context; FAST model avoids latency overhead.")

        # 4. Task State Preference
        if request.task_state == TaskRoutingState.REPAIR:
            if request.coding and model.role == ModelRole.CODING:
                score += 5
                reasons.append("Repair state on coding task; specialist preferred.")
            elif not request.coding and model.role == ModelRole.PRIMARY:
                score += 5
                reasons.append("Repair state; PRIMARY preferred for stability.")

        # 5. Memory Suitability
        if request.memory_pressure == MemoryPressure.HIGH:
            if model.role == ModelRole.FAST:
                score += 15
                reasons.append("High memory pressure; lightweight FAST model strongly preferred.")
            elif model.role == ModelRole.PRIMARY:
                score -= 10
                reasons.append("High memory pressure; PRIMARY model is risky.")

        return score, reasons
