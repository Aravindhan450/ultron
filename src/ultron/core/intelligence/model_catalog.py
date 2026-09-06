"""ultron.core.intelligence.model_catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Centralized model catalog for Ultron's multi-model architecture.

Provides strongly typed metadata, capability declarations, and role
assignments for each local GGUF model available to the system.  This module
is the **single source of truth** for model identity — future components
(``ModelRouter``, ``ModelLifecycleManager``, benchmarking) consume the
catalog rather than embedding model knowledge independently.

Architecture note:
    The catalog is a *static registry* — it describes which models exist and
    what they can do, but it does **not** load, unload, or route to models.
    Those responsibilities belong to future phases.
"""
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, unique
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Capability taxonomy
# ---------------------------------------------------------------------------

@unique
class ModelCapability(str, Enum):
    """Declared capability of a model.

    Each value represents an *intent* the model can serve.  The capability
    set is used by the future ``ModelRouter`` to match user tasks to models
    without hard-coding model names throughout the codebase.
    """

    GENERAL = "general"
    REASONING = "reasoning"
    LIGHTWEIGHT_REASONING = "lightweight_reasoning"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    AGENT = "agent"
    CODING = "coding"
    DEBUGGING = "debugging"
    REPOSITORY_ANALYSIS = "repository_analysis"
    CODE_GENERATION = "code_generation"
    STRUCTURED_OUTPUT = "structured_output"
    VISION = "vision"
    SUMMARIZATION = "summarization"


# ---------------------------------------------------------------------------
# Role taxonomy
# ---------------------------------------------------------------------------

@unique
class ModelRole(str, Enum):
    """Operational role a model fills in the Ultron fleet.

    Roles are *independent* of capabilities — a ``PRIMARY`` model need not
    support ``VISION``, and a ``CODING`` model may also declare ``GENERAL``.
    """

    PRIMARY = "primary"
    FAST = "fast"
    CODING = "coding"


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    """Immutable specification of a single GGUF model.

    All metadata required to identify, locate, and describe a model is
    captured here.  Fields marked with ``# future`` are reserved for later
    phases and are not populated by the initial catalog.
    """

    model_id: str = Field(description="Canonical short identifier, e.g. 'qwen3-8b'.")
    display_name: str = Field(description="Human-readable display name.")
    family: str = Field(description="Model family, e.g. 'qwen3', 'gemma-3'.")
    parameter_count: str = Field(description="Parameter count label, e.g. '8B', '4B'.")
    quantization: str = Field(description="Exact quantization variant, e.g. 'Q5_K_M'.")
    filename: str = Field(description="GGUF filename (not an absolute path).")
    role: ModelRole = Field(description="Operational role in the Ultron fleet.")
    capabilities: frozenset[ModelCapability] = Field(
        description="Declared capabilities of the model."
    )
    recommended_context_length: int = Field(
        default=8192,
        description="Conservative default context length for this hardware profile.",
    )

    model_config = {"frozen": True}

    # -- Convenience capability queries ------------------------------------

    @property
    def supports_vision(self) -> bool:
        return ModelCapability.VISION in self.capabilities

    @property
    def supports_tool_use(self) -> bool:
        return ModelCapability.TOOL_USE in self.capabilities

    @property
    def supports_structured_output(self) -> bool:
        return ModelCapability.STRUCTURED_OUTPUT in self.capabilities

    @property
    def supports_coding(self) -> bool:
        return ModelCapability.CODING in self.capabilities

    @property
    def supports_reasoning(self) -> bool:
        return (
            ModelCapability.REASONING in self.capabilities
            or ModelCapability.LIGHTWEIGHT_REASONING in self.capabilities
        )

    def has_capability(self, capability: ModelCapability) -> bool:
        """Return whether the model declares *capability*."""
        return capability in self.capabilities

    def resolve_path(self, models_dir: Path | None = None) -> Path:
        """Return the absolute path to the GGUF file.

        Uses *models_dir* if supplied, otherwise falls back to
        ``~/models`` as a conventional default.  The caller (e.g. the future
        ``ModelLifecycleManager``) is responsible for verifying the file
        exists on disk before attempting to load it.
        """
        base = models_dir or Path.home() / "models"
        return base / self.filename


# ---------------------------------------------------------------------------
# Catalog registry
# ---------------------------------------------------------------------------

class ModelCatalog:
    """Read-only registry of all models known to Ultron.

    The catalog is instantiated with the three canonical models.  It
    exposes deterministic lookup by ID, by role, and by capability.
    """

    def __init__(self, models: Sequence[ModelSpec] | None = None) -> None:
        entries = models if models is not None else _DEFAULT_MODELS
        self._by_id: dict[str, ModelSpec] = {}
        self._by_role: dict[ModelRole, ModelSpec] = {}
        for spec in entries:
            if spec.model_id in self._by_id:
                raise ValueError(f"Duplicate model_id in catalog: {spec.model_id!r}")
            self._by_id[spec.model_id] = spec
            if spec.role in self._by_role:
                raise ValueError(
                    f"Duplicate role {spec.role.value!r} in catalog: "
                    f"{spec.model_id!r} conflicts with "
                    f"{self._by_role[spec.role].model_id!r}"
                )
            self._by_role[spec.role] = spec

    # -- Lookup by ID ------------------------------------------------------

    def get(self, model_id: str) -> ModelSpec:
        """Return the spec for *model_id*, or raise ``KeyError``."""
        try:
            return self._by_id[model_id]
        except KeyError:
            available = ", ".join(sorted(self._by_id))
            raise KeyError(
                f"Unknown model_id {model_id!r}. Available: {available}"
            ) from None

    # -- Lookup by role ----------------------------------------------------

    def get_primary(self) -> ModelSpec:
        """Return the model assigned the ``PRIMARY`` role."""
        return self._by_role[ModelRole.PRIMARY]

    def get_fast(self) -> ModelSpec:
        """Return the model assigned the ``FAST`` role."""
        return self._by_role[ModelRole.FAST]

    def get_coding(self) -> ModelSpec:
        """Return the model assigned the ``CODING`` role."""
        return self._by_role[ModelRole.CODING]

    # -- Enumeration -------------------------------------------------------

    def list_models(self) -> list[ModelSpec]:
        """Return all registered models in registration order."""
        return list(self._by_id.values())

    def models_with_capability(self, capability: ModelCapability) -> list[ModelSpec]:
        """Return all models that declare *capability*."""
        return [m for m in self._by_id.values() if capability in m.capabilities]

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._by_id

    def __repr__(self) -> str:
        ids = ", ".join(sorted(self._by_id))
        return f"ModelCatalog([{ids}])"


# ---------------------------------------------------------------------------
# Default model definitions — single source of truth
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="qwen3-8b",
        display_name="Qwen3-8B",
        family="qwen3",
        parameter_count="8B",
        quantization="Q5_K_M",
        filename="Qwen3-8B-Q5_K_M.gguf",
        role=ModelRole.PRIMARY,
        capabilities=frozenset({
            ModelCapability.GENERAL,
            ModelCapability.REASONING,
            ModelCapability.PLANNING,
            ModelCapability.TOOL_USE,
            ModelCapability.AGENT,
            ModelCapability.STRUCTURED_OUTPUT,
        }),
        recommended_context_length=8192,
    ),
    ModelSpec(
        model_id="gemma-3-4b-it",
        display_name="Gemma 3 4B IT",
        family="gemma-3",
        parameter_count="4B",
        quantization="Q8_0",
        filename="gemma-3-4b-it-Q8_0.gguf",
        role=ModelRole.FAST,
        capabilities=frozenset({
            ModelCapability.GENERAL,
            ModelCapability.LIGHTWEIGHT_REASONING,
            ModelCapability.SUMMARIZATION,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.VISION,
        }),
        recommended_context_length=8192,
    ),
    ModelSpec(
        model_id="qwen2.5-coder-7b-instruct",
        display_name="Qwen2.5-Coder-7B-Instruct",
        family="qwen2.5-coder",
        parameter_count="7B",
        quantization="Q8_0",
        filename="qwen2.5-coder-7b-instruct-q8_0.gguf",
        role=ModelRole.CODING,
        capabilities=frozenset({
            ModelCapability.CODING,
            ModelCapability.DEBUGGING,
            ModelCapability.REPOSITORY_ANALYSIS,
            ModelCapability.CODE_GENERATION,
            ModelCapability.STRUCTURED_OUTPUT,
            ModelCapability.TOOL_USE,
        }),
        recommended_context_length=8192,
    ),
)


def get_default_catalog() -> ModelCatalog:
    """Return a fresh ``ModelCatalog`` populated with the canonical models."""
    return ModelCatalog()
