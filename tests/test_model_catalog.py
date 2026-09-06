"""Tests for ultron.core.intelligence.model_catalog.

Validates the Phase 1 Model Catalog: model registration, IDs, quantizations,
roles, capabilities, lookup, and path resolution.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ultron.core.intelligence.model_catalog import (
    ModelCapability,
    ModelCatalog,
    ModelRole,
    ModelSpec,
    get_default_catalog,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog() -> ModelCatalog:
    """Return the default catalog populated with the three canonical models."""
    return get_default_catalog()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    """Verify exactly the three selected models are registered."""

    def test_exactly_three_models(self, catalog: ModelCatalog) -> None:
        assert len(catalog) == 3

    def test_model_ids(self, catalog: ModelCatalog) -> None:
        ids = {m.model_id for m in catalog.list_models()}
        assert ids == {"qwen3-8b", "gemma-3-4b-it", "qwen2.5-coder-7b-instruct"}

    def test_contains(self, catalog: ModelCatalog) -> None:
        assert "qwen3-8b" in catalog
        assert "gemma-3-4b-it" in catalog
        assert "qwen2.5-coder-7b-instruct" in catalog
        assert "nonexistent" not in catalog


# ---------------------------------------------------------------------------
# Quantizations
# ---------------------------------------------------------------------------

class TestQuantizations:
    """Verify exact quantization labels."""

    def test_qwen3_quantization(self, catalog: ModelCatalog) -> None:
        assert catalog.get("qwen3-8b").quantization == "Q5_K_M"

    def test_gemma_quantization(self, catalog: ModelCatalog) -> None:
        assert catalog.get("gemma-3-4b-it").quantization == "Q8_0"

    def test_coder_quantization(self, catalog: ModelCatalog) -> None:
        assert catalog.get("qwen2.5-coder-7b-instruct").quantization == "Q8_0"


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class TestRoles:
    """Verify role assignments are correct and independent of capabilities."""

    def test_primary_role(self, catalog: ModelCatalog) -> None:
        assert catalog.get("qwen3-8b").role == ModelRole.PRIMARY

    def test_fast_role(self, catalog: ModelCatalog) -> None:
        assert catalog.get("gemma-3-4b-it").role == ModelRole.FAST

    def test_coding_role(self, catalog: ModelCatalog) -> None:
        assert catalog.get("qwen2.5-coder-7b-instruct").role == ModelRole.CODING

    def test_get_primary(self, catalog: ModelCatalog) -> None:
        primary = catalog.get_primary()
        assert primary.model_id == "qwen3-8b"
        assert primary.role == ModelRole.PRIMARY

    def test_get_fast(self, catalog: ModelCatalog) -> None:
        fast = catalog.get_fast()
        assert fast.model_id == "gemma-3-4b-it"
        assert fast.role == ModelRole.FAST

    def test_get_coding(self, catalog: ModelCatalog) -> None:
        coding = catalog.get_coding()
        assert coding.model_id == "qwen2.5-coder-7b-instruct"
        assert coding.role == ModelRole.CODING


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class TestCapabilities:
    """Verify capability declarations and query methods."""

    def test_qwen3_reasoning(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen3-8b")
        assert m.has_capability(ModelCapability.REASONING)
        assert m.supports_reasoning

    def test_qwen3_tool_use(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen3-8b")
        assert m.has_capability(ModelCapability.TOOL_USE)
        assert m.supports_tool_use

    def test_qwen3_no_vision(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen3-8b")
        assert not m.has_capability(ModelCapability.VISION)
        assert not m.supports_vision

    def test_qwen3_no_coding(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen3-8b")
        assert not m.supports_coding

    def test_gemma_general(self, catalog: ModelCatalog) -> None:
        m = catalog.get("gemma-3-4b-it")
        assert m.has_capability(ModelCapability.GENERAL)

    def test_gemma_vision(self, catalog: ModelCatalog) -> None:
        m = catalog.get("gemma-3-4b-it")
        assert m.has_capability(ModelCapability.VISION)
        assert m.supports_vision

    def test_gemma_lightweight_reasoning(self, catalog: ModelCatalog) -> None:
        m = catalog.get("gemma-3-4b-it")
        assert m.has_capability(ModelCapability.LIGHTWEIGHT_REASONING)
        assert m.supports_reasoning  # lightweight counts

    def test_gemma_no_coding(self, catalog: ModelCatalog) -> None:
        m = catalog.get("gemma-3-4b-it")
        assert not m.supports_coding

    def test_coder_coding(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen2.5-coder-7b-instruct")
        assert m.has_capability(ModelCapability.CODING)
        assert m.supports_coding

    def test_coder_debugging(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen2.5-coder-7b-instruct")
        assert m.has_capability(ModelCapability.DEBUGGING)

    def test_coder_tool_use(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen2.5-coder-7b-instruct")
        assert m.supports_tool_use

    def test_coder_no_vision(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen2.5-coder-7b-instruct")
        assert not m.supports_vision

    def test_models_with_capability_tool_use(self, catalog: ModelCatalog) -> None:
        models = catalog.models_with_capability(ModelCapability.TOOL_USE)
        ids = {m.model_id for m in models}
        assert ids == {"qwen3-8b", "qwen2.5-coder-7b-instruct"}

    def test_models_with_capability_vision(self, catalog: ModelCatalog) -> None:
        models = catalog.models_with_capability(ModelCapability.VISION)
        assert len(models) == 1
        assert models[0].model_id == "gemma-3-4b-it"

    def test_models_with_capability_structured_output(self, catalog: ModelCatalog) -> None:
        models = catalog.models_with_capability(ModelCapability.STRUCTURED_OUTPUT)
        assert len(models) == 3  # all three models

    def test_models_with_capability_coding(self, catalog: ModelCatalog) -> None:
        models = catalog.models_with_capability(ModelCapability.CODING)
        assert len(models) == 1
        assert models[0].model_id == "qwen2.5-coder-7b-instruct"


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

class TestLookup:
    """Verify get() behavior for valid and invalid IDs."""

    def test_valid_id(self, catalog: ModelCatalog) -> None:
        spec = catalog.get("qwen3-8b")
        assert spec.display_name == "Qwen3-8B"
        assert spec.family == "qwen3"
        assert spec.parameter_count == "8B"

    def test_invalid_id_raises_key_error(self, catalog: ModelCatalog) -> None:
        with pytest.raises(KeyError, match="Unknown model_id 'not-a-model'"):
            catalog.get("not-a-model")

    def test_invalid_id_lists_available(self, catalog: ModelCatalog) -> None:
        with pytest.raises(KeyError, match="qwen3-8b"):
            catalog.get("bad-id")


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

class TestModelMetadata:
    """Verify display names, families, parameter counts, and filenames."""

    def test_qwen3_metadata(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen3-8b")
        assert m.display_name == "Qwen3-8B"
        assert m.family == "qwen3"
        assert m.parameter_count == "8B"
        assert m.filename == "Qwen3-8B-Q5_K_M.gguf"

    def test_gemma_metadata(self, catalog: ModelCatalog) -> None:
        m = catalog.get("gemma-3-4b-it")
        assert m.display_name == "Gemma 3 4B IT"
        assert m.family == "gemma-3"
        assert m.parameter_count == "4B"
        assert m.filename == "gemma-3-4b-it-Q8_0.gguf"

    def test_coder_metadata(self, catalog: ModelCatalog) -> None:
        m = catalog.get("qwen2.5-coder-7b-instruct")
        assert m.display_name == "Qwen2.5-Coder-7B-Instruct"
        assert m.family == "qwen2.5-coder"
        assert m.parameter_count == "7B"
        assert m.filename == "qwen2.5-coder-7b-instruct-q8_0.gguf"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPathResolution:
    """Verify resolve_path uses models_dir correctly (no real GGUF required)."""

    def test_resolve_with_explicit_dir(self, catalog: ModelCatalog) -> None:
        spec = catalog.get("qwen3-8b")
        resolved = spec.resolve_path(models_dir=Path("/opt/models"))
        assert resolved == Path("/opt/models/Qwen3-8B-Q5_K_M.gguf")

    def test_resolve_default_home_models(self, catalog: ModelCatalog) -> None:
        spec = catalog.get("gemma-3-4b-it")
        resolved = spec.resolve_path()
        assert resolved == Path.home() / "models" / "gemma-3-4b-it-Q8_0.gguf"

    def test_resolve_with_tmp_dir(self, catalog: ModelCatalog, tmp_path: Path) -> None:
        spec = catalog.get("qwen2.5-coder-7b-instruct")
        resolved = spec.resolve_path(models_dir=tmp_path)
        assert resolved == tmp_path / "qwen2.5-coder-7b-instruct-q8_0.gguf"


# ---------------------------------------------------------------------------
# Context length
# ---------------------------------------------------------------------------

class TestContextLength:
    """Verify recommended context length defaults."""

    def test_default_context_length(self, catalog: ModelCatalog) -> None:
        for spec in catalog.list_models():
            assert spec.recommended_context_length == 8192


# ---------------------------------------------------------------------------
# Catalog construction edge cases
# ---------------------------------------------------------------------------

class TestCatalogConstruction:
    """Verify catalog construction validation."""

    def test_duplicate_model_id_rejected(self) -> None:
        dup = ModelSpec(
            model_id="qwen3-8b",
            display_name="Duplicate",
            family="test",
            parameter_count="1B",
            quantization="Q4_0",
            filename="dup.gguf",
            role=ModelRole.FAST,
            capabilities=frozenset(),
        )
        with pytest.raises(ValueError, match="Duplicate model_id"):
            ModelCatalog(models=[dup, dup])

    def test_duplicate_role_rejected(self) -> None:
        m1 = ModelSpec(
            model_id="a",
            display_name="A",
            family="test",
            parameter_count="1B",
            quantization="Q4_0",
            filename="a.gguf",
            role=ModelRole.PRIMARY,
            capabilities=frozenset(),
        )
        m2 = ModelSpec(
            model_id="b",
            display_name="B",
            family="test",
            parameter_count="1B",
            quantization="Q4_0",
            filename="b.gguf",
            role=ModelRole.PRIMARY,
            capabilities=frozenset(),
        )
        with pytest.raises(ValueError, match="Duplicate role"):
            ModelCatalog(models=[m1, m2])

    def test_empty_catalog(self) -> None:
        cat = ModelCatalog(models=[])
        assert len(cat) == 0
        assert cat.list_models() == []

    def test_repr(self, catalog: ModelCatalog) -> None:
        r = repr(catalog)
        assert "ModelCatalog" in r
        assert "qwen3-8b" in r


# ---------------------------------------------------------------------------
# ModelSpec immutability
# ---------------------------------------------------------------------------

class TestModelSpecImmutability:
    """ModelSpec instances should be frozen (immutable)."""

    def test_frozen(self, catalog: ModelCatalog) -> None:
        spec = catalog.get("qwen3-8b")
        with pytest.raises(ValidationError):
            spec.model_id = "hacked"  # type: ignore[misc]
