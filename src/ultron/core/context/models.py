"""
ultron.core.context.models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured models for repository-aware context items, retrieval results,
and assembled context snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContextSourceType(str, Enum):
    """Origin category of a context item."""

    USER_TASK = "user_task"
    FILE_CONTENT = "file_content"
    SYMBOL_DEFINITION = "symbol_definition"
    SYMBOL_REFERENCE = "symbol_reference"
    SEARCH_RESULT = "search_result"
    GIT_STATE = "git_state"
    OBSERVATION = "observation"
    TEST_RESULT = "test_result"
    ARTIFACT = "artifact"
    PROJECT_CONFIG = "project_config"
    PROJECT_MEMORY = "project_memory"
    SESSION_MEMORY = "session_memory"


class ContextPriority(int, Enum):
    """
    Deterministic priority ordering for context assembly.
    Lower number = higher priority (retained first under budget constraints).
    """

    USER_TASK = 1
    DIRECT_FILE = 2
    SYMBOL = 3
    SEARCH = 4
    CHANGES_AND_DIFF = 5
    TESTS_AND_OBSERVATIONS = 6
    PROJECT_CONFIG = 7
    ARTIFACTS = 8
    PROJECT_MEMORY = 9
    SESSION_MEMORY = 10
    GENERAL_REPO = 11


class ContextItem(BaseModel):
    """
    A single granular unit of evidence assembled into model context.
    """

    source_type: ContextSourceType
    priority: ContextPriority
    title: str
    content: str
    target: str = ""
    estimated_tokens: int = 0
    is_exact_evidence: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prompt_block(self) -> str:
        """Formatted string representation for LLM prompt."""
        header = f"[{self.source_type.value.upper()}: {self.title}]"
        return f"{header}\n{self.content.strip()}"


class ContextRetrievalStatus(str, Enum):
    """Result status of a context retrieval attempt."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    ACCESS_DENIED = "access_denied"
    ERROR = "error"


class ContextRetrievalResult(BaseModel):
    """
    Structured outcome of querying a file, symbol, or search target.
    Explicitly represents NOT_FOUND rather than hallucinating missing assets.
    """

    target: str
    status: ContextRetrievalStatus
    source_type: ContextSourceType
    items: list[ContextItem] = Field(default_factory=list)
    error_message: str | None = None
    searched_locations: list[str] = Field(default_factory=list)

    @property
    def is_found(self) -> bool:
        return self.status is ContextRetrievalStatus.FOUND and len(self.items) > 0


class ContextSnapshot(BaseModel):
    """
    Observable snapshot of context assembled for an execution turn.
    Exposes exactly what evidence was provided to the agent/model.
    """

    assembled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    items: list[ContextItem] = Field(default_factory=list)
    total_estimated_tokens: int = 0
    total_characters: int = 0
    dropped_items_count: int = 0
    compacted: bool = False
    source_contributions: dict[str, int] = Field(default_factory=dict)

    @property
    def formatted_context(self) -> str:
        """Returns the formatted prompt string containing all accepted items."""
        return "\n\n".join(item.prompt_block() for item in self.items)


def estimate_tokens(text: str) -> int:
    """
    Standard documented estimation of token count (~4 characters per token).
    Clearly distinguished as an estimation.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)

