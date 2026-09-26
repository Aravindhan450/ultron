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

    # Static instructions
    SYSTEM = "system"

    # Task & planning
    USER_TASK = "user_task"
    ACTIVE_PLAN_STEP = "active_plan_step"

    # Failures & verification
    FAILURE = "failure"
    VERIFICATION = "verification"

    # Repository & code
    FILE_CONTENT = "file_content"
    SYMBOL_DEFINITION = "symbol_definition"
    SYMBOL_REFERENCE = "symbol_reference"
    SEARCH_RESULT = "search_result"
    REPO_MAP = "repo_map"
    GIT_STATE = "git_state"
    CHANGES_AND_DIFF = "changes_and_diff"
    PROJECT_CONFIG = "project_config"

    # Dialogue & history
    RECENT_DIALOGUE = "recent_dialogue"
    OLDER_HISTORY = "older_history"

    # Observations & artifacts
    OBSERVATION = "observation"
    TOOL_OBSERVATION = "tool_observation"
    TEST_RESULT = "test_result"
    ARTIFACT = "artifact"

    # Long-term memory
    PROJECT_MEMORY = "project_memory"
    SESSION_MEMORY = "session_memory"
    LONG_TERM_MEMORY = "long_term_memory"


class ContextPriority(int, Enum):
    """
    Deterministic priority ordering for context assembly per Gap Closure Plan:
    1. CURRENT TASK
    2. ACTIVE PLAN STEP
    3. LATEST FAILURE
    4. RELEVANT REPOSITORY CONTEXT (direct files, symbols, diffs, search, repo map)
    5. RECENT DIALOGUE / OBSERVATIONS
    6. OLDER HISTORY / ARTIFACTS / ENVIRONMENT CONFIG
    7. LONG-TERM MEMORY (project memory, session memory)

    Lower number = higher priority (retained first under budget constraints).
    """

    SYSTEM = 0
    USER_TASK = 1
    ACTIVE_PLAN_STEP = 2
    LATEST_FAILURE = 3

    # Repository context
    DIRECT_FILE = 4
    SYMBOL = 5
    CHANGES_AND_DIFF = 6
    SEARCH = 7
    REPO_MAP = 8

    # Recent observations & dialogue
    RECENT_OBSERVATIONS = 9
    RECENT_DIALOGUE = 10

    # Older history & configuration
    PROJECT_CONFIG = 11
    OLDER_HISTORY = 12
    ARTIFACTS = 13
    GENERAL_REPO = 14

    # Long-term memory
    PROJECT_MEMORY = 15
    SESSION_MEMORY = 16
    LONG_TERM_MEMORY = 17


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

