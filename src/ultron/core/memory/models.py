"""ultron.core.memory.models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured memory representation for FIX #6.

The :class:`MemoryRecord` is the single shape every memory type (project,
session, long-term, episodic) persists and retrieves. It carries the
metadata the ContextManager needs to decide trust and relevance:

- ``source`` — WHERE the memory came from. A fact observed directly from
  source code is different from an LLM guess, and retrieval must treat
  them differently.
- ``confidence`` — a coarse, honest confidence label. No probabilistic
  precision is pretended: the system either observed the fact directly,
  inferred it, or was told it by the user / model.
- ``validity`` — lifecycle state. A superseded/stale record is never
  presented as current fact.

``MemoryRecord`` is a pydantic model (serializable via ``model_dump``) so
it round-trips through SQLite and JSON exactly like TaskState / PlanStep.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryKind(str, Enum):
    """The five memory types FIX #6 separates (plus long-term facts)."""

    PROJECT = "project"  # workspace-scoped facts about a repository
    SESSION = "session"  # current conversation/session
    LONG_TERM = "long_term"  # cross-session user facts (existing stores)
    EPISODIC = "episodic"  # what happened during a previous task
    WORKING = "working"  # ephemeral current-reasoning view (not persisted)


class MemorySource(str, Enum):
    """Where a memory came from — critical for trusting it."""

    USER = "user"  # the user told Ultron directly
    REPOSITORY_INSPECTION = "repository_inspection"  # observed in the repo
    TOOL_RESULT = "tool_result"  # a tool (command/file tool) returned it
    CODE_INTELLIGENCE = "code_intelligence"  # symbol/definition/reference layer
    TEST_RESULT = "test_result"  # observed from a test/build run
    SYSTEM_KNOWLEDGE = "system_knowledge"  # hardcoded, documented facts
    LLM_INFERENCE = "llm_inference"  # the model reasoned/guessed it


class MemoryConfidence(str, Enum):
    """Coarse trust labels — never fake precision."""

    DIRECT_OBSERVATION = "direct_observation"  # seen in source/output
    HIGH_CONFIDENCE = "high_confidence"  # consistent, corroborated
    INFERRED = "inferred"  # derived, plausible but not verified
    USER_PROVIDED = "user_provided"  # told by the user
    UNKNOWN = "unknown"


class MemoryValidity(str, Enum):
    """Lifecycle state — stale memory must not look current."""

    VALID = "valid"
    STALE = "stale"  # believed outdated (e.g. file moved); needs re-verification
    SUPERSEDED = "superseded"  # replaced by a newer record (history kept)


class MemoryRecord(BaseModel):
    """One structured memory."""

    id: int | None = None
    kind: MemoryKind = MemoryKind.PROJECT
    name: str = ""  # stable key within (workspace, kind)
    content: str = ""
    source: MemorySource = MemorySource.LLM_INFERENCE
    confidence: MemoryConfidence = MemoryConfidence.UNKNOWN
    workspace: str = ""  # project identity / workspace root
    revision: str | None = None  # optional git revision the fact was seen at
    validity: MemoryValidity = MemoryValidity.VALID
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    supersedes_id: int | None = None  # the record this one replaced
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_prompt_line(self, max_len: int = 200) -> str:
        """Compact single-line rendering for ContextManager injection."""
        tag = f"[{self.confidence.value}]"
        if self.validity is not MemoryValidity.VALID:
            tag += f"/{self.validity.value}"
        head = f"{self.name or 'memory'}: {self.content}"
        # The tag counts toward the budget so the rendered line never exceeds
        # max_len.
        room = max_len - len(tag) - 1
        if room <= 0:
            return tag[:max_len]
        if len(head) <= room:
            return f"{tag} {head}"
        return f"{tag} {head[: room - 3]}..."
