"""ultron.core.coding.observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structured observations for the coding agent.

Instead of forcing every tool result into one giant string, an
:class:`Observation` distinguishes *what kind* of result it is (file
content, search result, command result, test result, build result, error,
diff, repository state) and carries both a compact one-line summary (for
prompt context) and the full detail (for the executor).

Observations are pydantic models, so they serialize cleanly inside a
CodeContext / TaskState and can be logged or persisted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class ObservationKind(str, Enum):
    """The kind of result an observation represents."""

    FILE_CONTENT = "file_content"
    SEARCH_RESULT = "search_result"
    COMMAND_RESULT = "command_result"
    TEST_RESULT = "test_result"
    BUILD_RESULT = "build_result"
    ERROR = "error"
    DIFF = "diff"
    REPOSITORY_STATE = "repository_state"
    EDIT_RESULT = "edit_result"


class Observation(BaseModel):
    """One structured result produced by the coding agent's execution."""

    kind: ObservationKind
    source: str = ""  # tool name / file path / command that produced it
    summary: str = ""  # compact one-line summary for prompt context
    detail: str = ""  # full text (bounded by callers to keep context small)
    success: bool | None = None
    exit_code: int | None = None
    duration_ms: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def error(cls, source: str, summary: str, detail: str = "") -> Observation:
        """Builds a failed observation (kind=ERROR)."""
        return cls(
            kind=ObservationKind.ERROR,
            source=source,
            summary=summary,
            detail=detail,
            success=False,
        )

    def to_prompt_line(self, max_len: int = 200) -> str:
        """One-line rendering for prompt/context injection."""
        status = ""
        if self.success is True:
            status = " [ok]"
        elif self.success is False:
            status = " [FAILED]"
        head = f"[{self.kind.value}] {self.source}{status}"
        if self.exit_code is not None:
            head += f" (exit {self.exit_code})"
        if self.summary:
            return f"{head}: {self.summary[: max_len - len(head) - 2]}"
        return head
