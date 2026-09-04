"""ultron.core.memory
~~~~~~~~~~~~~~~~~~~~

FIX #6 — memory hierarchy + context management foundation.

Exports the memory types (records, enums), the workspace-scoped project
memory store, the bounded session memory, the ephemeral working memory
view, and the deterministic ContextManager that assembles what reaches the
LLM. See docs for each module.
"""

from ultron.core.memory.context_manager import ContextBudget, ContextManager
from ultron.core.memory.models import (
    MemoryConfidence,
    MemoryKind,
    MemoryRecord,
    MemorySource,
    MemoryValidity,
)
from ultron.core.memory.project_memory import (
    ProjectMemoryStore,
    current_revision,
    project_memory_db_path,
)
from ultron.core.memory.provider import MemoryProvider
from ultron.core.memory.session_memory import SessionMemory
from ultron.core.memory.working_memory import WorkingMemory

__all__ = [
    "ContextBudget",
    "ContextManager",
    "MemoryConfidence",
    "MemoryKind",
    "MemoryProvider",
    "MemoryRecord",
    "MemorySource",
    "MemoryValidity",
    "ProjectMemoryStore",
    "SessionMemory",
    "WorkingMemory",
    "current_revision",
    "project_memory_db_path",
]
