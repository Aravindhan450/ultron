"""ultron.core.coding.intelligence.lsp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LSP (Language Server Protocol) abstraction for the code intelligence layer
(Fix #4).

This module defines the CONTRACT only — no concrete language server is
bundled. It provides:

- :class:`LSPOperation` — the operations a client may support.
- :class:`LSPCapabilities` — which operations a server supports.
- :class:`LSPClient` — the protocol any real client must implement.
- :class:`LSPServerManager` — the protocol for discovering/starting servers.
- :class:`NoLSPServers` — the deterministic default manager: nothing is
  available, every operation degrades gracefully (returns ``None`` or an
  explicit "not available" marker) instead of raising.

Every operation returns ``None`` for "unavailable / unsupported" so the
facade can fall back to the index/lexical layers without error handling
sprawl. Capability checks are explicit: an operation a server does not
support returns ``None``, never a crash.

NOT YET IMPLEMENTED (by design): request timeouts and crash recovery. A
concrete client (``LSPClient`` implementor) owns process lifecycle, and the
management layer must add timeout/kill-on-hang handling when a real server
is wired in. Until then the abstraction degrades deterministically.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel


class LSPOperation(str, Enum):
    """Operations a language server may or may not support."""

    INITIALIZE = "initialize"
    SHUTDOWN = "shutdown"
    DEFINITION = "definition"
    REFERENCES = "references"
    HOVER = "hover"
    DOCUMENT_SYMBOLS = "document_symbols"
    WORKSPACE_SYMBOLS = "workspace_symbols"
    IMPLEMENTATION = "implementation"
    CALL_HIERARCHY = "call_hierarchy"


class LSPCapabilities(BaseModel):
    """Which operations an available server supports."""

    operations: set[LSPOperation] = set()  # type: ignore[assignment]

    def supports(self, operation: LSPOperation) -> bool:
        return operation in self.operations


class LSPLocation(BaseModel):
    """One LSP location result (file URI + line/character range)."""

    uri: str
    line: int  # 0-based in LSP
    character: int
    end_line: int | None = None
    end_character: int | None = None

    def to_prompt_line(self) -> str:
        return f"{self.uri}:{self.line + 1}"


class LSPSymbol(BaseModel):
    """One document/workspace symbol result."""

    name: str
    kind: str  # LSP SymbolKind numeric value as string
    uri: str
    line: int
    character: int


class LSPClient(Protocol):
    """Contract for any concrete LSP client."""

    server_id: str
    capabilities: LSPCapabilities

    def initialize(self, root: str) -> bool: ...
    def shutdown(self) -> None: ...
    def definition(self, uri: str, line: int, character: int) -> list[LSPLocation] | None: ...
    def references(self, uri: str, line: int, character: int) -> list[LSPLocation] | None: ...
    def hover(self, uri: str, line: int, character: int) -> str | None: ...
    def document_symbols(self, uri: str) -> list[LSPSymbol] | None: ...
    def workspace_symbols(self, query: str) -> list[LSPSymbol] | None: ...
    def implementation(self, uri: str, line: int, character: int) -> list[LSPLocation] | None: ...
    def call_hierarchy(self, uri: str, line: int, character: int) -> list[LSPLocation] | None: ...


class LSPServerManager(Protocol):
    """Contract for discovering and managing language servers."""

    def detect(self) -> list[str]: ...
    def start(self, server_id: str, root: str) -> LSPClient | None: ...
    def stop(self, client: LSPClient) -> None: ...


class NoLSPServers:
    """
    Deterministic default: no language servers are configured or detected.

    Every operation degrades gracefully (returns None) so higher layers
    always have a well-defined fallback path. This is the correct behavior
    in CI and on machines without language servers installed.
    """

    def detect(self) -> list[str]:
        return []

    def start(self, server_id: str, root: str) -> LSPClient | None:
        return None

    def stop(self, client: LSPClient) -> None:
        return None


class LSPUnavailableError(Exception):
    """Raised when an LSP operation is requested but no server exists."""


class LSPFacade:
    """
    Convenience facade over a manager + an optional started client.

    ``available()`` reports whether any server was detected. Operations
    return None (or empty lists) when no server is available or the
    operation is unsupported — callers should fall back to the index.
    """

    def __init__(self, manager: LSPServerManager | None = None) -> None:
        self.manager: LSPServerManager = manager or NoLSPServers()
        self.client: LSPClient | None = None

    @property
    def available(self) -> bool:
        return bool(self.client is not None)

    def start(self, root: str, preferred: str | None = None) -> bool:
        """Starts the first (or preferred) detected server for *root*."""
        servers = self.manager.detect()
        if not servers:
            return False
        server_id = preferred if preferred in servers else servers[0]
        client = self.manager.start(server_id, root)
        if client is None or not client.initialize(root):
            return False
        self.client = client
        return True

    def stop(self) -> None:
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception:  # noqa: BLE001 — shutdown must never raise
                self.client = None
            self.manager.stop(self.client)
            self.client = None

    # -- operations (all degrade to None when unavailable/unsupported) -----

    def _supports(self, operation: LSPOperation) -> bool:
        return bool(
            self.client is not None
            and self.client.capabilities.supports(operation)
        )

    def definition(self, uri: str, line: int, character: int) -> list[LSPLocation] | None:
        if not self._supports(LSPOperation.DEFINITION) or self.client is None:
            return None
        return self.client.definition(uri, line, character)

    def references(self, uri: str, line: int, character: int) -> list[LSPLocation] | None:
        if not self._supports(LSPOperation.REFERENCES) or self.client is None:
            return None
        return self.client.references(uri, line, character)

    def hover(self, uri: str, line: int, character: int) -> str | None:
        if not self._supports(LSPOperation.HOVER) or self.client is None:
            return None
        return self.client.hover(uri, line, character)

    def document_symbols(self, uri: str) -> list[LSPSymbol] | None:
        if not self._supports(LSPOperation.DOCUMENT_SYMBOLS) or self.client is None:
            return None
        return self.client.document_symbols(uri)

    def workspace_symbols(self, query: str) -> list[LSPSymbol] | None:
        if not self._supports(LSPOperation.WORKSPACE_SYMBOLS) or self.client is None:
            return None
        return self.client.workspace_symbols(query)

    def implementation(self, uri: str, line: int, character: int) -> list[LSPLocation] | None:
        if not self._supports(LSPOperation.IMPLEMENTATION) or self.client is None:
            return None
        return self.client.implementation(uri, line, character)

    def call_hierarchy(self, uri: str, line: int, character: int) -> list[LSPLocation] | None:
        if not self._supports(LSPOperation.CALL_HIERARCHY) or self.client is None:
            return None
        return self.client.call_hierarchy(uri, line, character)
