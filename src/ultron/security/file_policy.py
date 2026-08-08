"""
ultron.security.file_policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

File access policy — decides whether a path may be read or written.

The policy is the single source of truth for *which files the agents may
touch*. It layers several rules, evaluated in order for every request:

1. **Base-dir confinement** — the resolved path (symlinks followed, ``..``
   collapsed) must stay inside the directory Ultron was launched from
   (``ALLOWED_BASE_DIR`` in ``core.tools.paths``). This is the hard boundary;
   the guardrails turn any escape into a ``deny``.
2. **Protected paths** (state-changing ops only) — system / credential
   files (``.env``, ``.ssh/``, ``/etc/``, key material, …) are flagged so
   the boundary can escalate them to CRITICAL. Unlike escapes they are
   never hard-blocked here — the user keeps the final say, per the
   "permission first" model.
3. **Allow / deny glob rules** — optional ``allow_globs`` / ``deny_globs``
   for embedders that want stricter per-project policies.
4. **Extension filters** — optional ``allowed_extensions`` restriction for
   state-changing operations.

The security module reuses this policy: ``GuardrailsEngine.check_path``
delegates the confinement check here (rule ``path_escape``), and
``SecurityBoundary._touches_system_path`` delegates the protected-path
detection here, so the marker list lives in exactly one place.

Usage::

    policy = FilePolicy()
    result = policy.check("notes.txt", operation="write")
    result.ok      # True
    policy.is_protected(".env")   # True
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

from ultron.core.tools.paths import is_path_safe

# System / credential markers. A path containing any of these is considered
# protected: writes are escalated to CRITICAL by the boundary (never silently
# executed, never hard-blocked either — the user keeps the final say).
SYSTEM_PATH_MARKERS = (
    "/etc/",
    "/etc/passwd",
    "/etc/shadow",
    "/boot/",
    "/System/",
    "/Library/LaunchDaemons",
    ".ssh/",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    ".aws/",
    ".git-credentials",
    ".htpasswd",
    "authorized_keys",
    "key.pem",
    "wallet",
    "secrets.json",
    ".env",
)


@dataclass
class FilePolicyResult:
    """
    Verdict for one file access request.

    Attributes:
        ok: True when the request is permitted.
        operation: ``read`` or ``write``.
        reason: Human-readable explanation when ``ok`` is False.
        resolved: The resolved absolute path (as far as it could be
            resolved), when available.
    """

    ok: bool
    operation: str = "read"
    reason: str = ""
    resolved: Path | None = None

    def __bool__(self) -> bool:
        return self.ok


class FilePolicy:
    """
    Evaluates file access requests against the configured rules.

    Args:
        base_dir: Optional override for the confinement root. When None
            (default), the module-level ``ALLOWED_BASE_DIR`` from
            ``core.tools.paths`` is read at call time — so changing it
            (e.g. in tests) takes effect immediately.
        allow_globs: When non-empty, a path must match at least one glob
            (matched against its path relative to the base dir).
        deny_globs: Any matching glob denies the request outright.
        allowed_extensions: When given, write operations are restricted to
            files with one of these suffixes (e.g. ``{".txt", ".md"}``).
    """

    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        allow_globs: tuple[str, ...] = (),
        deny_globs: tuple[str, ...] = (),
        allowed_extensions: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None
        self.allow_globs = tuple(allow_globs)
        self.deny_globs = tuple(deny_globs)
        self.allowed_extensions = (
            frozenset(e.lower() for e in allowed_extensions)
            if allowed_extensions is not None
            else None
        )

    # ------------------------------------------------------------------
    # The gate
    # ------------------------------------------------------------------

    def check(self, path: str | Path, operation: str = "read") -> FilePolicyResult:
        """
        Evaluates one file access request.

        Rules are applied in order: emptiness, base-dir confinement,
        protected paths and extension filters (state-changing ops only),
        and allow/deny globs. The first failure wins.
        """
        raw = str(path or "").strip()
        if not raw:
            return FilePolicyResult(ok=False, operation=operation, reason="Path is empty")

        safe, resolved = self._confinement(raw)
        if not safe:
            return FilePolicyResult(
                ok=False,
                operation=operation,
                reason="Target path is outside the allowed working directory",
                resolved=resolved,
            )

        # Any state-changing operation ("write", "overwrite", "delete", …)
        # triggers the protected-path and extension rules; only reads skip
        # them. Matching on `!= "read"` avoids aliases silently bypassing
        # the rules (the agent layer uses both write_file and overwrite_file).
        if operation != "read" and self.is_protected(str(resolved)):
            return FilePolicyResult(
                ok=False,
                operation=operation,
                reason="Target path is protected (system/credential path)",
                resolved=resolved,
            )

        rel = self._relative(resolved)
        if not self._matches_allow(rel):
            return FilePolicyResult(
                ok=False,
                operation=operation,
                reason="Path does not match any allow rule",
                resolved=resolved,
            )
        denied = next((g for g in self.deny_globs if fnmatch.fnmatch(rel, g)), None)
        if denied is not None:
            return FilePolicyResult(
                ok=False,
                operation=operation,
                reason=f"Path matches deny rule '{denied}'",
                resolved=resolved,
            )

        if operation != "read" and self.allowed_extensions is not None:
            suffix = Path(resolved).suffix.lower()
            if suffix not in self.allowed_extensions:
                return FilePolicyResult(
                    ok=False,
                    operation=operation,
                    reason=(
                        f"File extension '{suffix or '(none)'}' is not allowed "
                        f"(allowed: {', '.join(sorted(self.allowed_extensions))})"
                    ),
                    resolved=resolved,
                )

        return FilePolicyResult(ok=True, operation=operation, resolved=resolved)

    # ------------------------------------------------------------------
    # Individual rules (reusable on their own)
    # ------------------------------------------------------------------

    def is_path_safe(self, path: str | Path) -> tuple[bool, Path]:
        """
        Confinement check only.

        Returns ``(is_safe, resolved_path)`` — the same contract as
        ``core.tools.paths.is_path_safe``. When ``base_dir`` was given it is
        used directly; otherwise the check delegates to the module-level
        function so a patched ``ALLOWED_BASE_DIR`` still takes effect.
        """
        if self.base_dir is None:
            return is_path_safe(path)
        try:
            resolved = Path(path).expanduser().resolve()
            resolved.relative_to(self.base_dir)
            return True, resolved
        except (OSError, ValueError):
            return False, Path(path)

    def is_protected(self, path: str | Path) -> bool:
        """
        True when *path* references a system/credential path.

        Substring match on the lowercased path against
        :data:`SYSTEM_PATH_MARKERS`, mirroring the boundary's tier-escalation
        rule.
        """
        text = str(path or "").lower()
        return any(marker in text for marker in SYSTEM_PATH_MARKERS)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _confinement(self, raw: str) -> tuple[bool, Path]:
        return self.is_path_safe(raw)

    def _relative(self, resolved: Path) -> str:
        """Path relative to the base dir, for glob matching."""
        if self.base_dir is not None:
            try:
                return resolved.relative_to(self.base_dir).as_posix()
            except ValueError:
                return resolved.as_posix()
        # Deliberately imported *inside* the method (not at module top):
        # `from X import NAME` binds the value at import time, which would
        # freeze ALLOWED_BASE_DIR and break the monkeypatch-sensitive tests.
        # Reading it here keeps the call-time semantics of is_path_safe.
        from ultron.core.tools.paths import ALLOWED_BASE_DIR

        try:
            return resolved.relative_to(ALLOWED_BASE_DIR).as_posix()
        except (ValueError, OSError):
            return resolved.as_posix()

    def _matches_allow(self, rel: str) -> bool:
        if not self.allow_globs:
            return True
        return any(fnmatch.fnmatch(rel, glob) for glob in self.allow_globs)


# Lazily-created default policy, shared across the security module.
_default_policy: FilePolicy | None = None


def get_file_policy() -> FilePolicy:
    """Returns the shared default :class:`FilePolicy`."""
    global _default_policy
    if _default_policy is None:
        _default_policy = FilePolicy()
    return _default_policy


def check_file_access(path: str | Path, operation: str = "read") -> FilePolicyResult:
    """
    Convenience wrapper: evaluates one request against the default policy.
    """
    return get_file_policy().check(path, operation)


def is_protected(path: str | Path) -> bool:
    """
    Convenience wrapper: True when *path* references a system/credential path.

    Used by the boundary's tier escalation and available to callers that
    need the protected-path rule without the full policy object.
    """
    return get_file_policy().is_protected(path)
