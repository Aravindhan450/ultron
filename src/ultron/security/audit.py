"""
ultron.security.audit
~~~~~~~~~~~~~~~~~~~~~

Structured audit trail for the security boundary.

Every verdict the boundary produces (``allow`` / ``confirm`` / ``deny``) is
appended to a JSON-lines file — one JSON object per line — so the history of
decisions is machine-readable, greppable, and trivially tail-able:

    ~/.ultron/security_audit.jsonl

A single record looks like::

    {
        "ts": "2026-08-07T12:34:56.789012+00:00",
        "event": "security.verdict",
        "schema": 1,
        "action_type": "run_command",
        "target": "rm -rf /",
        "target_truncated": false,
        "tier": "critical",
        "decision": "confirm",
        "mode": "interactive",
        "reason": "Risk tier 'critical' -> 'confirm' under security mode 'interactive'",
        "guardrails": {"passed": true, "blocked": false, "findings": []}
    }

Security properties:

- **Never records raw content.** The audit trail stores the *target* (the
  command / filename / URL / SQL) but deliberately omits free-form payloads
  such as file contents or HTTP bodies, which could contain secrets.
- **Targets are secret-scanned before persisting.** Even when a target slips
  past the guardrails (a credential format the scanner does not know, or an
  action that was merely allowed), anything that *looks* like a credential
  (API key, bearer token, JWT, …) is redacted to asterisks. Note the scanner
  is heuristic — it cannot guarantee no secret ever appears, but it applies
  the same detection the gate itself uses.
- **Guardrail findings record rule names only.** The matched snippet is
  dropped — a leaked credential must not be persisted, even truncated.
- **Targets are capped** at a fixed length and flagged with
  ``target_truncated``.
- **Writes are thread-safe** (append under a lock) and the file rotates at a
  fixed size, mirroring the main application log.
- **Auditing never breaks the gate.** Serialization or I/O failures are
  logged and swallowed so a broken audit path can never block a decision.
"""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from ultron.core.config import settings
from ultron.core.logging import get_logger
from ultron.security.models import BoundaryResult
from ultron.security.scanners.secret import mask_matches, scan_secrets

logger = get_logger("ultron.security.audit")

# Default location for the audit trail, next to the main application log.
DEFAULT_AUDIT_FILE = settings.data_dir / "security_audit.jsonl"

# Schema version for forward-compatible readers.
_SCHEMA_VERSION = 1

# Targets longer than this are truncated before being persisted.
_TARGET_MAX_LEN = 400

# Rotation policy, mirroring the main log (10 MB, keep a few backups).
_ROTATE_BYTES = 10 * 1024 * 1024
_ROTATE_BACKUPS = 3

# Guards append + rotation so concurrent verdicts never interleave lines.
_LOCK = threading.Lock()


class AuditLog:
    """
    Append-only JSON-lines audit log for security verdicts.

    Args:
        path: Where records are written. Defaults to
            ``~/.ultron/security_audit.jsonl`` (``settings.data_dir``).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_AUDIT_FILE

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(self, result: BoundaryResult, mode: str | None = None) -> dict | None:
        """
        Appends one verdict as a JSON line and returns the record written.

        Returns ``None`` when the record could not be serialized or written
        (the failure is logged, never raised).
        """
        record = self._build_record(result, mode)
        if self._append(record):
            return record
        return None

    def _build_record(self, result: BoundaryResult, mode: str | None) -> dict:
        """Turns a verdict into the exact JSON object that gets persisted."""
        # Targets are redacted through the same secret scanner the gate uses:
        # a credential embedded in a command/URL must not be persisted even
        # when the action itself was allowed.
        target = result.target or ""
        secret_hits = scan_secrets(target)
        if secret_hits:
            target = mask_matches(target, secret_hits)
        truncated = len(target) > _TARGET_MAX_LEN
        if truncated:
            target = target[:_TARGET_MAX_LEN] + "…"

        entry: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "event": "security.verdict",
            "schema": _SCHEMA_VERSION,
            "action_type": result.action_type,
            "target": target,
            "target_truncated": truncated,
            "tier": result.tier.value,
            "decision": result.decision.value,
            "mode": mode,
            "reason": result.reason,
        }

        # Always present (None when the verdict carried no guardrail scan)
        # so readers never have to handle a missing key.
        entry["guardrails"] = (
            {
                "passed": result.guardrails.passed,
                "blocked": result.guardrails.blocked,
                # Rule names only — never persist matched snippets (secrets).
                "findings": [
                    {"rule": f.rule, "severity": f.severity, "location": f.location}
                    for f in result.guardrails.findings
                ],
            }
            if result.guardrails is not None
            else None
        )
        return entry

    def _append(self, record: dict) -> bool:
        """Serializes and appends one record (rotating first if needed).

        Returns True when the record was persisted, False otherwise.
        """
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("audit: could not serialize verdict: %s", exc)
            return False

        with _LOCK:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._maybe_rotate()
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                return True
            except OSError as exc:
                logger.warning("audit: could not write %s: %s", self.path, exc)
                return False

    def _maybe_rotate(self) -> None:
        """Rolls the file once it passes the size cap (``audit.jsonl.1`` …)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return  # file does not exist yet
        if size < _ROTATE_BYTES:
            return

        for i in range(_ROTATE_BACKUPS, 1, -1):
            src = Path(f"{self.path}.{i - 1}")
            dst = Path(f"{self.path}.{i}")
            try:
                if dst.exists():
                    dst.unlink()
                if src.exists():
                    src.rename(dst)
            except OSError:
                pass
        try:
            self.path.rename(Path(f"{self.path}.1"))
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> list[dict]:
        """
        Returns every record currently in the active file, in order.

        Corrupt or truncated lines (e.g. a crash mid-write) are skipped.
        """
        if not self.path.exists():
            return []
        records: list[dict] = []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("audit: skipping unparseable line in %s", self.path)
        except OSError as exc:
            logger.warning("audit: could not read %s: %s", self.path, exc)
        return records


# Shared singleton, created once from the active settings. Pass an explicit
# path/instance when embedding Ultron to redirect the audit trail.
_audit_log: AuditLog | None = None


def get_audit_log(path: Path | str | None = None) -> AuditLog:
    """
    Returns the shared :class:`AuditLog`.

    A ``path`` creates a standalone instance (used by tests and embedders)
    that does **not** rebind the shared singleton — later ``record_verdict``
    calls still write to the default file. To redirect the whole process's
    audit trail, construct and pass an explicit ``AuditLog`` to the
    ``SecurityBoundary`` instead.
    """
    if path is not None:
        return AuditLog(path)
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog()
    return _audit_log


def record_verdict(result: BoundaryResult, mode: str | None = None) -> dict | None:
    """
    Records one boundary verdict to the shared audit log.

    Convenience wrapper used by the security boundary itself.
    """
    return get_audit_log().record(result, mode)
