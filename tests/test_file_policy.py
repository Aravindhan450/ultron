"""
Tests for the file access policy (ultron.security.file_policy).

Covers confinement (symlinks, ``..`` escapes, absolute paths), protected-path
detection, glob allow/deny rules, extension filters, and the integration with
GuardrailsEngine.check_path and SecurityBoundary's CRITICAL-tier escalation.
"""

import pytest

from ultron.core.tools import paths as tools_paths
from ultron.security import (
    Decision,
    FilePolicy,
    RiskTier,
    SecurityBoundary,
    check_file_access,
    get_file_policy,
    is_protected,
)
from ultron.security.guardrails import GuardrailsEngine


@pytest.fixture
def base(tmp_path, monkeypatch):
    """Patches ALLOWED_BASE_DIR to a temp dir and returns it."""
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Confinement
# ---------------------------------------------------------------------------


def test_file_inside_base_allowed(base):
    result = check_file_access(base / "notes.txt", operation="read")
    assert result.ok
    assert result.resolved == (base / "notes.txt").resolve()


def test_absolute_path_outside_denied(base):
    outside = base.parent / "secret.txt"
    outside.write_text("x")
    result = check_file_access(str(outside), operation="read")
    assert not result.ok
    assert "outside the allowed working directory" in result.reason


def test_dotdot_escape_denied(base):
    result = check_file_access("../secret.txt", operation="read")
    assert not result.ok


def test_empty_path_denied(base):
    result = check_file_access("", operation="read")
    assert not result.ok
    assert "empty" in result.reason


def test_symlink_escaping_base_denied(base):
    outside = base.parent / "real_outside.txt"
    outside.write_text("x")
    link = base / "innocent_link.txt"
    link.symlink_to(outside)
    result = check_file_access(link, operation="read")
    assert not result.ok  # symlink resolved outside the base


def test_base_dir_itself_allowed(base):
    assert check_file_access(base, operation="read").ok


# ---------------------------------------------------------------------------
# Protected paths
# ---------------------------------------------------------------------------


def test_is_protected_detects_credentials():
    assert is_protected(".env")
    assert is_protected("/base/conf/.env")
    assert is_protected("~/.ssh/id_rsa")
    assert is_protected("/etc/passwd")
    assert not is_protected("notes.txt")
    assert not is_protected("env.example")  # marker is '.env', needs the dot


def test_write_to_protected_denied_inside_base(base):
    result = check_file_access(base / ".env", operation="write")
    assert not result.ok
    assert "protected" in result.reason


def test_read_of_protected_still_allowed(base):
    # Reads are never blocked by the protected-path rule (only writes are),
    # matching the boundary's tier model where reads stay LOW.
    result = check_file_access(base / ".env", operation="read")
    assert result.ok


# ---------------------------------------------------------------------------
# Globs and extensions
# ---------------------------------------------------------------------------


def test_allow_globs_restrict(base, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", base)
    policy = FilePolicy(allow_globs=("*.txt",))
    assert policy.check(base / "notes.txt", operation="read").ok
    denied = policy.check(base / "notes.md", operation="read")
    assert not denied.ok
    assert "allow rule" in denied.reason


def test_deny_globs_block(base, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", base)
    policy = FilePolicy(deny_globs=("secrets/*",))
    (base / "secrets").mkdir()
    (base / "secrets" / "x.txt").write_text("x")
    denied = policy.check(base / "secrets" / "x.txt", operation="read")
    assert not denied.ok
    assert "deny rule" in denied.reason
    assert policy.check(base / "notes.txt", operation="read").ok


def test_extension_filter_applies_to_writes_only(base, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", base)
    policy = FilePolicy(allowed_extensions={".txt", ".md"})
    assert policy.check(base / "notes.txt", operation="write").ok
    assert policy.check(base / "notes.md", operation="write").ok
    denied = policy.check(base / "notes.json", operation="write")
    assert not denied.ok
    assert "extension" in denied.reason
    # Reads are not restricted by the extension filter.
    assert policy.check(base / "notes.json", operation="read").ok


# ---------------------------------------------------------------------------
# base_dir override (embedder path)
# ---------------------------------------------------------------------------


def test_base_dir_override(tmp_path):
    policy = FilePolicy(base_dir=tmp_path)
    assert policy.check(tmp_path / "notes.txt").ok
    outside = tmp_path.parent / "x.txt"
    assert not policy.check(outside).ok
    assert not policy.is_path_safe(outside)[0]


def test_get_file_policy_singleton():
    assert get_file_policy() is get_file_policy()


# ---------------------------------------------------------------------------
# Integration: guardrails + boundary keep their exact behavior
# ---------------------------------------------------------------------------


def test_guardrails_path_escape_still_blocked(base):
    engine = GuardrailsEngine()
    result = engine.evaluate(
        action_type="read_file", target=str(base.parent / "secret.txt")
    )
    assert result.blocked
    assert any(f.rule == "path_escape" for f in result.findings)


def test_guardrails_protected_write_not_blocked(base):
    # A write to .env *inside* the base is confined (safe) — the guardrails
    # must NOT hard-block it; the boundary escalates it to CRITICAL instead.
    engine = GuardrailsEngine()
    result = engine.evaluate(
        action_type="write_file", target=str(base / ".env"), content="TOKEN=x"
    )
    assert not result.blocked


def test_boundary_escalates_protected_write_to_critical(base, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", base)
    boundary = SecurityBoundary()
    verdict = boundary.check("write_file", str(base / ".env"))
    assert verdict.tier == RiskTier.CRITICAL
    assert verdict.decision == Decision.CONFIRM  # interactive: user keeps final say


def test_boundary_plain_write_stays_high(base, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", base)
    verdict = SecurityBoundary().check("write_file", str(base / "notes.txt"))
    assert verdict.tier == RiskTier.HIGH


def test_boundary_escalates_more_protected_markers(base, monkeypatch):
    # Regression: the moved marker list must still escalate beyond .env
    # (id_rsa, /etc/passwd) to CRITICAL.
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", base)
    for target in (".ssh/id_rsa", "etc/passwd", "conf/.aws/credentials"):
        verdict = SecurityBoundary().check("write_file", str(base / target))
        assert verdict.tier == RiskTier.CRITICAL, target


def test_overwrite_operation_respects_protected_and_extensions(base, monkeypatch):
    # 'overwrite' is a state-changing operation like 'write' — it must not
    # bypass the protected-path or extension rules (regression for the
    # narrow `operation == "write"` check).
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", base)
    policy = FilePolicy(allowed_extensions={".txt"})
    assert not policy.check(base / ".env", operation="overwrite").ok
    assert not policy.check(base / "notes.json", operation="overwrite").ok
    assert policy.check(base / "notes.txt", operation="overwrite").ok
