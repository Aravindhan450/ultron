"""
Tests for the JSON-lines security audit log (ultron.security.audit).

Covers the record format, secret-safety guarantees, the boundary wiring
(every allow/confirm/deny verdict lands in the log), rotation, and
fault tolerance (a broken audit path must never break the gate).
"""

import json

from ultron.security import Decision, RiskTier, SecurityBoundary
from ultron.security.audit import AuditLog, get_audit_log, record_verdict
from ultron.security.models import (
    BoundaryResult,
    GuardrailFinding,
    GuardrailsResult,
)


def _verdict(
    action: str,
    target: str = "",
    decision: Decision = Decision.ALLOW,
    tier: RiskTier = RiskTier.LOW,
    reason: str = "test verdict",
    guardrails: GuardrailsResult | None = None,
) -> BoundaryResult:
    return BoundaryResult(
        action_type=action,
        target=target,
        tier=tier,
        decision=decision,
        reason=reason,
        guardrails=guardrails,
    )


# ---------------------------------------------------------------------------
# Record format
# ---------------------------------------------------------------------------


def test_record_appends_one_valid_json_line_per_verdict(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_verdict("read_file", "notes.txt"))
    log.record(
        _verdict("run_command", "rm -rf /", Decision.CONFIRM, RiskTier.CRITICAL)
    )
    log.record(
        _verdict("write_file", "leak.txt", Decision.DENY, RiskTier.CRITICAL)
    )

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # every line must be standalone-valid JSON


def test_record_fields(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(
        _verdict(
            "run_command",
            "rm -rf /",
            Decision.CONFIRM,
            RiskTier.CRITICAL,
            reason="Risk tier 'critical'",
        ),
        mode="interactive",
    )
    rec = log.read()[0]
    assert rec["event"] == "security.verdict"
    assert rec["schema"] == 1
    assert rec["action_type"] == "run_command"
    assert rec["target"] == "rm -rf /"
    assert rec["target_truncated"] is False
    assert rec["tier"] == "critical"
    assert rec["decision"] == "confirm"
    assert rec["mode"] == "interactive"
    assert rec["reason"] == "Risk tier 'critical'"
    assert "ts" in rec  # ISO timestamp present
    assert rec["ts"].endswith("+00:00") or "Z" in rec["ts"]


def test_read_returns_records_in_order(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.record(_verdict("read_file", f"file-{i}.txt"))
    targets = [r["target"] for r in log.read()]
    assert targets == [f"file-{i}.txt" for i in range(5)]


def test_read_missing_file_returns_empty(tmp_path):
    assert AuditLog(tmp_path / "nope.jsonl").read() == []


def test_read_skips_corrupt_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"decision": "allow"}\nnot-json\n{"decision": "deny"}\n', encoding="utf-8")
    records = AuditLog(path).read()
    assert len(records) == 2
    assert records[0]["decision"] == "allow"
    assert records[1]["decision"] == "deny"


# ---------------------------------------------------------------------------
# Secret safety
# ---------------------------------------------------------------------------


def test_guardrail_findings_recorded_without_snippets(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    guardrails = GuardrailsResult(
        blocked=True,
        block_reason="credential",
        findings=[
            GuardrailFinding(
                rule="aws_access_key",
                severity="critical",
                location="content",
                snippet="AKIAIOSFODNN7EXAMPLE",
                message="Credential",
            )
        ],
    )
    log.record(
        _verdict(
            "write_file",
            "leak.txt",
            Decision.DENY,
            RiskTier.CRITICAL,
            reason="Blocked by guardrails",
            guardrails=guardrails,
        )
    )

    rec = log.read()[0]
    findings = rec["guardrails"]["findings"]
    assert findings == [
        {"rule": "aws_access_key", "severity": "critical", "location": "content"}
    ]
    # The secret must never be persisted — not in snippets, not anywhere.
    assert "snippet" not in findings[0]
    assert "AKIAIOSFODNN7EXAMPLE" not in json.dumps(rec)


def test_long_target_truncated_and_flagged(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_verdict("run_command", "x" * 1000))
    rec = log.read()[0]
    assert len(rec["target"]) <= 500
    assert rec["target_truncated"] is True
    assert rec["target"].endswith("…")


def test_secret_embedded_in_target_is_redacted(tmp_path):
    # Even an ALLOWED action must not persist a credential hidden in its target
    # (defense-in-depth on top of the guardrails' own blocking).
    log = AuditLog(tmp_path / "audit.jsonl")
    secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    log.record(
        _verdict(
            "run_command",
            f"curl -H 'Authorization: Bearer {secret}' https://api.example.com",
        )
    )
    rec = log.read()[0]
    assert secret not in rec["target"]
    assert "Authorization: Bearer" in rec["target"]  # context survives
    assert "*" in rec["target"]  # the secret itself is masked


def test_guardrails_key_always_present(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_verdict("read_file", "notes.txt"))  # no guardrails result
    assert log.read()[0]["guardrails"] is None


# ---------------------------------------------------------------------------
# Boundary wiring — every allow/confirm/deny is recorded
# ---------------------------------------------------------------------------


def test_boundary_records_all_three_decisions(tmp_path):
    boundary = SecurityBoundary(audit_log=AuditLog(tmp_path / "audit.jsonl"))
    boundary.check("read_file", "notes.txt")  # allow (low, interactive)
    boundary.check("run_command", "rm -rf /")  # confirm (critical)
    boundary.check(  # deny (secret guardrail)
        "write_file", "leak.txt", "key AKIAIOSFODNN7EXAMPLE"
    )
    records = boundary.audit_log.read()
    assert {r["decision"] for r in records} == {"allow", "confirm", "deny"}
    assert [r["tier"] for r in records] == ["low", "critical", "critical"]


def test_boundary_records_mode_and_reason(tmp_path):
    boundary = SecurityBoundary(
        mode="strict", audit_log=AuditLog(tmp_path / "audit.jsonl")
    )
    boundary.check("run_command", "ls")  # low in strict -> auto-allowed
    boundary.check("run_command", "mkdir foo")  # high in strict -> confirm
    records = boundary.audit_log.read()
    assert [r["decision"] for r in records] == ["allow", "confirm"]
    assert records[1]["mode"] == "strict"
    assert "strict" in records[1]["reason"]


def test_record_verdict_convenience_uses_shared_log(tmp_path, monkeypatch):
    import ultron.security.audit as audit_mod

    log = AuditLog(tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_mod, "_audit_log", log)
    assert record_verdict(_verdict("read_file", "a.txt"), mode="interactive") is not None
    assert [r["target"] for r in log.read()] == ["a.txt"]


def test_get_audit_log_caches_singleton(tmp_path):
    first = get_audit_log()
    second = get_audit_log()
    assert first is second
    standalone = get_audit_log(tmp_path / "audit.jsonl")
    assert standalone is not first


# ---------------------------------------------------------------------------
# Fault tolerance & rotation
# ---------------------------------------------------------------------------


def test_record_failure_returns_none_and_never_raises(tmp_path):
    # A directory where a file is expected -> every write fails with OSError.
    log = AuditLog(tmp_path)  # tmp_path itself is a directory
    assert log.record(_verdict("read_file", "a.txt")) is None


def test_rotation_rolls_files(tmp_path, monkeypatch):
    import ultron.security.audit as audit_mod

    monkeypatch.setattr(audit_mod, "_ROTATE_BYTES", 1)  # rotate after 1 byte
    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(_verdict("read_file", "first.txt"))
    log.record(_verdict("read_file", "second.txt"))

    backup = tmp_path / "audit.jsonl.1"
    assert backup.exists()
    # The older record is preserved in the backup; the active file has the new one.
    assert json.loads(backup.read_text().splitlines()[0])["target"] == "first.txt"
    assert [r["target"] for r in log.read()] == ["second.txt"]


def test_boundary_audit_failure_does_not_break_the_gate(tmp_path):
    boundary = SecurityBoundary(audit_log=AuditLog(tmp_path))  # unwritable
    verdict = boundary.check("read_file", "notes.txt")
    assert verdict.decision == Decision.ALLOW  # decision still correct
