"""
Tests for the permission layer: the classifier (which reuses the security
boundary's risk tiers) and the interactive confirmation flow.

The confirmation flow is tested with injected prompt/render functions so no
TTY or questionary interaction is required.
"""

import asyncio

from ultron.core.types import PendingAction
from ultron.permissions import PermissionClassifier, PermissionLevel, confirm_action
from ultron.security import Decision, RiskTier, SecurityBoundary


def _classifier(mode: str | None = None) -> PermissionClassifier:
    if mode:
        return PermissionClassifier(boundary=SecurityBoundary(mode=mode))
    return PermissionClassifier()


# ---------------------------------------------------------------------------
# classifier: reuses the boundary's tiers
# ---------------------------------------------------------------------------


def test_read_file_auto_allowed():
    req = _classifier().classify("read_file", "notes.txt")
    assert req.tier == RiskTier.LOW
    assert req.decision == Decision.ALLOW
    assert req.permission == PermissionLevel.AUTO
    assert req.needs_confirmation is False


def test_write_file_requires_confirmation():
    req = _classifier().classify("write_file", "notes.txt", "hello")
    assert req.tier == RiskTier.HIGH
    assert req.decision == Decision.CONFIRM
    assert req.permission == PermissionLevel.CONFIRM
    assert req.needs_confirmation is True


def test_system_path_write_is_critical_typed():
    req = _classifier().classify("write_file", ".env", "x=1")
    assert req.tier == RiskTier.CRITICAL
    assert req.permission == PermissionLevel.CONFIRM_CRITICAL
    assert req.needs_confirmation is True
    assert req.is_critical is True


def test_secret_content_denied():
    req = _classifier().classify("write_file", "leak.txt", "AKIA1234567890ABCDEF")
    assert req.decision == Decision.DENY
    assert req.permission == PermissionLevel.DENY
    assert req.needs_confirmation is False


def test_dangerous_command_is_critical():
    req = _classifier().classify("run_command", "rm -rf /")
    assert req.tier == RiskTier.CRITICAL
    assert req.permission == PermissionLevel.CONFIRM_CRITICAL


def test_permissive_mode_auto_allows_write():
    req = _classifier(mode="permissive").classify("write_file", "notes.txt")
    assert req.decision == Decision.ALLOW
    assert req.permission == PermissionLevel.AUTO


def test_strict_mode_confirms_write():
    req = _classifier(mode="strict").classify("write_file", "notes.txt")
    assert req.permission == PermissionLevel.CONFIRM


def test_classify_pending_normalizes_fetch_page():
    # The agent layer emits action_type "fetch_page"; the boundary knows it as
    # "fetch_page_text". The classifier must normalize so it is LOW/auto.
    pending = PendingAction(action_type="fetch_page", target="https://example.com")
    req = _classifier().classify_pending(pending)
    assert req.permission == PermissionLevel.AUTO


def test_classify_pending_write_confirms():
    pending = PendingAction(action_type="write_file", target="a.txt", content="hi")
    req = _classifier().classify_pending(pending)
    assert req.permission == PermissionLevel.CONFIRM


def test_classify_pending_sweeps_all_action_types():
    cl = _classifier()
    cases = [
        (PendingAction(action_type="read_file", target="a.txt"), PermissionLevel.AUTO),
        (PendingAction(action_type="write_file", target="a.txt", content="hi"), PermissionLevel.CONFIRM),
        (PendingAction(action_type="overwrite_file", target="a.txt", content="hi"), PermissionLevel.CONFIRM),
        (PendingAction(action_type="run_command", target="ls"), PermissionLevel.AUTO),
        (PendingAction(action_type="run_command", target="mkdir x"), PermissionLevel.CONFIRM),
        (PendingAction(action_type="web_search", target="python"), PermissionLevel.AUTO),
        (PendingAction(action_type="fetch_page", target="https://example.com"), PermissionLevel.AUTO),
        (PendingAction(action_type="db_query", target="SELECT * FROM users"), PermissionLevel.AUTO),
        (PendingAction(action_type="db_query", target="DROP TABLE users"), PermissionLevel.CONFIRM_CRITICAL),
    ]
    for pending, expected in cases:
        assert cl.classify_pending(pending).permission == expected, pending


def test_classify_pending_http_encoding():
    # main.py encodes HTTP requests as run_command with an http_request:
    # target; the classifier must decode them so GET auto-allows and
    # POST/PUT/DELETE confirm.
    cl = _classifier()
    get = PendingAction(
        action_type="run_command",
        target="http_request:GET:https://api.example.com/status",
    )
    assert cl.classify_pending(get).permission == PermissionLevel.AUTO

    post = PendingAction(
        action_type="run_command",
        target="http_request:POST:https://api.example.com/items",
    )
    assert cl.classify_pending(post).permission == PermissionLevel.CONFIRM

    delete = PendingAction(
        action_type="run_command",
        target="http_request:DELETE:https://api.example.com/items/1",
    )
    assert cl.classify_pending(delete).permission == PermissionLevel.CONFIRM


def test_request_labels_and_titles():
    req = _classifier().classify("write_file", "notes.txt", "hi")
    assert req.action_label == "Create new file"
    assert req.prompt_title == "Confirmation Required"

    critical = _classifier().classify("write_file", ".env", "x=1")
    assert critical.prompt_title == "⚠ Critical Action — Confirmation Required"


# ---------------------------------------------------------------------------
# confirm flow
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def test_confirm_auto_allow_never_prompts():
    calls = []

    async def ask(kind, message, choices=None):
        calls.append((kind, message))
        return "No, don't allow"

    req = _classifier().classify("read_file", "notes.txt")
    outcome = _run(confirm_action(req, ask=ask))
    assert outcome.approved is True
    assert outcome.was_asked is False
    assert calls == []  # never prompted


def test_confirm_deny_never_prompts():
    calls = []

    async def ask(kind, message, choices=None):
        calls.append(1)
        return "Yes, allow"

    req = _classifier().classify("write_file", "leak.txt", "AKIA1234567890ABCDEF")
    outcome = _run(confirm_action(req, ask=ask))
    assert outcome.approved is False
    assert outcome.was_asked is False
    assert calls == []  # hard block is never offered for confirmation


def test_confirm_select_yes():
    async def ask(kind, message, choices=None):
        assert kind == "select"
        return "Yes, allow"

    req = _classifier().classify("write_file", "notes.txt", "hi")
    outcome = _run(confirm_action(req, ask=ask))
    assert outcome.approved is True
    assert outcome.was_asked is True


def test_confirm_select_no():
    async def ask(kind, message, choices=None):
        return "No, don't allow"

    req = _classifier().classify("run_command", "mkdir newdir")
    outcome = _run(confirm_action(req, ask=ask))
    assert outcome.approved is False
    assert outcome.choice == "No, don't allow"


def test_confirm_critical_requires_typed_word():
    asked = []

    async def ask(kind, message, choices=None):
        asked.append(kind)
        return "CONFIRM"  # typed the magic word (case-insensitive)

    req = _classifier().classify("write_file", ".env", "x=1")
    outcome = _run(confirm_action(req, ask=ask))
    assert outcome.approved is True
    assert asked == ["text"]


def test_confirm_critical_rejects_without_word():
    async def ask(kind, message, choices=None):
        return "yes"  # wrong word

    req = _classifier().classify("write_file", ".env", "x=1")
    outcome = _run(confirm_action(req, ask=ask))
    assert outcome.approved is False


def test_confirm_critical_typed_step_can_be_disabled():
    async def ask(kind, message, choices=None):
        assert kind == "select"  # falls back to a plain Yes/No menu
        return "Yes, allow"

    req = _classifier().classify("write_file", ".env", "x=1")
    outcome = _run(
        confirm_action(req, ask=ask, typed_confirmation_for_critical=False)
    )
    assert outcome.approved is True


def test_confirm_render_called_with_request():
    rendered = []

    def render(request):
        rendered.append(request)

    async def ask(kind, message, choices=None):
        return "Yes, allow"

    req = _classifier().classify("run_command", "mkdir newdir")
    outcome = _run(confirm_action(req, ask=ask, render=render))
    assert outcome.approved is True
    assert len(rendered) == 1
    assert rendered[0] is req
