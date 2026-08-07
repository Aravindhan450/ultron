"""
ultron.permissions.confirm
~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive confirmation flow for the permission layer.

Takes a :class:`~ultron.permissions.classifier.PermissionRequest` (from the
classifier) and:

- ``AUTO``             → approved immediately, never prompts
- ``DENY``             → rejected immediately, never prompts (hard block)
- ``CONFIRM``          → renders a confirmation card and asks Yes/No
- ``CONFIRM_CRITICAL`` → renders a warning card and requires the user to
  type ``confirm`` (the extra-verification step for CRITICAL actions)

The prompt and card renderers are injectable so the flow is fully testable
without a TTY; by default it uses questionary + the Rich UI theme.

Every approval and denial is logged, honoring the docs' promise that the
user always has the final say and every decision is recorded.
"""

from dataclasses import dataclass

from ultron.core.logging import get_logger
from ultron.permissions.classifier import PermissionLevel, PermissionRequest
from ultron.ui.theme import UI

logger = get_logger("ultron.permissions.confirm")

YES = "Yes, allow"
NO = "No, don't allow"

# Exact word a user must type to allow a CRITICAL action (case-insensitive).
CRITICAL_CONFIRM_WORD = "confirm"

# Action types whose confirmation card shows a content preview.
_PREVIEW_ACTIONS = {"write_file", "overwrite_file"}


@dataclass
class ConfirmationOutcome:
    """
    Result of an interactive confirmation.

    Attributes:
        approved: True when the action may proceed.
        request: The PermissionRequest that was confirmed.
        choice: The raw user answer (``None`` when no prompt was shown —
            i.e. the action was auto-allowed or hard-blocked).
    """

    approved: bool
    request: PermissionRequest
    choice: str | None = None

    @property
    def was_asked(self) -> bool:
        """True when the user was actually prompted."""
        return self.choice is not None


async def _default_ask(kind: str, message: str, choices=None) -> str | None:
    """
    Default prompt implementation backed by questionary.

    ``kind`` is ``"select"`` (a menu) or ``"text"`` (free-text input used for
    the CRITICAL typed confirmation).
    """
    import questionary

    if kind == "text":
        return await questionary.text(message).ask_async()
    return await questionary.select(message, choices=choices).ask_async()


def _default_render(request: PermissionRequest) -> None:
    """
    Renders the confirmation card through the shared Rich UI theme.
    """
    preview = (request.content or "")[:120]
    if len(request.content or "") > 120:
        preview += "…"
    UI.render_action_card(
        title=request.prompt_title,
        action=request.action_label,
        target=request.target,
        preview=preview if request.action_type in _PREVIEW_ACTIONS else "",
    )


async def confirm_action(
    request: PermissionRequest,
    *,
    ask=None,
    render=None,
    typed_confirmation_for_critical: bool = True,
) -> ConfirmationOutcome:
    """
    Confirms a PermissionRequest interactively and returns the outcome.

    Only ``CONFIRM`` / ``CONFIRM_CRITICAL`` requests prompt the user;
    auto-allowed and hard-blocked actions short-circuit without a prompt.

    Args:
        request: The classified action to confirm.
        ask: Optional async ``(kind, message, choices) -> answer`` prompt
            implementation (injectable for tests).
        render: Optional ``(request) -> None`` card renderer.
        typed_confirmation_for_critical: When True (default), CRITICAL
            actions require typing the word ``confirm`` instead of a Yes/No
            menu.
    """
    ask = ask or _default_ask
    render = render or _default_render

    # Auto-allowed actions never prompt.
    if request.permission == PermissionLevel.AUTO:
        return ConfirmationOutcome(approved=True, request=request)

    # Hard-blocked actions are never offered for confirmation.
    if request.permission == PermissionLevel.DENY:
        logger.warning(
            "action=%s denied before confirmation tier=%s reason=%s",
            request.action_type,
            request.tier.value,
            request.reason,
        )
        return ConfirmationOutcome(approved=False, request=request)

    render(request)

    choice: str | None = None
    if (
        request.permission == PermissionLevel.CONFIRM_CRITICAL
        and typed_confirmation_for_critical
    ):
        typed = await ask(
            "text",
            f"⚠ CRITICAL: type '{CRITICAL_CONFIRM_WORD}' to allow this action",
            None,
        )
        approved = (typed or "").strip().lower() == CRITICAL_CONFIRM_WORD
        choice = "typed-confirm" if approved else "typed-reject"
    else:
        choice = await ask("select", "Do you want to allow this action?", [YES, NO])
        approved = choice == YES

    if approved:
        logger.info(
            "action=%s approved tier=%s target=%s",
            request.action_type,
            request.tier.value,
            request.target,
        )
    else:
        logger.warning(
            "action=%s rejected tier=%s target=%s",
            request.action_type,
            request.tier.value,
            request.target,
        )
    return ConfirmationOutcome(approved=approved, request=request, choice=choice)
