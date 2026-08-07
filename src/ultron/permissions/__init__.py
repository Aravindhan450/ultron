"""
ultron.permissions
~~~~~~~~~~~~~~~~~~

Permission & Approval layer: turns the security boundary's risk verdict into
an interactive confirmation flow.

    from ultron.permissions import PermissionClassifier, confirm_action

    classifier = PermissionClassifier()
    request = classifier.classify("run_command", "rm -rf /")
    request.permission  # PermissionLevel.CONFIRM_CRITICAL
    outcome = await confirm_action(request)   # prompts only when needed
"""

from ultron.permissions.classifier import (
    PermissionClassifier,
    PermissionLevel,
    PermissionRequest,
)
from ultron.permissions.confirm import ConfirmationOutcome, confirm_action

__all__ = [
    "ConfirmationOutcome",
    "PermissionClassifier",
    "PermissionLevel",
    "PermissionRequest",
    "confirm_action",
]
