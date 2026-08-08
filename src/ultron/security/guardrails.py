"""
ultron.security.guardrails
~~~~~~~~~~~~~~~~~~~~~~~~~~

GuardrailsEngine — the scanning half of the security boundary.

It sits *before* execution of any tool call and answers one question: does
this action look like it should be blocked outright?

What gets a hard block (``blocked=True``):
- Content that would carry a secret credential out of the machine
  (file writes, memory writes, HTTP bodies, command strings).
- Requests to non-https / non-localhost URLs.
- File targets outside the directory Ultron was launched from.

What gets flagged but NOT blocked:
- PII in content (email, phone, SSN, credit-card shape) → warning finding
  plus a redacted ``sanitized_text`` suggestion.
- Dangerous shell patterns (``rm -rf /``, ``curl | sh``, fork bombs, ...)
  → critical finding. The boundary uses this to escalate the risk tier; the
  user still keeps the final say per Ultron's "permission first" model.

URL and path checks deliberately reuse the existing implementations in
``core.tools.builtin.http_client`` and ``core.tools.paths`` so policy stays
in one place.
"""

import re

from ultron.core.tools.builtin.http_client import check_url_safety
from ultron.security.file_policy import get_file_policy
from ultron.security.models import GuardrailFinding, GuardrailsResult
from ultron.security.scanners.pii import scan_pii
from ultron.security.scanners.secret import mask_matches, scan_secrets

# ---------------------------------------------------------------------------
# Dangerous shell patterns
# ---------------------------------------------------------------------------

# Each entry: (rule name, compiled regex). Used to escalate commands that can
# destroy data or affect the whole system to CRITICAL tier.
DANGEROUS_COMMAND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("destructive_rm", re.compile(r"\brm\s+-[a-z]*r[a-z]*(?:f[a-z]*)?\s+(?:--\s*)?(?:~/?|/+)", re.IGNORECASE)),
    ("mkfs", re.compile(r"\bmkfs(?:\.\w+)?\b", re.IGNORECASE)),
    ("dd_to_device", re.compile(r"\bdd\b[^\n|]*\bof=/dev/", re.IGNORECASE)),
    ("fork_bomb", re.compile(r":\(\)\s*\{\s*:\|:&\s*\};", re.IGNORECASE)),
    ("pipe_to_shell", re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba)?sh\b", re.IGNORECASE)),
    ("chmod_recursive_root", re.compile(r"\bchmod\s+-R\s+777\s+~?/?\s*$", re.IGNORECASE)),
    ("overwrite_block_device", re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE)),
    ("shutdown_command", re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.IGNORECASE)),
    ("init_runlevel", re.compile(r"\binit\s+[06]\b", re.IGNORECASE)),
    ("forced_push", re.compile(r"\bgit\s+push\b[^\n|]*--force", re.IGNORECASE)),
    ("disk_wipe", re.compile(r"\b(?:wipefs|sfdisk|fdisk)\b", re.IGNORECASE)),
]


class GuardrailsEngine:
    """
    Stateless scanner that evaluates one tool action at a time.

    Usage:
        engine = GuardrailsEngine()
        result = engine.evaluate(action_type="write_file", target="notes.txt",
                                 content="hello")
        if result.blocked:
            print(result.block_reason)
    """

    def check_command(self, command: str) -> GuardrailFinding | None:
        """
        Returns a critical finding if *command* matches a dangerous pattern.
        """
        if not command:
            return None
        for rule, pattern in DANGEROUS_COMMAND_PATTERNS:
            match = pattern.search(command)
            if match:
                return GuardrailFinding(
                    rule=rule,
                    severity="critical",
                    location="command",
                    snippet=match.group(0)[:60],
                    message=f"Dangerous command pattern '{rule}' detected",
                )
        return None

    def check_url(self, url: str) -> GuardrailFinding | None:
        """
        Returns a blocking finding for non-https / non-localhost URLs.
        """
        clean = (url or "").strip()
        if not clean:
            return GuardrailFinding(
                rule="empty_url",
                severity="critical",
                location="url",
                message="URL target is empty",
            )
        error = check_url_safety(clean)
        if error:
            return GuardrailFinding(
                rule="unsafe_url",
                severity="critical",
                location="url",
                snippet=clean[:60],
                message=error,
            )
        return None

    def check_path(self, path: str) -> GuardrailFinding | None:
        """
        Returns a blocking finding when *path* escapes the allowed base dir.

        The confinement check is delegated to the shared file policy
        (``FilePolicy.is_path_safe``), which keeps the same contract as the
        underlying ``core.tools.paths.is_path_safe`` — including honoring a
        patched ``ALLOWED_BASE_DIR`` at call time.
        """
        if not path or not str(path).strip():
            return GuardrailFinding(
                rule="empty_path",
                severity="critical",
                location="path",
                message="File target is empty",
            )
        is_safe, _resolved = get_file_policy().is_path_safe(path)
        if not is_safe:
            return GuardrailFinding(
                rule="path_escape",
                severity="critical",
                location="path",
                snippet=str(path)[:60],
                message="Target path is outside the allowed working directory",
            )
        return None

    def evaluate(
        self,
        *,
        action_type: str,
        target: str,
        content: str | None = None,
    ) -> GuardrailsResult:
        """
        Runs every applicable check for one action and returns a verdict.

        Returns a GuardrailsResult; ``blocked`` is True when the action must
        be denied, in which case ``block_reason`` explains why.
        """
        findings: list[GuardrailFinding] = []
        blocked = False
        block_reason: str | None = None
        sanitized: str | None = None

        # --- Content: secrets and PII -----------------------------------
        if content:
            secret_hits = scan_secrets(content)
            pii_hits = scan_pii(content)
            findings.extend(secret_hits)
            findings.extend(pii_hits)
            if secret_hits:
                blocked = True
                block_reason = (
                    f"Outgoing content contains what looks like a credential "
                    f"(rule: {secret_hits[0].rule})"
                )
            if secret_hits or pii_hits:
                sanitized = mask_matches(content, [*secret_hits, *pii_hits])

        # --- Command targets --------------------------------------------
        # A run_parallel batch carries its commands newline-joined in the
        # target; every command is scanned individually so one dangerous or
        # credential-bearing command denies the whole batch.
        if action_type == "run_command":
            command_segments = [target or ""]
        elif action_type == "run_parallel":
            command_segments = [
                seg for seg in (target or "").splitlines() if seg.strip()
            ]
        else:
            command_segments = []
        for segment in command_segments:
            danger = self.check_command(segment)
            if danger:
                findings.append(danger)
            # Commands that would carry a credential out of the machine are
            # denied like any other secret-bearing content (see module
            # docstring). This covers e.g. `grep <aws-key> file` or
            # `curl ... -d '{"token": ...}'` even though callers pass the
            # command as the *target* rather than as content.
            command_secrets = scan_secrets(segment)
            if command_secrets:
                findings.extend(command_secrets)
                blocked = True
                block_reason = block_reason or (
                    f"Command string contains what looks like a credential "
                    f"(rule: {command_secrets[0].rule})"
                )

        # --- URL targets ------------------------------------------------
        # Network actions get the URL safety scan. learn_api_schema fetches an
        # OpenAPI document, so it is scanned like any other outbound request.
        # The other schema tools (api_usage_hint / get_api_knowledge /
        # forget_api) only read/write the LOCAL knowledge store — they never
        # touch the network, so a host key is not a URL to police.
        if action_type in {
            "make_http_request",
            "fetch_page_text",
            "check_connectivity",
            "learn_api_schema",
        }:
            url = _extract_url(target)
            bad_url = self.check_url(url)
            if bad_url:
                findings.append(bad_url)
                blocked = True
                block_reason = block_reason or bad_url.message
        elif action_type == "retrieve":
            # The orchestrator also carries bare search queries (no URL). Only
            # enforce URL safety when the target actually contains a URL — a
            # plain search query is not a URL and must not be blocked as one.
            if re.search(r"https?://\S+", target or ""):
                url = _extract_url(target)
                bad_url = self.check_url(url)
                if bad_url:
                    findings.append(bad_url)
                    blocked = True
                    block_reason = block_reason or bad_url.message

        # --- File targets -----------------------------------------------
        if action_type in {"read_file", "write_file", "overwrite_file"}:
            bad_path = self.check_path(target)
            if bad_path:
                findings.append(bad_path)
                blocked = True
                block_reason = block_reason or bad_path.message

        return GuardrailsResult(
            passed=not blocked,
            blocked=blocked,
            findings=findings,
            sanitized_text=sanitized,
            block_reason=block_reason,
        )


def _extract_url(target: str) -> str:
    """
    Pulls the URL out of a target string.

    The CLI's pending-action flow encodes HTTP requests as
    ``http_request:GET:https://...[:body]``. We can't split on ``:`` because
    URLs contain colons themselves (``https://host:8443/path``), so we locate
    the URL with a regex instead and discard any trailing body suffix.
    """
    text = (target or "").strip()
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else text
