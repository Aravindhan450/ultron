"""
ultron.security.scanners.secret
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regex-based detection of common credential patterns.

The goal is not to be exhaustive — it is to catch the credentials that
actually show up in everyday code and shell output (cloud keys, API tokens,
private keys, JWTs, and generic ``key=value`` assignments). Everything here is
heuristic: a match is a *finding* for the guardrails layer to act on, never a
guarantee that the text is a real secret.

Snippets are truncated before being surfaced so findings themselves never leak
more of a secret than necessary.
"""

import re
from collections.abc import Sequence

from ultron.security.models import GuardrailFinding

_SNIPPET_LEN = 40

# Each rule: (rule name, compiled regex, severity, short explanation).
_SECRET_RULES: list[tuple[str, re.Pattern[str], str, str]] = [
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "critical",
        "AWS access key ID detected",
    ),
    (
        "github_token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs)_[A-Za-z0-9]{20,}\b"),
        "critical",
        "GitHub personal access token detected",
    ),
    (
        "openai_api_key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
        "critical",
        "OpenAI-style API key detected",
    ),
    (
        "anthropic_api_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
        "critical",
        "Anthropic-style API key detected",
    ),
    (
        "google_api_key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "critical",
        "Google API key detected",
    ),
    (
        "stripe_key",
        re.compile(r"\bsk_live_[0-9a-zA-Z]{20,}\b"),
        "critical",
        "Stripe live secret key detected",
    ),
    (
        "slack_token",
        re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
        "critical",
        "Slack token detected",
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----"),
        "critical",
        "Private key material detected",
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "critical",
        "JWT (JSON Web Token) detected",
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bauthorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]+"),
        "critical",
        "Authorization bearer token detected",
    ),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*['\"]?[^\s'\"]{6,}"
        ),
        "warning",
        "Possible credential assignment (e.g. password=...) detected",
    ),
]


def scan_secrets(text: str) -> list[GuardrailFinding]:
    """
    Scans *text* for credential patterns.

    Returns a list of GuardrailFindings (one per rule that matched). Each
    finding carries a truncated snippet so no full secret is echoed back.
    """
    if not text:
        return []

    findings: list[GuardrailFinding] = []
    for rule, pattern, severity, message in _SECRET_RULES:
        for match in pattern.finditer(text):
            findings.append(
                GuardrailFinding(
                    rule=rule,
                    severity=severity,
                    location="content",
                    snippet=match.group(0)[:_SNIPPET_LEN],
                    message=message,
                )
            )
    return findings


def mask_matches(text: str, findings: Sequence[GuardrailFinding]) -> str:
    """
    Replaces every matched snippet in *text* with asterisks.

    Used by the guardrails layer to produce a redacted copy of content that
    contained secrets or PII.
    """
    masked = text
    for finding in findings:
        if finding.snippet:
            masked = masked.replace(finding.snippet, "*" * len(finding.snippet))
    return masked
