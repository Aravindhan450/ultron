"""
ultron.security.scanners.pii
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regex-based detection of common personal data patterns: email addresses,
phone numbers, US Social Security numbers, and credit-card-shaped numbers.

Like the secret scanner, this is heuristic. A match is a finding for the
guardrails layer — the intent is to flag content that *looks* personal so it
can be redacted or reviewed, not to make identity-law judgements.
"""

import re

from ultron.security.models import GuardrailFinding

_SNIPPET_LEN = 40

_PII_RULES: list[tuple[str, re.Pattern[str], str, str]] = [
    (
        "email_address",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "warning",
        "Email address detected",
    ),
    (
        "phone_number",
        re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
        "warning",
        "Phone number detected",
    ),
    (
        "us_ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "critical",
        "US Social Security number detected",
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "critical",
        "Credit-card-shaped number detected",
    ),
]


def scan_pii(text: str) -> list[GuardrailFinding]:
    """
    Scans *text* for personal-data patterns.

    Returns one GuardrailFinding per rule that matched, with a truncated
    snippet. The credit-card rule is deliberately conservative (digits only)
    and does not apply a Luhn check — it is a shape heuristic.
    """
    if not text:
        return []

    findings: list[GuardrailFinding] = []
    for rule, pattern, severity, message in _PII_RULES:
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


