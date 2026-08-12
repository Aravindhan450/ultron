"""ultron.core.nlp.normalize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Command normalization for the natural-language → tool pipeline.

The core fix for the observed failures:

- ``Execute: pwd`` was passed to the shell verbatim
  (``/bin/sh: Execute:: command not found``)
- ``Run the command `pwd` `` produced a "which command?" clarification
  instead of running ``pwd``

:func:`normalize_terminal_command` strips the *outer* natural-language
wrapper (politeness, "the command", colons, backticks, trailing sentence
punctuation) and returns the actual command.  Inner content is preserved:

- ``echo "Execute: pwd"``      -> unchanged
- ``python -c 'print(...)'``   -> unchanged

A bare token only counts as a command when it is command-shaped (a known
command word, a path, or shell syntax), so conversational prose such as
``what is the current directory?`` is never sent to the shell.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Wrapper stripping
# ---------------------------------------------------------------------------

# Wrapper prefixes matched at the START of the string, most specific first.
# Each pattern is applied once; the longest match wins so "Run the command
# `pwd`" strips all three words instead of just "Run".
_WRAPPER_PATTERNS: tuple[str, ...] = (
    # "use the terminal tool to execute X" / "use the terminal and execute X"
    r"use\s+the\s+terminal(?:\s+tool)?\s+(?:to|and)\s+(?:run|execute)\s+",
    # "please run: X / please execute: X"
    r"please\s+(?:run|execute)\s*:\s*",
    # "please run / please execute X"
    r"please\s+(?:run|execute)\s+",
    # "can you / could you / would you run or execute X"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:run|execute)\s+",
    # "go ahead and run X"
    r"go\s+ahead\s+and\s+(?:run|execute)\s+",
    # "run/execute the (shell) command X"
    r"(?:run|execute)\s+the\s+(?:shell\s+)?command\s+",
    # "run/execute: X"
    r"(?:run|execute)\s*:\s*",
    # "run/execute X"
    r"(?:run|execute)\s+",
)

_KNOWN_COMMANDS: frozenset[str] = frozenset(
    {
        # navigation / inspection
        "pwd", "ls", "dir", "ll", "cd", "cat", "less", "more", "head", "tail",
        "wc", "find", "grep", "rg", "which", "whereis", "type", "tree",
        "du", "df", "stat", "file", "readlink",
        # git
        "git", "gitk",
        # files / edits
        "touch", "mkdir", "rmdir", "cp", "mv", "rm", "ln", "chmod", "chown",
        "echo", "printf", "tee", "sed", "awk", "sort", "uniq", "cut", "tr",
        "xargs", "basename", "dirname", "realpath",
        # processes / system
        "ps", "top", "htop", "kill", "pkill", "pgrep", "jobs", "fg", "bg",
        "nohup", "time", "date", "uptime", "uname", "whoami", "id", "env",
        "export", "hostname", "hostnamectl", "systemctl", "service",
        "free", "vmstat", "iostat", "lsof", "netstat", "ss",
        # package managers / toolchains
        "pip", "pip3", "python", "python3", "node", "npm", "npx", "yarn",
        "pnpm", "bun", "cargo", "rustc", "go", "make", "cmake", "gradle",
        "mvn", "docker", "docker-compose", "kubectl", "brew", "apt",
        "apt-get", "yum", "dnf", "pacman", "conda", "uv", "poetry",
        # test / lint / format
        "pytest", "ruff", "mypy", "black", "isort", "flake8", "eslint",
        "prettier", "tsc", "jest", "vitest", "mocha", "coverage",
        # network
        "curl", "wget", "ping", "ssh", "scp", "rsync", "nc", "telnet",
        "dig", "nslookup", "host",
        # misc
        "clear", "history", "man", "help", "true", "false", "yes", "sleep",
        "sh", "bash", "zsh", "openssl",
    }
)

# Shell syntax that makes a bare string unmistakably a command.
_SHELL_SYNTAX_RE = re.compile(r"[|&;<>`$]|\b(?:&&|\|\|)\b|\./|\.\./")

# Words that, when stripped of a wrapper, mean the remainder is prose, not a
# command ("Run the tests" must NOT become the command "the tests").
_PROSE_START = frozenset({"the", "a", "an", "it", "this", "that", "these",
                          "those", "my", "your", "our", "their", "all", "any",
                          "some", "me", "us", "them", "him", "her", "its"})

# Question/description structure that makes a bare string prose, not a command
# ("find where pwd is used", "how does git work", "what is the current
# directory") — the spec's "do not confuse description with command".
_PROSE_GUARD_RE = re.compile(
    r"\b(?:what|which|who|whose|whom|when|where|why|how)\b"
    r"|\btell\s+me\b"
    r"|\b(?:explain|describe)\b",
    re.IGNORECASE,
)


def _looks_like_command(text: str) -> bool:
    """True when *text* is command-shaped (known word, path, or shell syntax)."""
    stripped = text.strip().strip("`\"'")
    if not stripped:
        return False
    if _SHELL_SYNTAX_RE.search(stripped):
        return True
    if stripped.startswith(("./", "../", "/", "~/")):
        return True
    first = stripped.split(None, 1)[0].strip("`\"'")
    return first.lower() in _KNOWN_COMMANDS


def _unwrap_whole(text: str) -> str:
    """Strips backticks/quotes that wrap the ENTIRE command.

    ```pwd``` -> pwd  and  "pwd" -> pwd.  Quotes that are part of the command
    (``echo "hi"``) are left untouched because they do not wrap the whole
    string.
    """
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("`", '"', "'"):
        return t[1:-1].strip()
    return t


def _strip_sentence_tail(text: str) -> str:
    """Strips trailing sentence punctuation (?, !, or a trailing period).

    A trailing period is only stripped when it is attached to the command
    ("pwd." -> pwd) and never when preceded by whitespace ("ls -la ." keeps
    its argument).
    """
    t = text.rstrip()
    if not t:
        return t
    if t[-1] in "?!":
        return _strip_sentence_tail(t[:-1].rstrip())
    if t[-1] == "." and len(t) >= 2 and not t[-2].isspace():
        return _strip_sentence_tail(t[:-1].rstrip())
    return t


def normalize_terminal_command(text: str) -> str | None:
    """
    Strips the natural-language wrapper from a terminal request and returns
    the actual command, or None when *text* is not (clearly) a command.

    Handled forms (case-insensitive, wrapper only at the start):

    - ``Execute: pwd`` / ``Run: git status``
    - ``Run the command `pwd` `` / ``Execute the command `python -m pytest` ``
    - ``Please run `pytest tests/` ``
    - ``Can you execute pwd?`` / ``Could you run pwd``
    - ``Use the terminal (tool) to run pwd``
    - bare command-shaped text: ``pwd``, ``git status``, ``ls -la .``

    Never modified: ``echo "Execute: pwd"``, ``python -c 'print(...)'``.
    """
    if not text or not text.strip():
        return None

    stripped = text.strip()

    # Try the wrapper prefixes (longest match wins).
    for pattern in _WRAPPER_PATTERNS:
        m = re.match(pattern, stripped, re.IGNORECASE)
        if not m:
            continue
        remainder = stripped[m.end():].strip()
        remainder = _unwrap_whole(remainder)
        remainder = _strip_sentence_tail(remainder)
        if not remainder:
            return None
        # "Run the tests" -> remainder "the tests" is prose, not a command.
        first = remainder.split(None, 1)[0].strip("`\"'").lower() if remainder else ""
        if first in _PROSE_START and not _looks_like_command(remainder):
            return None
        # "Run how does git work" -> prose question, not a command.
        if _PROSE_GUARD_RE.search(remainder):
            return None
        return remainder

    # No wrapper: accept only command-shaped bare text that is not prose.
    if _looks_like_command(stripped) and not _PROSE_GUARD_RE.search(stripped):
        return stripped
    return None


def detect_explicit_test_command(text: str) -> str | None:
    """
    Extracts a concrete test command WITH a target from natural language:

    - ``Run pytest tests/test_api.py``        -> ``pytest tests/test_api.py``
    - ``Run python -m pytest tests/ -q``      -> ``python -m pytest tests/ -q``

    Returns the exact command when a test runner + target path is present,
    otherwise None (a generic ``run the tests`` stays with the test-intent
    routing).
    """
    if not text or not text.strip():
        return None
    m = re.match(
        r"^\s*(?:please\s+)?(?:run|execute)\s+"
        r"(?P<runner>python\s+-m\s+pytest|npm\s+test|cargo\s+test|"
        r"go\s+test|pytest)\s+"
        r"(?P<target>.+?)\s*[.!?]?\s*$",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    target = m.group("target").strip()
    if not target or target.startswith("-"):
        return None
    return f"{m.group('runner')} {target}".strip()
