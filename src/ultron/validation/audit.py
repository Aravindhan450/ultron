"""Anti-hardcoding audit (Phase 16 of STEP 3).

The validation framework must check that production code is not memorizing
test requests or historical symbols.  This audit is a static scan: it parses
Python files with ``ast`` and looks for *executable string literals* that
match historical diagnostic prompts or project entity names used as routing
keys.

Comments and docstrings are ignored (AST walk skips docstring nodes), so
legitimate documentation examples never produce findings.  The audit only
reports; it never changes production code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# Historical diagnostic entities from the STEP 1-2C cycles.  These are only
# *audit targets* — the audit flags their presence as executable literals; it
# is not validation logic that depends on them.
HISTORICAL_ENTITIES: tuple[str, ...] = (
    "TaskState",
    "Supervisor",
    "CodingExecutor",
    "OrchestrationValidator",
)

# Historical diagnostic prompts (from the STEP 1-2C cycles).
HISTORICAL_PROMPTS: tuple[str, ...] = (
    "Find where TaskState is defined",
    "Where is TaskState used",
    "Find references to TaskState",
    "How does the Supervisor delegate work",
    "Where is CodingExecutor implemented",
    "Run the relevant tests",
    "Run the full test suite",
    "List the files in the current directory",
    "Execute: git status",
    "Run pwd",
    "Show me the current git diff",
)

# Suspicious routing patterns: comparing the raw user input against a fixed
# literal is how a system memorizes test requests.
_SUSPICIOUS_COMPARE = re.compile(
    r"(?:request|query|prompt|user_input|text|command|message)\s*[=!]=\s*['\"]"
)

# Routing layers that must never contain executable historical literals,
# expressed relative to the ultron package root.
_ROUTING_GLOBS = (
    "core/agents/*.py",
    "core/nlp/*.py",
    "core/capabilities/*.py",
    "core/tools/*.py",
    "security/*.py",
    "permissions/*.py",
)

# Production roots scanned by the audit.  The validation framework itself
# (ultron/validation) is deliberately excluded: it is the observer, and its
# task templates are generated test data — not production routing.
_PRODUCTION_SUBDIRS = ("core", "security", "permissions")


@dataclass
class AuditFinding:
    file: str
    line: int
    kind: str  # "historical_prompt" | "historical_entity" | "input_literal_compare"
    detail: str
    severity: str = "warning"


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    files_scanned: int = 0

    @property
    def count(self) -> int:
        return len(self.findings)

    def critical(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "critical"]


def _is_docstring(node: ast.AST) -> bool:
    parent = getattr(node, "_ultron_parent", None)
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and isinstance(parent, ast.Expr)
        and isinstance(getattr(parent, "value", None), ast.Constant)
    )


def _scan_file(path: Path, routing_only: bool) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return findings
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._ultron_parent = node
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if _is_docstring(node):
            continue
        text = node.value
        lowered = text.lower()
        for prompt in HISTORICAL_PROMPTS:
            if prompt.lower() in lowered:
                findings.append(
                    AuditFinding(
                        file=str(path),
                        line=getattr(node, "lineno", 0),
                        kind="historical_prompt",
                        detail=f"executable literal contains historical prompt: {prompt!r}",
                        severity="critical",
                    )
                )
        if routing_only:
            for entity in HISTORICAL_ENTITIES:
                if entity.lower() in lowered:
                    findings.append(
                        AuditFinding(
                            file=str(path),
                            line=getattr(node, "lineno", 0),
                            kind="historical_entity",
                            detail=f"executable literal references project entity {entity!r} in routing layer",
                        )
                    )
    return findings


def _package_root(root: Path) -> Path | None:
    """Locates the ultron package root under repo root / src/ root."""
    if (root / "src" / "ultron").is_dir():
        return root / "src" / "ultron"
    if (root / "ultron").is_dir():
        return root / "ultron"
    return None


def audit_production(root: str | Path = "src") -> AuditReport:
    """Scans production Python for hardcoded test-memorization patterns.

    Scans the routing/security layers (core, security, permissions) only;
    the validation framework itself is the observer and is not scanned.
    ``root`` may be the repository root or the ``src/`` directory.
    """
    root = Path(root)
    pkg = _package_root(root)
    report = AuditReport()
    if pkg is None:
        return report
    py_files: list[Path] = []
    for sub in _PRODUCTION_SUBDIRS:
        base = pkg / sub
        if base.is_dir():
            py_files.extend(sorted(base.rglob("*.py")))
    py_files = sorted(set(py_files))
    report.files_scanned = len(py_files)
    for path in py_files:
        rel = str(path.relative_to(pkg))
        routing_only = any(re.fullmatch(glob, rel) for glob in _ROUTING_GLOBS)
        report.findings.extend(_scan_file(path, routing_only=routing_only))
    report.findings.sort(key=lambda f: (f.file, f.line))
    return report
