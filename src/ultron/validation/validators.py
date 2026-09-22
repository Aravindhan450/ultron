"""Generic scenario-specific outcome validators for Acceptance Criteria.

Deterministic validation functions that test real observable results:
- assert_file_exists
- assert_file_contains
- assert_command_success
- launch_application
- validate_functional_operation
- validate_verification_evidence
"""

from __future__ import annotations

import re
from pathlib import Path

from ultron.validation.model import (
    AcceptanceCriterion,
    EnvironmentSnapshot,
    Verdict,
    VerificationEvidence,
)
from ultron.validation.probes import (
    CommandProcessProbe,
    FilesystemProbe,
    WorkspaceProbe,
)


def validate_workspace_confinement_criterion(
    criterion: AcceptanceCriterion,
    environment: EnvironmentSnapshot | None,
    workspace_root: str | Path | None = None,
) -> AcceptanceCriterion:
    """Validates that project files are created strictly within the configured workspace."""
    probe = WorkspaceProbe(workspace_root or (environment.workspace_root if environment else None))
    paths = []
    if environment:
        paths.extend(environment.files_created)
        paths.extend(environment.files_modified)

    is_confined, escaped = probe.check_confinement(paths)
    if not is_confined:
        criterion.status = Verdict.FAIL
        criterion.observed_result = False
        criterion.error = f"Artifacts escaped workspace: {escaped}"
        criterion.details = {"escaped": escaped, "workspace": str(probe.workspace_root)}
    else:
        criterion.status = Verdict.PASS
        criterion.observed_result = True
        criterion.details = {"workspace": str(probe.workspace_root), "file_count": len(paths)}
    return criterion


def validate_artifact_exists_criterion(
    criterion: AcceptanceCriterion,
    workspace_root: str | Path,
    expected_filenames: list[str] | None = None,
) -> AcceptanceCriterion:
    """Validates that expected project artifacts exist on disk."""
    root = Path(workspace_root).resolve()
    filenames = expected_filenames or criterion.details.get("filenames", [])
    
    found: list[str] = []
    missing: list[str] = []
    
    # Also search subdirectories of workspace
    for name in filenames:
        if (root / name).exists():
            found.append(name)
        else:
            # Check if anywhere in tree
            matches = list(root.glob(f"**/{name}"))
            if matches:
                found.append(str(matches[0].relative_to(root)))
            else:
                missing.append(name)

    if missing:
        criterion.status = Verdict.FAIL
        criterion.observed_result = False
        criterion.error = f"Missing required artifacts: {missing}"
        criterion.details = {"found": found, "missing": missing}
    else:
        criterion.status = Verdict.PASS
        criterion.observed_result = True
        criterion.details = {"found": found}
    return criterion


def validate_dependency_validity_criterion(
    criterion: AcceptanceCriterion,
    environment: EnvironmentSnapshot | None,
) -> AcceptanceCriterion:
    """Validates that dependency declarations do not include invalid or stdlib modules."""
    from ultron.validation.invariants import check_invalid_dependency_declarations
    
    files = (environment.files_created + environment.files_modified) if environment else None
    commands = environment.commands_executed if environment else None
    res = check_invalid_dependency_declarations(files=files, commands=commands)
    
    if not res.passed:
        criterion.status = Verdict.FAIL
        criterion.observed_result = False
        criterion.error = res.failure_reason
        criterion.details = res.evidence
    else:
        criterion.status = Verdict.PASS
        criterion.observed_result = True
    return criterion


def validate_application_launch_criterion(
    criterion: AcceptanceCriterion,
    workspace_root: str | Path,
    entrypoint: str | None = None,
    timeout: float = 3.0,
) -> AcceptanceCriterion:
    """Tests if the application entrypoint can be launched without immediately crashing."""
    root = Path(workspace_root).resolve()
    
    target_script: Path | None = None
    if entrypoint and (root / entrypoint).exists():
        target_script = root / entrypoint
    else:
        # Search for entrypoints
        candidates = list(root.glob("*.py")) + list(root.glob("src/*.py"))
        for c in candidates:
            content = FilesystemProbe.read_text(c) or ""
            if "tkinter" in content.lower() or "app" in c.name.lower() or "main" in c.name.lower():
                target_script = c
                break
        if not target_script and candidates:
            target_script = candidates[0]

    if not target_script or not target_script.exists():
        criterion.status = Verdict.FAIL
        criterion.observed_result = False
        criterion.error = "No launchable application entrypoint file found."
        return criterion

    # Probe launch using python syntax/import compilation probe or brief subprocess run
    probe_code = (
        f"import py_compile; py_compile.compile('{target_script}', doraise=True); "
        f"print('SYNTAX_COMPILED_OK')"
    )
    code, out, err = CommandProcessProbe.execute([f"{sys_python()}", "-c", probe_code], cwd=root, timeout=timeout)
    if code != 0:
        criterion.status = Verdict.FAIL
        criterion.observed_result = False
        criterion.error = f"Application entrypoint failed compilation/syntax check: {err}"
        criterion.details = {"entrypoint": str(target_script), "error": err}
        return criterion

    criterion.status = Verdict.PASS
    criterion.observed_result = True
    criterion.details = {"entrypoint": str(target_script), "stdout": out.strip()}
    return criterion


def validate_functional_operation_criterion(
    criterion: AcceptanceCriterion,
    workspace_root: str | Path,
    operation_check: str | None = None,
) -> AcceptanceCriterion:
    """Checks functional implementation details (e.g. SQLite database logic, CSV export)."""
    root = Path(workspace_root).resolve()
    py_files = list(root.glob("**/*.py"))
    
    all_content = ""
    for f in py_files:
        content = FilesystemProbe.read_text(f)
        if content:
            all_content += "\n" + content

    required_keywords = criterion.details.get("required_keywords", ["sqlite3", "csv"])
    missing_ops = [kw for kw in required_keywords if kw not in all_content.lower()]
    
    if missing_ops:
        criterion.status = Verdict.FAIL
        criterion.observed_result = False
        criterion.error = f"Missing required functional implementations: {missing_ops}"
        criterion.details = {"missing": missing_ops}
    else:
        criterion.status = Verdict.PASS
        criterion.observed_result = True
        criterion.details = {"verified_keywords": required_keywords}
    return criterion


def validate_verification_evidence_criterion(
    criterion: AcceptanceCriterion,
    evidence: VerificationEvidence | None,
    transcript: str = "",
) -> AcceptanceCriterion:
    """Validates that Ultron did NOT claim completion without independent verification."""
    model_claimed = False
    if evidence:
        model_claimed = evidence.model_claimed_success
    else:
        model_claimed = bool(
            re.search(r"\b(completed|done|working application|successfully built)\b", transcript, re.IGNORECASE)
        )

    has_independent_evidence = False
    if evidence and evidence.independent_evidence_produced:
        has_independent_evidence = True
    else:
        # Check if actual verification commands or test executions happened in transcript
        if re.search(r"\b(pytest|python -m unittest|testing workflow|verified)\b", transcript, re.IGNORECASE):
            has_independent_evidence = True

    if model_claimed and not has_independent_evidence:
        criterion.status = Verdict.FAIL
        criterion.observed_result = False
        criterion.error = "Model declared completion without independent verification evidence."
        criterion.details = {"claimed": model_claimed, "independent_evidence": has_independent_evidence}
    else:
        criterion.status = Verdict.PASS
        criterion.observed_result = True
        criterion.details = {"verified": has_independent_evidence}
    return criterion


def sys_python() -> str:
    """Returns absolute path to current python interpreter."""
    import sys
    return sys.executable
