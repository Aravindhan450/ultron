"""Three-layer capability evaluation (Parts 7-13 of STEP 3.1).

The single evaluation path is replaced by three INDEPENDENT layers:

    A. CAPABILITY TRUTH — what the task actually requires (task metadata,
       contract, repository subject).  Never inferred from the model.
    B. EXECUTION / EVIDENCE TRUTH — did the agent perform an appropriate
       operation and retrieve VALID evidence?  Tool names, arguments,
       security trace, and — where deterministically available — the actual
       repository (file existence, file content, directory contents).
    C. FINAL ANSWER TRUTH — relevance, factual grounding, completeness,
       uncertainty calibration of the final response.

Markers (file:line, "definition", "likely", tool-name strings) are evidence
SIGNALS for evaluation, never the evaluation itself: a claimed location is
only PASS when it verifies against repository state, and speculative
phrasing on a definitive claim is a failure.

Overall verdict (Part 11, explicit):

    any layer FAIL                                -> FAIL
    capability PASS and execution PASS
        and answer PASS                           -> PASS
    at least one layer PASS, others UNRESOLVED    -> PARTIAL
    no layer PASS                                 -> UNRESOLVED

Failure attribution (Part 13) uses the (expected, deterministic-router,
model) triple recorded on the case and trace.
"""

from __future__ import annotations

import re
from pathlib import Path

from ultron.core.tools.definitions import tools_with_capability
from ultron.validation.generator import symbol_variants
from ultron.validation.model import (
    CapabilityTestCase,
    Evaluation,
    FailureKind,
    TaskTrace,
    Verdict,
)
from ultron.validation.runner import _REPLY_WINDOW, ParsedTrace, parse_trace

# Project root used for repository verification when no explicit root is given.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Fine-grained detail dimensions (evidence for the report; the verdict comes
# from the three layers).
DIMENSIONS = (
    "intent",
    "capability",
    "tool_selection",
    "argument",
    "security",
    "evidence",
    "investigation",
    "final_answer",
)

# Capabilities whose evidence is a *verified repository claim*.
_REPO_CLAIM_CAPS = {
    "definition_lookup",
    "reference_lookup",
    "symbol_search",
    "symbol_inspection",
    "semantic_search",
    "repository_inspection",
    "repository_investigation",
    "dependency_analysis",
    "code_search",
    "file_search",
}

# Capabilities whose evidence is the returned content itself.
_CONTENT_CAPS = {
    "file_read",
    "file_inspection",
    "directory_list",
}

# Capabilities where an explicit "no X found" answer is legitimate evidence.
_NO_RESULT_OK = {"reference_lookup", "code_search", "file_search", "memory_query", "graph_reasoning"}

# Capabilities that need multi-step investigation (completeness signal).
_INVESTIGATION_CAPS = {
    "repository_investigation",
    "repository_inspection",
    "semantic_search",
    "dependency_analysis",
}

# Capabilities where a definitive location claim must be expressed with
# certainty (uncertainty calibration applies).
_DEFINITIVE_CAPS = {
    "definition_lookup",
    "symbol_search",
    "symbol_inspection",
    "file_read",
    "file_inspection",
}

# Uncertainty markers (speculative claims) vs honest-uncertainty markers.
_SPECULATIVE_RE = re.compile(
    r"\b(?:probably|likely|maybe|perhaps|I think|I believe|possibly)\b", re.IGNORECASE
)
_HONEST_UNCERTAIN_RE = re.compile(
    r"\b(?:could not|couldn'?t|unable to|no verified|not verify|cannot verify|no exact|no definitive)\b",
    re.IGNORECASE,
)
_EMPTY_RE = re.compile(r"couldn'?t\s+generate\s+a\s+response|could not generate a response", re.IGNORECASE)

_FILE_PATH_RE = re.compile(r"[\w./-]+\.(?:py|ts|js|rs|go|java|md|toml|yaml|yml|json)(?::\d+(?:[-,]\d+)*)?\b")

# Wrapper words that must not leak into an extracted symbol ("is TaskState").
_WRAPPER_LEAK = re.compile(r"^['\"]?(?:is|are|the|a|an|to|of)\s+[A-Za-z_]", re.IGNORECASE)

_QUOTED_ENTITY = re.compile(r"['\"]([A-Za-z_][\w]*(?:\s+[\w.]+)*)['\"]")


# ---------------------------------------------------------------------------
# Repository ground truth (Part 9) — deterministic, read-only.
# ---------------------------------------------------------------------------


def _repo_file_exists(root: Path, rel_path: str) -> bool:
    """Whether the path exists (files AND directories — subject paths may be
    either, e.g. directory_list subjects are directories)."""
    return (root / rel_path).exists()


def _repo_mentions(root: Path, rel_path: str, subject: str | None) -> bool:
    """Whether a repository file actually mentions the subject (any variant)."""
    if not subject or not _repo_file_exists(root, rel_path):
        return False
    try:
        content = (root / rel_path).read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return any(v.lower() in content for v in symbol_variants(subject))


_BOX_CHARS = re.compile(r"[│╭╰╮╯─]")
_MD_PREFIX = re.compile(r"^[#\-*>+\d.`\s]+")


def _repo_content_shown(root: Path, rel_path: str, transcript: str, min_hits: int = 2) -> bool:
    """Whether distinctive chunks of the file's ACTUAL content appear in the
    transcript — proof the file was really read (the read_file tool returns
    raw content; the rendered box carries no filename header, so a filename
    check would false-fail).  Box borders, markdown prefixes (the box strips
    ``#``/``-``/``*`` markers) and whitespace are normalized away."""
    path = root / rel_path
    if not path.is_file():
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    flat = re.sub(r"\s+", " ", _BOX_CHARS.sub("", transcript))
    hit = 0
    for line in content.splitlines():
        s = _MD_PREFIX.sub("", line.strip())
        s = re.sub(r"\s+", " ", s).strip()
        if not (12 <= len(s) <= 100):
            continue  # too short is noise; too long is wrapped by the box
        if s in flat:
            hit += 1
            if hit >= min_hits:
                return True
    return False


def _claimed_paths(transcript: str) -> list[str]:
    """File paths claimed in the reply (without line numbers)."""
    paths: list[str] = []
    for m in _FILE_PATH_RE.finditer(transcript):
        path = m.group(0)
        if ":" in path:
            path = path.split(":", 1)[0]
        if path not in paths:
            paths.append(path)
    return paths


def _verified_claim(root: Path, case: CapabilityTestCase, transcript: str) -> bool:
    """A claimed repository location is verified when the file exists and
    actually mentions the subject (or the file IS the subject for file caps)."""
    for path in _claimed_paths(transcript):
        if not _repo_file_exists(root, path):
            continue
        if case.expected_capability.value in _CONTENT_CAPS:
            return True  # the file exists and its content was the point
        if _repo_mentions(root, path, case.subject):
            return True
    return False


def _verify_directory(root: Path, case: CapabilityTestCase, transcript: str) -> bool:
    """Directory listing is verified against the actual directory contents.

    Verification uses the *reply window* (where the rendered tool output and
    final answer appear) — never the whole transcript, which contains the
    echoed user prompt and would trivially satisfy a subject check.
    """
    if not case.subject_path:
        return False
    dir_path = root / case.subject_path
    if not dir_path.is_dir():
        return False
    try:
        entries = sorted(p.name for p in dir_path.iterdir() if not p.name.startswith("."))
    except OSError:
        return False
    lowered = transcript[-_REPLY_WINDOW:].lower()
    if not entries:
        # Empty directory: the reply must actually say so ("empty",
        # "no files", ...) — a clarification prompt is not a listing.
        return bool(re.search(r"\b(?:empty|nothing|no files|no entries|no items)\b", lowered))
    return any(e.lower() in lowered for e in entries[:20])


def _verify_answer_claims(root: Path, case: CapabilityTestCase, transcript: str) -> bool:
    """Part 8: a claimed location must verify, not merely contain markers."""
    cap = case.expected_capability.value
    if cap in _CONTENT_CAPS:
        if cap == "directory_list":
            return _verify_directory(root, case, transcript)
        # file_read / file_inspection: the actual file content must appear in
        # the transcript (the read_file tool returns raw content — there is no
        # filename header in the rendered box).
        if not case.subject_path:
            return False
        return _repo_content_shown(root, case.subject_path, transcript)
    return _verified_claim(root, case, transcript)


# ---------------------------------------------------------------------------
# Layer C sub-evaluators (Part 10).
# ---------------------------------------------------------------------------


def _evidence_verdict(root: Path, parsed: ParsedTrace, case: CapabilityTestCase, transcript: str) -> Verdict:
    """Part 8: evidence signals must VERIFY against repository state.

    - speculative phrasing on a definitive claim        -> FAIL
    - claims that do not verify against the repo        -> FAIL
    - affirmative claim with no verifiable evidence     -> FAIL
    - honest "could not find" when the entity is absent -> PASS
    - explicit "no references/definition found" for caps where absence
      is a legitimate outcome                           -> PASS
    - nothing to verify                                 -> UNRESOLVED
    """
    cap = case.expected_capability.value
    if parsed.speculative and cap in _DEFINITIVE_CAPS:
        return Verdict.FAIL
    if cap in _REPO_CLAIM_CAPS | _CONTENT_CAPS:
        if _verify_answer_claims(root, case, transcript):
            return Verdict.PASS
        if _claimed_paths(transcript):
            return Verdict.FAIL  # claims that do not verify
        honest = _HONEST_UNCERTAIN_RE.search(transcript) is not None
        subject_exists = bool(case.subject_path and (root / case.subject_path).exists())
        if honest and cap in _NO_RESULT_OK:
            return Verdict.PASS  # explicit no-result is legitimate for these
        if honest and not subject_exists:
            return Verdict.PASS  # honest miss on an entity that is absent
        if case.subject and any(v.lower() in transcript.lower() for v in symbol_variants(case.subject)):
            return Verdict.FAIL  # affirmative/claiming answer without evidence
        return Verdict.UNRESOLVED
    # Non-repository capabilities: markers are signals; absence answers and
    # a present answer are acceptable evidence.
    markers = _EVIDENCE_MARKERS.get(cap, ())
    if markers and any(re.search(p, transcript[-800:], re.IGNORECASE) for p, _ in markers):
        return Verdict.PASS
    if cap in _NO_RESULT_OK and any(m in parsed.failure_markers for m in ("no_definition", "no_references")):
        return Verdict.PASS
    return Verdict.PASS if parsed.has_answer else Verdict.UNRESOLVED


def _answer_factual(root: Path, parsed: ParsedTrace, case: CapabilityTestCase, transcript: str) -> Verdict:
    cap = case.expected_capability.value
    if parsed.empty_response or "traceback" in parsed.failure_markers:
        return Verdict.FAIL
    if cap in _REPO_CLAIM_CAPS | _CONTENT_CAPS:
        return _evidence_verdict(root, parsed, case, transcript)
    return Verdict.PASS if parsed.has_answer else Verdict.UNRESOLVED


def _answer_relevance(root: Path, parsed: ParsedTrace, case: CapabilityTestCase, transcript: str) -> Verdict:
    """The FINAL RESPONSE must answer the task.  The answer region (reply
    window) is checked — a tool action naming the subject does not make an
    off-topic answer relevant.  Content-returning capabilities (file read /
    directory list) are exempt: their content IS the answer."""
    if case.subject is None:
        return Verdict.PASS  # subject-free tasks (git/terminal/tests)
    if case.expected_capability.value in _CONTENT_CAPS:
        if _verify_answer_claims(root, case, transcript):
            return Verdict.PASS
        return Verdict.UNRESOLVED
    reply = transcript[-800:]
    if any(v.lower() in reply.lower() for v in symbol_variants(case.subject)):
        return Verdict.PASS
    if _HONEST_UNCERTAIN_RE.search(reply):
        return Verdict.PARTIAL  # engaged but unable; at least calibrated
    if parsed.has_answer and len(reply.strip()) > 80:
        return Verdict.FAIL  # substantial answer that never names the subject
    return Verdict.UNRESOLVED


def _answer_grounding(root: Path, parsed: ParsedTrace, case: CapabilityTestCase, transcript: str) -> Verdict:
    if case.expected_capability.value in _REPO_CLAIM_CAPS | _CONTENT_CAPS:
        return _evidence_verdict(root, parsed, case, transcript)
    return Verdict.PASS if parsed.has_answer else Verdict.UNRESOLVED


def _answer_completeness(parsed: ParsedTrace, case: CapabilityTestCase, transcript: str) -> Verdict:
    cap = case.expected_capability.value
    if parsed.empty_response:
        return Verdict.FAIL
    if not parsed.has_answer:
        return Verdict.UNRESOLVED
    # Completeness markers: the expected *kind* of result is present.
    markers = _EVIDENCE_MARKERS.get(cap, ())
    if markers and any(re.search(p, transcript[-800:], re.IGNORECASE) for p, _ in markers):
        return Verdict.PASS
    if any(m in parsed.failure_markers for m in ("no_definition", "no_references")):
        return Verdict.PASS  # an explicit absence answer is complete for lookup
    return Verdict.PARTIAL


def _answer_calibration(parsed: ParsedTrace, case: CapabilityTestCase) -> Verdict:
    cap = case.expected_capability.value
    if parsed.speculative and cap in _DEFINITIVE_CAPS:
        return Verdict.FAIL
    if any(m in parsed.failure_markers for m in ("no_definition", "no_references")):
        return Verdict.PASS  # honest "not found"
    if parsed.has_answer:
        return Verdict.PASS  # answer present without speculative overreach
    return Verdict.UNRESOLVED


# ---------------------------------------------------------------------------
# Detail evidence markers (signals only — the layers verify them).
# ---------------------------------------------------------------------------

_EVIDENCE_MARKERS: dict[str, tuple[tuple[str, str], ...]] = {
    "definition_lookup": ((r"\b(?:defined|definition|class|function|def|enum|type|interface|struct)\b", "definition_keyword"),),
    "reference_lookup": ((r"\breferences?\b", "reference_keyword"),),
    "symbol_search": ((r"\bsymbol\b", "symbol_keyword"),),
    "symbol_inspection": ((r"\b(?:class|function|def|enum)\b", "definition_keyword"),),
    "repository_investigation": (
        (r"\b(?:primary|implementation|definition)\b", "implementation_keyword"),
    ),
    "directory_list": (
        (r"\b(?:files?|entries|items|directories?|folders?)\b", "listing_keyword"),
        (r"[\w-]+\.(?:yaml|yml|toml|json|py|md|txt|cfg|ini)\b", "listing_entry"),
    ),
    "terminal_execution": (
        (r"\b(?:pwd|git|ls|date|whoami)\b", "command_keyword"),
        (r"\d{1,2}:\d{2}(?::\d{2})?\b", "time_output"),
        (r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s+\w+\s+\d{1,2}", "date_output"),
    ),
    "git_operation": ((r"\b(?:status|diff|commit|branch|log)\b", "git_keyword"),),
    "memory_query": ((r"\b(?:remember|knowledge|memory|fact)\b", "memory_keyword"),),
    "graph_reasoning": ((r"\b(?:know|knowledge|graph|relationship)\b", "graph_keyword"),),
    "resource_monitoring": ((r"\b(?:cpu|memory|ram|disk|usage|process)\b", "resource_keyword"),),
    "debug_environment": ((r"\b(?:env|environment|variable|config)\b", "env_keyword"),),
    "structured_output": ((r"\b(?:structured|json|schema)\b", "structured_keyword"),),
    "api_schema_learning": ((r"\b(?:schema|api|endpoint)\b", "schema_keyword"),),
    "parallel_batch": ((r"\b(?:batch|parallel)\b", "batch_keyword"),),
    "file_search": ((r"\.(?:py|ts|js|rs|go|java|md|toml|yaml|yml|json)\b", "file_path"),),
    "code_search": ((r"\.(?:py|ts|js|rs|go|java|md|toml|yaml|yml|json)\b", "file_path"),),
}


# ---------------------------------------------------------------------------
# Layer evaluators.
# ---------------------------------------------------------------------------


def _layer_capability(case: CapabilityTestCase, root: Path) -> Verdict:
    """Layer A — the task's true capability (task truth, not model score)."""
    if case.contract is None:
        return Verdict.FAIL
    if case.subject_path and not _repo_file_exists(root, case.subject_path):
        return Verdict.FAIL  # the task's subject does not exist in the repo
    return Verdict.PASS


def _layer_execution(
    root: Path, parsed: ParsedTrace, case: CapabilityTestCase, transcript: str
) -> tuple[Verdict, dict[str, Verdict]]:
    """Layer B — appropriate operation + valid evidence."""
    dims: dict[str, Verdict] = {}

    # Security.
    if "denied" in (parsed.security_decision or ""):
        dims["security"] = Verdict.FAIL
    elif parsed.security_decision is not None:
        dims["security"] = Verdict.PASS
    else:
        dims["security"] = Verdict.PASS if parsed.has_answer else Verdict.UNRESOLVED

    # Empty / crash means execution did not complete.
    if parsed.empty_response or "traceback" in parsed.failure_markers or "tool_error" in parsed.failure_markers:
        return Verdict.FAIL, {**dims, "tool_selection": Verdict.UNRESOLVED, "argument": Verdict.UNRESOLVED, "evidence": Verdict.FAIL}

    # Tool selection.
    valid_tools = set(tools_with_capability(case.expected_capability))
    contract = case.contract
    related_tools: set[str] = set()
    if contract is not None:
        for rel in contract.related_capabilities:
            related_tools.update(tools_with_capability(rel))
    observed = parsed.observed_tool
    if observed is None:
        dims["tool_selection"] = Verdict.UNRESOLVED
    elif observed in valid_tools:
        dims["tool_selection"] = Verdict.PASS
    elif observed in related_tools:
        dims["tool_selection"] = Verdict.PARTIAL  # capability preserved, wrong tool
    else:
        dims["tool_selection"] = Verdict.FAIL

    # Argument correctness (subject mention + no wrapper leak).
    arg: Verdict = Verdict.PASS
    if case.subject:
        if not any(v.lower() in transcript.lower() for v in symbol_variants(case.subject)):
            arg = Verdict.FAIL
        else:
            for ent in _QUOTED_ENTITY.finditer(transcript[-800:]):
                if any(v.lower() in ent.group(1).lower() for v in symbol_variants(case.subject)) and _WRAPPER_LEAK.match(ent.group(1)):
                    arg = Verdict.FAIL
                    break
    dims["argument"] = arg

    # Evidence: signals MUST verify against repository state (Part 8).
    dims["evidence"] = _evidence_verdict(root, parsed, case, transcript)

    if any(v is Verdict.FAIL for v in dims.values()):
        return Verdict.FAIL, dims
    if dims["tool_selection"] is Verdict.PASS and dims["evidence"] is Verdict.PASS and arg is Verdict.PASS:
        return Verdict.PASS, dims
    if dims["evidence"] is Verdict.PASS and arg is Verdict.PASS:
        return Verdict.PARTIAL, dims  # evidence retrieved, operation not fully observable
    return Verdict.UNRESOLVED, dims


def _layer_answer(
    root: Path, parsed: ParsedTrace, case: CapabilityTestCase, transcript: str
) -> tuple[Verdict, dict[str, Verdict]]:
    """Layer C — final-answer truth, evaluated independently of the tool trace."""
    dims = {
        "factual_correctness": _answer_factual(root, parsed, case, transcript),
        "task_relevance": _answer_relevance(root, parsed, case, transcript),
        "evidence_grounding": _answer_grounding(root, parsed, case, transcript),
        "completeness": _answer_completeness(parsed, case, transcript),
        "uncertainty_calibration": _answer_calibration(parsed, case),
    }
    if any(v is Verdict.FAIL for v in dims.values()):
        return Verdict.FAIL, dims
    if all(v is Verdict.PASS for v in dims.values()):
        return Verdict.PASS, dims
    if any(v is Verdict.PASS for v in dims.values()):
        return Verdict.PARTIAL, dims
    return Verdict.UNRESOLVED, dims


def _aggregate(capability: Verdict, execution: Verdict, answer: Verdict) -> Verdict:
    """Explicit overall rule (Part 11): no successful dimension hides a failure."""
    if any(v is Verdict.FAIL for v in (capability, execution, answer)):
        return Verdict.FAIL
    if capability is Verdict.PASS and execution is Verdict.PASS and answer is Verdict.PASS:
        return Verdict.PASS
    if any(v is Verdict.PASS for v in (capability, execution, answer)):
        return Verdict.PARTIAL
    return Verdict.UNRESOLVED


# ---------------------------------------------------------------------------
# Failure attribution (Part 13).
# ---------------------------------------------------------------------------


def _attribution(
    case: CapabilityTestCase, parsed: ParsedTrace, dims: dict[str, Verdict]
) -> FailureKind | None:
    if "denied" in (parsed.security_decision or ""):
        return FailureKind.SECURITY_FAILURE
    if parsed.empty_response:
        return FailureKind.MODEL_LIMITATION
    if "traceback" in parsed.failure_markers or "tool_error" in parsed.failure_markers:
        return FailureKind.EXECUTION_FAILURE
    if "command_not_found" in parsed.failure_markers:
        return FailureKind.ENVIRONMENT_FAILURE
    # Part 13: a repository question routed to external web search is a
    # routing/capability-selection failure — never an evidence failure.
    if "web_search_routing" in parsed.failure_markers:
        expected = case.expected_capability.value
        model_caps = set(parsed.observed_capabilities)
        if case.router_capability == "web_search" or "web_search" in model_caps:
            return FailureKind.ROUTING_FAILURE  # router/model both chose external
        if case.router_agreement:
            return FailureKind.CAPABILITY_SELECTION_FAILURE  # router was clear, model went external
        return FailureKind.CAPABILITY_SELECTION_FAILURE
    # Part 13: clarification fallback instead of answering — if the
    # deterministic router was also confused, routing is responsible; if the
    # router was clear but the model still punted, it is a model limitation.
    # (A clarification reply never names the subject, so the argument/evidence
    # dims read as failed — the punt itself is the root cause.)
    if "clarification_prompt" in parsed.failure_markers and dims.get("final_answer") in (Verdict.FAIL, Verdict.PARTIAL):
        if case.router_agreement is False and case.router_capability is not None:
            return FailureKind.ROUTING_FAILURE
        return FailureKind.MODEL_LIMITATION
    if dims.get("argument") is Verdict.FAIL:
        return FailureKind.ARGUMENT_FAILURE
    tool = dims.get("tool_selection")
    if tool is Verdict.FAIL:
        model_caps = set(parsed.observed_capabilities)
        expected = case.expected_capability.value
        if expected in model_caps:
            return FailureKind.TOOL_SELECTION_FAILURE  # right capability, wrong tool
        if case.router_agreement:
            return FailureKind.CAPABILITY_SELECTION_FAILURE  # router was clear, model chose wrong
        if case.router_capability in model_caps:
            return FailureKind.ROUTING_FAILURE  # router reinforced the wrong choice
        return FailureKind.CAPABILITY_SELECTION_FAILURE
    if dims.get("evidence") is Verdict.FAIL:
        return FailureKind.EVIDENCE_FAILURE
    if parsed.speculative:
        return FailureKind.EVIDENCE_FAILURE
    if dims.get("final_answer") is Verdict.FAIL:
        return FailureKind.SYNTHESIS_FAILURE  # execution fine, explanation wrong
    if any(v is Verdict.FAIL for v in dims.values()):
        return FailureKind.UNKNOWN
    return None


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------


def evaluate_trace(trace: TaskTrace, repo_root: str | Path | None = None) -> Evaluation:
    """Evaluates one trace on the three layers (deterministic)."""
    root = Path(repo_root) if repo_root is not None else _PROJECT_ROOT
    case = trace.case
    parsed = parse_trace(trace.transcript)
    transcript = trace.transcript

    capability = _layer_capability(case, root)
    execution, exec_dims = _layer_execution(root, parsed, case, transcript)
    answer, answer_dims = _layer_answer(root, parsed, case, transcript)
    overall = _aggregate(capability, execution, answer)

    dims: dict[str, Verdict] = {}
    dims["intent"] = Verdict.PASS if execution is Verdict.PASS else Verdict.UNRESOLVED
    dims["capability"] = Verdict.PASS if execution is Verdict.PASS else (
        Verdict.UNRESOLVED if exec_dims.get("tool_selection") is not Verdict.FAIL else Verdict.FAIL
    )
    dims.update(exec_dims)
    dims["final_answer"] = answer

    notes: list[str] = []
    if parsed.observed_tool:
        notes.append(f"tool observed: {parsed.observed_tool}")
    if parsed.security_decision:
        notes.append(f"security decision: {parsed.security_decision}")
    if parsed.failure_markers:
        notes.append(f"failure markers: {', '.join(parsed.failure_markers)}")
    if case.router_capability:
        agree = "agree" if case.router_agreement else "disagree"
        notes.append(f"router: {case.router_capability} ({agree})")

    model_capability = ",".join(parsed.observed_capabilities) or None
    failure_kind = _attribution(case, parsed, dims) if overall is Verdict.FAIL else None

    return Evaluation(
        capability=capability,
        execution=execution,
        answer=answer,
        answer_dimensions=answer_dims,
        overall=overall,
        dimensions=dims,
        model_capability=model_capability,
        failure_kind=failure_kind,
        notes=notes,
    )


def evaluate_many(traces: list[TaskTrace], repo_root: str | Path | None = None) -> list[Evaluation]:
    return [evaluate_trace(t, repo_root=repo_root) for t in traces]
