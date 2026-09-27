"""ultron.core.intelligence.intent_understanding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

INTENT UNDERSTANDING & REQUIREMENTS DERIVATION LAYER:

Translates a natural-language software goal into a structured UserIntent:
- Product Type detection (Desktop GUI, Web App, CLI Tool, REST API, Database App, Library, etc.)
- Explicit Requirements extraction
- Inferred Necessary Requirements
- Ambiguities vs Assumptions resolution
- Product-aware Acceptance Criteria with explicit verification methods & evidence levels
- Verification Strategy formulation

This serves as the authoritative contract before planning and execution.
"""

from __future__ import annotations

import re

from ultron.core.types import (
    AcceptanceCriterion,
    AcceptanceCriterionStatus,
    AuthorityLevel,
    Capability,
    EvidenceLevel,
    ExecutionIntent,
    ExecutionPolicy,
    ProductType,
    TaskType,
    UserIntent,
)

# ---------------------------------------------------------------------------
# Intent & Policy Prohibitions and Permissions
# ---------------------------------------------------------------------------

_PROHIBIT_MUTATION_RE = re.compile(
    r"\b(do\s+not\s+(?:modify|change|edit|write|touch|delete)|don'?t\s+(?:modify|change|edit|write|touch|delete)|"
    r"read[-\s]?only|no\s+(?:modifications?|changes?|edits?|writes?|mutations?)(?!\s+to\s+(?:behavior|functionality|semantics))|"
    r"without\s+(?:modifying|changing|editing|touching)(?!\s+(?:behavior|functionality|semantics|logic|meaning))|"
    r"only\s+investigat\w*|just\s+explain\w*|analyze\s+without\s+changing(?!\s+(?:behavior|functionality|semantics))|"
    r"don'?t\s+make\s+changes|do\s+not\s+make\s+changes)\b",
    re.IGNORECASE,
)

_PROHIBIT_EXECUTION_RE = re.compile(
    r"\b(do\s+not\s+(?:run|execute)|don'?t\s+(?:run|execute)|no\s+(?:commands?|execution)|"
    r"without\s+(?:running|executing)|do\s+not\s+run\s+tests?|don'?t\s+run\s+tests?|no\s+tests?)\b",
    re.IGNORECASE,
)

_PROHIBIT_DELETION_RE = re.compile(
    r"\b(do\s+not\s+delete|don'?t\s+delete|no\s+delet\w*|without\s+deleting)\b",
    re.IGNORECASE,
)

_PERMIT_REPAIR_RE = re.compile(
    r"\b(fix\w*|repair\w*|patch\w*|resolve\w*|correct\w*|remedy\w*|solve\w*|make\s+(?:the\s+)?tests?\s+pass|make\s+it\s+pass|apply\s+(?:a\s+)?fix)\b",
    re.IGNORECASE,
)

_PERMIT_PLAN_RE = re.compile(
    r"\b(propose\s+(?:a\s+)?(?:fix|plan|solution)|plan\s+(?:the\s+)?(?:fix|implementation)|design\s+(?:a\s+)?(?:fix|solution)|"
    r"suggest\s+(?:a\s+)?fix|outline\s+(?:a\s+)?fix)\b",
    re.IGNORECASE,
)

_PERMIT_IMPLEMENT_RE = re.compile(
    r"\b(implement\w*|create\w*|build\w*|scaffold\w*|develop\w*|add\w*|write\w*|make\w*|change\w*|modify\w*|update\w*|edit\w*|refactor\w*|set\w*)\b",
    re.IGNORECASE,
)

_PERMIT_EXPLAIN_RE = re.compile(
    r"\b(explain\w*|how\s+(?:does|do|can|is)|walk\s+me\s+through|what\s+is|overview\s+of|teach\s+me)\b",
    re.IGNORECASE,
)

_PERMIT_INSPECT_RE = re.compile(
    r"\b(inspect\w*|list\b|find\s+files|show\s+files|explore\b|structure\s+of|locate\s+files)\b",
    re.IGNORECASE,
)

_PERMIT_DIAGNOSE_RE = re.compile(
    r"\b(diagnos\w*|investigat\w*|trace\s+why|why\s+does|identify\s+(?:the\s+)?(?:root\s+cause|problem|issue|bug)|troubleshoot\w*)\b",
    re.IGNORECASE,
)


def resolve_execution_intent(
    prompt: str,
    task_type: TaskType | None = None,
) -> ExecutionIntent:
    """
    Resolves the user's intended outcome independently of the task domain.
    """
    text = prompt.lower()
    has_prohibit_mutation = bool(_PROHIBIT_MUTATION_RE.search(text))
    has_repair = bool(_PERMIT_REPAIR_RE.search(text))
    has_plan = bool(_PERMIT_PLAN_RE.search(text))
    has_implement = bool(_PERMIT_IMPLEMENT_RE.search(text))
    has_diagnose = bool(_PERMIT_DIAGNOSE_RE.search(text))
    has_explain = bool(_PERMIT_EXPLAIN_RE.search(text))
    has_inspect = bool(_PERMIT_INSPECT_RE.search(text))

    # 1. Propose / plan fix
    if has_plan and not (has_repair and not has_prohibit_mutation):
        return ExecutionIntent.PLAN

    # 2. Explicit prohibition on mutation takes precedence over repair verbs
    if has_prohibit_mutation:
        if has_diagnose:
            return ExecutionIntent.DIAGNOSE
        if has_inspect:
            return ExecutionIntent.INSPECT
        return ExecutionIntent.EXPLAIN

    # 3. Explicit repair / fix request
    if has_repair:
        return ExecutionIntent.REPAIR

    # 4. Explicit implementation request
    if has_implement:
        return ExecutionIntent.IMPLEMENT

    # 5. Diagnostic investigation without repair request
    if has_diagnose:
        return ExecutionIntent.DIAGNOSE

    # 6. Explanatory request
    if has_explain:
        return ExecutionIntent.EXPLAIN

    # 7. Inspection request
    if has_inspect:
        return ExecutionIntent.INSPECT

    # 8. Domain fallback from TaskType
    if task_type in (TaskType.SOFTWARE_ENGINEERING, TaskType.MULTI_STEP):
        return ExecutionIntent.IMPLEMENT
    if task_type in (TaskType.SYSTEM_OPERATION, TaskType.CONFIGURATION, TaskType.DATA_OPERATION):
        return ExecutionIntent.EXECUTE
    if task_type == TaskType.DEBUGGING:
        return ExecutionIntent.DIAGNOSE
    if task_type in (TaskType.RESEARCH, TaskType.INFORMATIONAL):
        return ExecutionIntent.EXPLAIN
    if task_type == TaskType.CODE_REVIEW:
        return ExecutionIntent.INSPECT

    return ExecutionIntent.EXPLAIN


def derive_execution_policy(
    intent: ExecutionIntent,
    prompt: str,
    task_type: TaskType | None = None,
) -> ExecutionPolicy:
    """
    Derives the authoritative ExecutionPolicy from the resolved intent and explicit constraints.
    """
    text = prompt.lower()
    has_prohibit_mutation = bool(_PROHIBIT_MUTATION_RE.search(text))
    has_prohibit_execution = bool(_PROHIBIT_EXECUTION_RE.search(text))
    has_prohibit_deletion = bool(_PROHIBIT_DELETION_RE.search(text))

    base_caps: set[Capability] = {
        Capability.READ_FILES,
        Capability.SEARCH_REPOSITORY,
        Capability.INSPECT_SYMBOLS,
        Capability.GIT_READ,
    }

    # Pure read-only intents or explicit prohibition
    if has_prohibit_mutation or intent == ExecutionIntent.DIAGNOSE:
        return ExecutionPolicy(
            allowed_capabilities=base_caps,
            mutation_authority=AuthorityLevel.FORBIDDEN,
            execution_authority=AuthorityLevel.FORBIDDEN,
            confirmation_requirements=["Read-only: no file mutations or commands authorized"],
        )

    if intent in (ExecutionIntent.EXPLAIN, ExecutionIntent.INSPECT):
        caps = set(base_caps)
        if not has_prohibit_execution:
            caps.add(Capability.NETWORK)
        return ExecutionPolicy(
            allowed_capabilities=caps,
            mutation_authority=AuthorityLevel.FORBIDDEN,
            execution_authority=AuthorityLevel.FORBIDDEN,
            confirmation_requirements=["Read-only: no file mutations or commands authorized"],
        )

    if intent == ExecutionIntent.PLAN:
        return ExecutionPolicy(
            allowed_capabilities=base_caps,
            mutation_authority=AuthorityLevel.FORBIDDEN,
            execution_authority=AuthorityLevel.FORBIDDEN,
            confirmation_requirements=["Plan-only: research permitted; no mutations or command execution authorized"],
        )

    # REPAIR, IMPLEMENT, EXECUTE
    allowed_caps = set(base_caps)
    mutation_auth = AuthorityLevel.REQUIRES_CONFIRMATION
    exec_auth = AuthorityLevel.REQUIRES_CONFIRMATION

    if not has_prohibit_mutation:
        allowed_caps.add(Capability.WRITE_FILES)
        if not has_prohibit_deletion:
            allowed_caps.add(Capability.DELETE_FILES)
    else:
        mutation_auth = AuthorityLevel.FORBIDDEN

    if not has_prohibit_execution:
        allowed_caps.add(Capability.RUN_COMMANDS)
        allowed_caps.add(Capability.RUN_TESTS)
    else:
        exec_auth = AuthorityLevel.FORBIDDEN

    allowed_caps.add(Capability.NETWORK)
    allowed_caps.add(Capability.EXTERNAL_ACTIONS)

    return ExecutionPolicy(
        allowed_capabilities=allowed_caps,
        mutation_authority=mutation_auth,
        execution_authority=exec_auth,
        confirmation_requirements=["State-modifying actions require user confirmation"],
    )


# ---------------------------------------------------------------------------
# Deterministic Product Type Classification
# ---------------------------------------------------------------------------

_DESKTOP_GUI_RE = re.compile(
    r"\b(desktop\s+(?:app|application|weather|tool)|gui\b|tkinter|pyqt|wxpython|"
    r"window\s+interface|graphical\s+user\s+interface|desktop\s+gui)\b",
    re.IGNORECASE,
)

_WEB_APP_RE = re.compile(
    r"\b(web\s+(?:app|application|dashboard|interface|page|server)|dashboard\b|"
    r"flask|fastapi|django|html|css|browser\s+ui|web\s+dashboard)\b",
    re.IGNORECASE,
)

_REST_API_RE = re.compile(
    r"\b(rest\s+api|api\s+server|json\s+api|endpoints?|crud\s+api|backend\s+api)\b",
    re.IGNORECASE,
)

_DATABASE_APP_RE = re.compile(
    r"\b(database\s+app|sqlite|sqlite3|sql\s+database|db\s+persistence|"
    r"expense\s+tracker|student\s+expense|record\s+manager)\b",
    re.IGNORECASE,
)

_CLI_TOOL_RE = re.compile(
    r"\b(cli\b|command[-\s]line\s+tool|terminal\s+tool|argparse|typer|console\s+app)\b",
    re.IGNORECASE,
)

_AUTOMATION_SCRIPT_RE = re.compile(
    r"\b(scraper|crawler|automation\s+script|downloader|fetcher\s+script)\b",
    re.IGNORECASE,
)

_LIBRARY_RE = re.compile(
    r"\b(library\b|package\b|sdk\b|reusable\s+module|utility\s+package)\b",
    re.IGNORECASE,
)


def detect_product_type(prompt: str) -> ProductType:
    """Classifies a user request into a concrete software product type."""
    if _DESKTOP_GUI_RE.search(prompt):
        return ProductType.DESKTOP_GUI
    if _WEB_APP_RE.search(prompt):
        return ProductType.WEB_APP
    if _REST_API_RE.search(prompt):
        return ProductType.REST_API
    if _DATABASE_APP_RE.search(prompt):
        return ProductType.DATABASE_APP
    if _CLI_TOOL_RE.search(prompt):
        return ProductType.CLI_TOOL
    if _AUTOMATION_SCRIPT_RE.search(prompt):
        return ProductType.AUTOMATION_SCRIPT
    if _LIBRARY_RE.search(prompt):
        return ProductType.LIBRARY
    return ProductType.GENERAL_SOFTWARE


def derive_acceptance_criteria(
    prompt: str,
    product_type: ProductType,
    execution_intent: ExecutionIntent = ExecutionIntent.IMPLEMENT,
    policy: ExecutionPolicy | None = None,
) -> list[AcceptanceCriterion]:
    """
    Generates outcome-oriented acceptance criteria tailored to intent and policy.
    """
    is_ro = policy.is_read_only if policy else execution_intent in (
        ExecutionIntent.EXPLAIN,
        ExecutionIntent.INSPECT,
        ExecutionIntent.DIAGNOSE,
        ExecutionIntent.PLAN,
    )

    if is_ro or execution_intent in (
        ExecutionIntent.EXPLAIN,
        ExecutionIntent.INSPECT,
        ExecutionIntent.DIAGNOSE,
        ExecutionIntent.PLAN,
    ):
        if execution_intent == ExecutionIntent.DIAGNOSE:
            return [
                AcceptanceCriterion(
                    id="relevant_code_inspected",
                    description="Relevant subsystem files and code paths are located and inspected",
                    verification_method="code_inspection",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_1_CREATED,
                ),
                AcceptanceCriterion(
                    id="failure_mechanism_identified",
                    description="Failure mechanism and variable/logic flow are traced across components",
                    verification_method="root_cause_analysis",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_1_CREATED,
                ),
                AcceptanceCriterion(
                    id="diagnosis_reported",
                    description=f"Evidence-backed diagnosis for '{prompt[:60]}' is synthesized and reported",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        elif execution_intent == ExecutionIntent.PLAN:
            return [
                AcceptanceCriterion(
                    id="current_state_analyzed",
                    description="Current implementation and failure points are analyzed",
                    verification_method="code_inspection",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_1_CREATED,
                ),
                AcceptanceCriterion(
                    id="plan_presented",
                    description=f"Implementation/fix plan for '{prompt[:60]}' is designed and presented",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        elif execution_intent == ExecutionIntent.INSPECT:
            return [
                AcceptanceCriterion(
                    id="repository_inspected",
                    description="Repository files and structure are inspected and inventoried",
                    verification_method="code_inspection",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_1_CREATED,
                ),
                AcceptanceCriterion(
                    id="inspection_reported",
                    description=f"Inspection findings for '{prompt[:60]}' are reported",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        else:  # EXPLAIN
            return [
                AcceptanceCriterion(
                    id="relevant_architecture_inspected",
                    description="Relevant files, symbols, and architecture are inspected",
                    verification_method="code_inspection",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_1_CREATED,
                ),
                AcceptanceCriterion(
                    id="explanation_reported",
                    description=f"Clear architectural explanation for '{prompt[:60]}' is provided",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]

    criteria: list[AcceptanceCriterion] = [
        AcceptanceCriterion(
            id="workspace_isolation",
            description="Project is scaffolded and isolated inside dedicated external workspace directory",
            verification_method="file_check",
            required=True,
            status=AcceptanceCriterionStatus.PENDING,
            evidence_level=EvidenceLevel.LEVEL_1_CREATED,
        ),
        AcceptanceCriterion(
            id="dependencies_and_code",
            description="All application files, entrypoints, and dependencies are implemented with complete code",
            verification_method="file_check",
            required=True,
            status=AcceptanceCriterionStatus.PENDING,
            evidence_level=EvidenceLevel.LEVEL_1_CREATED,
        ),
    ]

    if product_type == ProductType.DESKTOP_GUI:
        criteria.extend(
            [
                AcceptanceCriterion(
                    id="application_execution",
                    description="Desktop GUI process launches and initializes without runtime errors or crashes",
                    verification_method="process_check",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_3_LAUNCHED,
                ),
                AcceptanceCriterion(
                    id="gui_window_rendering",
                    description="GUI window and visual controls are rendered and interactive on display",
                    verification_method="gui_observation",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_3_LAUNCHED,
                ),
                AcceptanceCriterion(
                    id="behavior_interaction",
                    description="User interactions (item selection, actions, inputs) are executed and tested",
                    verification_method="execution",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_4_INTERACTED,
                ),
                AcceptanceCriterion(
                    id="verified_completion",
                    description=f"Desktop application output and requirements for '{prompt[:60]}' are verified with evidence",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        )
    elif product_type == ProductType.WEB_APP:
        criteria.extend(
            [
                AcceptanceCriterion(
                    id="application_execution",
                    description="Web server starts and binds to local network port successfully",
                    verification_method="process_check",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_3_LAUNCHED,
                ),
                AcceptanceCriterion(
                    id="http_endpoint_readiness",
                    description="Web routes respond with HTTP 200 OK and valid non-empty markup or JSON",
                    verification_method="http_probe",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_3_LAUNCHED,
                ),
                AcceptanceCriterion(
                    id="behavior_interaction",
                    description="Web flows (form submission, data display, route navigation) are exercised and verified",
                    verification_method="http_probe",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_4_INTERACTED,
                ),
                AcceptanceCriterion(
                    id="verified_completion",
                    description=f"Web application behavior for '{prompt[:60]}' is verified with live response evidence",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        )
    elif product_type == ProductType.DATABASE_APP:
        criteria.extend(
            [
                AcceptanceCriterion(
                    id="application_execution",
                    description="Database application runs and executes schema initialization",
                    verification_method="process_check",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_2_EXECUTED,
                ),
                AcceptanceCriterion(
                    id="behavior_interaction",
                    description="Data operations (insert, query, update, delete) are performed through application",
                    verification_method="db_query",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_4_INTERACTED,
                ),
                AcceptanceCriterion(
                    id="verified_completion",
                    description="Database records and calculated outputs are directly inspected and verified",
                    verification_method="db_query",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        )
    elif product_type == ProductType.CLI_TOOL:
        criteria.extend(
            [
                AcceptanceCriterion(
                    id="application_execution",
                    description="CLI binary/script executes with valid exit code and output",
                    verification_method="cli_test",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_2_EXECUTED,
                ),
                AcceptanceCriterion(
                    id="behavior_interaction",
                    description="CLI arguments, flags, and inputs are processed correctly across test cases",
                    verification_method="cli_test",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_4_INTERACTED,
                ),
                AcceptanceCriterion(
                    id="verified_completion",
                    description=f"CLI tool behavior for '{prompt[:60]}' is verified against expected outputs",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        )
    else:
        criteria.extend(
            [
                AcceptanceCriterion(
                    id="application_execution",
                    description="Software executes and runs without uncaught runtime errors",
                    verification_method="execution",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_2_EXECUTED,
                ),
                AcceptanceCriterion(
                    id="behavior_interaction",
                    description="Core features are exercised with realistic inputs and test scenarios",
                    verification_method="execution",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_4_INTERACTED,
                ),
                AcceptanceCriterion(
                    id="verified_completion",
                    description=f"Software goals for '{prompt[:60]}' are verified with machine evidence",
                    verification_method="reference_comparison",
                    required=True,
                    status=AcceptanceCriterionStatus.PENDING,
                    evidence_level=EvidenceLevel.LEVEL_5_VERIFIED,
                ),
            ]
        )

    return criteria


def understand_user_intent(
    prompt: str,
    cwd: str | None = None,
    task_type: TaskType | None = None,
) -> UserIntent:
    """
    Translates a natural language user request into a structured UserIntent.
    """
    cleaned = prompt.strip()
    product_type = detect_product_type(cleaned)
    exec_intent = resolve_execution_intent(cleaned, task_type)
    policy = derive_execution_policy(exec_intent, cleaned, task_type)

    # Extract explicit requirement statements
    explicit_reqs = [
        s.strip()
        for s in re.split(r"[.\n;]+", cleaned)
        if len(s.strip()) > 5
    ][:8]

    # Inferred necessary requirements based on product type
    inferred_reqs: list[str] = []
    assumptions: list[str] = []
    ambiguities: list[str] = []
    constraints: list[str] = [
        "Confine all created files strictly to external workspace",
        "Preserve security boundary and interactive confirmation gating",
    ]

    if policy.is_read_only:
        constraints.append("Read-only: do not modify, delete, or create repository files")
    if not policy.can_execute:
        constraints.append("Execution forbidden: do not run commands, servers, or test suites")

    if exec_intent in (ExecutionIntent.DIAGNOSE, ExecutionIntent.EXPLAIN, ExecutionIntent.INSPECT):
        inferred_reqs = [
            "Accurate code path and component identification",
            "Evidence-backed explanation of system flow",
            "Preserve existing repository state without changes",
        ]
        assumptions = ["Repository structure is intact and discoverable"]
    elif exec_intent == ExecutionIntent.PLAN:
        inferred_reqs = [
            "Root cause diagnosis",
            "Targeted architectural fix design",
            "Clear step-by-step implementation plan without modifying files",
        ]
        assumptions = ["No source code changes authorized during planning"]
    else:
        if product_type == ProductType.DESKTOP_GUI:
            inferred_reqs = [
                "Usable desktop graphical user interface with visual controls",
                "Event loop and interactive event handlers",
                "Real dynamic data loading or API integration",
                "Non-blocking user experience with error resilience",
            ]
            assumptions = [
                "Use native standard GUI library (e.g. Tkinter on macOS with Aqua bindings)",
                "Run via macOS native Python interpreter for GUI subsystem compatibility",
            ]
        elif product_type == ProductType.WEB_APP:
            inferred_reqs = [
                "Web server application listening on local port",
                "Clean HTML/CSS/JS frontend templates and views",
                "Working HTTP request routing and JSON/HTML rendering",
                "Readiness probing and live HTTP interaction test",
            ]
            assumptions = [
                "Bind server to localhost on an available standard port (e.g. 5000 or 8080)",
            ]
        elif product_type == ProductType.DATABASE_APP:
            inferred_reqs = [
                "Structured database schema definition",
                "Data models, insert, retrieve, and aggregation methods",
                "Persistent on-disk storage with direct verification",
            ]
            assumptions = [
                "Use SQLite for local, self-contained, zero-dependency persistence",
            ]
        elif product_type == ProductType.CLI_TOOL:
            inferred_reqs = [
                "Command-line entrypoint with argument parsing",
                "Formatted console output and exit code contracts",
                "Automated test execution exercising flags and inputs",
            ]
        else:
            inferred_reqs = [
                "Modular code implementation with clean entrypoint",
                "Automated validation test suite",
                "Verified runtime execution",
            ]

    # Check for material ambiguity
    if re.search(r"\bdeploy\b", cleaned, re.IGNORECASE) and not re.search(
        r"\b(local|docker|aws|k8s|test|workspace)\b", cleaned, re.IGNORECASE
    ):
        ambiguities.append("Target deployment environment not specified")

    acceptance_criteria = derive_acceptance_criteria(
        cleaned, product_type, execution_intent=exec_intent, policy=policy
    )

    if exec_intent == ExecutionIntent.DIAGNOSE:
        verification_strategy = (
            "Locate components -> trace execution flow -> isolate root cause -> deliver diagnostic report"
        )
    elif exec_intent == ExecutionIntent.EXPLAIN:
        verification_strategy = "Locate components -> inspect architecture -> deliver structured explanation"
    elif exec_intent == ExecutionIntent.PLAN:
        verification_strategy = "Inspect codebase -> formulate architecture -> present proposed implementation plan"
    else:
        strategy_map = {
            ProductType.DESKTOP_GUI: "Launch process -> confirm window -> interact with controls -> verify output rendering",
            ProductType.WEB_APP: "Start server -> probe HTTP endpoint -> send requests -> assert status 200 & content",
            ProductType.DATABASE_APP: "Initialize schema -> perform operations -> query SQLite tables directly to verify persistence",
            ProductType.CLI_TOOL: "Run command with arguments -> assert stdout format and return code 0",
            ProductType.REST_API: "Start API server -> send JSON requests -> validate HTTP status codes and schema",
            ProductType.LIBRARY: "Import module -> execute public API functions -> run pytest suite",
            ProductType.GENERAL_SOFTWARE: "Implement code -> execute entrypoint -> test behavior -> verify results",
        }
        verification_strategy = strategy_map.get(
            product_type, "Execute program -> exercise features -> verify evidence"
        )

    return UserIntent(
        raw_prompt=cleaned,
        goal=cleaned,
        objective=cleaned,
        task_type=task_type or TaskType.INFORMATIONAL,
        execution_intent=exec_intent,
        policy=policy,
        product_type=product_type,
        explicit_requirements=explicit_reqs,
        inferred_necessary_requirements=inferred_reqs,
        assumptions=assumptions,
        ambiguities=ambiguities,
        constraints=constraints,
        acceptance_criteria=acceptance_criteria,
        verification_strategy=verification_strategy,
    )
