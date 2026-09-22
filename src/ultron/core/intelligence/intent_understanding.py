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
    EvidenceLevel,
    ProductType,
    UserIntent,
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
) -> list[AcceptanceCriterion]:
    """
    Generates outcome-oriented acceptance criteria tailored to the product type.
    """
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
) -> UserIntent:
    """
    Translates a natural language user request into a structured UserIntent.
    """
    cleaned = prompt.strip()
    product_type = detect_product_type(cleaned)

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

    acceptance_criteria = derive_acceptance_criteria(cleaned, product_type)

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
        product_type=product_type,
        explicit_requirements=explicit_reqs,
        inferred_necessary_requirements=inferred_reqs,
        assumptions=assumptions,
        ambiguities=ambiguities,
        constraints=constraints,
        acceptance_criteria=acceptance_criteria,
        verification_strategy=verification_strategy,
    )
