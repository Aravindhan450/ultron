"""ultron.validation — generic capability-based model validation (STEP 3).

A measurement framework that tests whether the *actual* Ultron agent/model can
use the canonical capabilities for arbitrary natural-language requests.

The framework consumes the production architecture — it never redefines it:

    IntentCategory        (nlp/intent.py)      what the user wants
    ToolCapability        (tools/definitions)  the capability vocabulary
    CapabilityContract    (capabilities/)      behavioral expectations
    TOOL_DEFINITIONS      (tools/definitions)  tool metadata (authoritative)

Layers (per STEP 3):

    Layer A — deterministic regression tests (tests/), unchanged.
    Layer B — capability model tests: dynamically construct tasks from the
              current repository + capability contracts, execute them through
              the real Ultron CLI, capture full traces, and evaluate each
              task on eight dimensions (intent, capability, tool, argument,
              security, evidence, investigation, final answer).
    Layer C — holdout tests (STEP 4): generated independently from the
              implementation, never reused as production routing rules.

Modules:

    model       — CapabilityTestCase / TaskTrace / Evaluation data model
    subjects    — dynamic repository subject discovery (reuses Code Intelligence)
    generator   — capability classification + varied task generation
    runner      — real CLI execution (pty) + trace capture/parsing
    evaluate    — eight-dimension capability-level evaluation + failure classes
    audit       — anti-hardcoding static audit of production code
    report      — deterministic structured validation report

The framework never modifies production code to make a test pass; it exposes
failures classified by subsystem (see ``evaluate``).
"""

from ultron.validation.model import (
    CapabilityTestCase,
    Evaluation,
    FailureKind,
    TaskTrace,
    TestSplit,
    Verdict,
)

__all__ = [
    "CapabilityTestCase",
    "Evaluation",
    "FailureKind",
    "TaskTrace",
    "TestSplit",
    "Verdict",
]
