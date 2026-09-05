"""
Harness components for Model-in-the-Loop validation:
- MITLSandbox: Isolated temporary repository management.
- MITLRunner: Real agent execution and trace capture.
- MITLGrader: Independent objective verification and grading.
"""

from tests.model_in_loop.harness.grader import GradingReport, MITLGrader
from tests.model_in_loop.harness.runner import ExecutionTrace, MITLRunner
from tests.model_in_loop.harness.sandbox import MITLSandbox

__all__ = [
    "ExecutionTrace",
    "GradingReport",
    "MITLGrader",
    "MITLRunner",
    "MITLSandbox",
]
