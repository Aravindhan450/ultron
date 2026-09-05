"""
Calculator bug fix scenario for Model-in-the-Loop validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalculatorBugFixScenario:
    """
    A minimal software-engineering bug fix scenario:
    - calculator.py contains an intentional defect in `add()` (returns a - b instead of a + b)
    - test_calculator.py contains deterministic tests for `add()` and `multiply()`
    - pyproject.toml defines project configuration for pytest
    """

    name: str = "calculator_bug_fix"
    prompt: str = (
        "Fix the failing test in test_calculator.py in this calculator repository. "
        "Inspect the repository, identify the underlying implementation bug in calculator.py, "
        "make the minimal correct change, run pytest to verify the fix, and report completion."
    )
    target_file: str = "calculator.py"
    test_file: str = "test_calculator.py"

    initial_files: dict[str, str] = field(
        default_factory=lambda: {
            "calculator.py": (
                "def add(a: int | float, b: int | float) -> int | float:\n"
                '    """Returns the sum of a and b."""\n'
                "    return a - b  # Bug: subtraction instead of addition\n\n\n"
                "def multiply(a: int | float, b: int | float) -> int | float:\n"
                '    """Returns the product of a and b."""\n'
                "    return a * b\n"
            ),
            "test_calculator.py": (
                "from calculator import add, multiply\n\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n"
                "    assert add(-1, 1) == 0\n"
                "    assert add(0, 0) == 0\n\n\n"
                "def test_multiply():\n"
                "    assert multiply(3, 4) == 12\n"
                "    assert multiply(-2, 3) == -6\n"
            ),
            "pyproject.toml": (
                "[project]\n"
                'name = "calculator"\n'
                'version = "0.1.0"\n\n'
                "[tool.pytest.ini_options]\n"
                'testpaths = ["."]\n'
            ),
            ".gitignore": (
                "__pycache__/\n"
                "*.py[cod]\n"
                ".pytest_cache/\n"
                ".ultron*\n"
            ),
        }
    )

    def validate_implementation(self, sandbox: Any) -> tuple[bool, str | None]:
        """Validates that calculator.py contains the correct addition fix."""
        if not sandbox.file_exists(self.target_file):
            return False, f"Target file '{self.target_file}' is missing from sandbox."
        content = sandbox.read_file(self.target_file)
        if "a + b" in content and "a - b" not in content:
            return True, None
        return False, f"Implementation in '{self.target_file}' does not contain expected fix ('a + b')."
