"""
Repair scenario for Model-in-the-Loop validation.
Tests autonomous EDIT -> TEST -> OBSERVE FAILURE -> DIAGNOSE -> REPAIR -> RETEST -> VERIFY loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlugifyRepairScenario:
    """
    A multi-case string utility repair scenario:
    - string_util.py has an initial naive/broken slugify implementation
    - test_string_util.py contains tests for basic, special chars, multiple separators, and whitespace stripping
    """

    name: str = "slugify_repair_scenario"
    prompt: str = (
        "Fix the failing tests in this repository. "
        "Inspect the repository and test failures in test_string_util.py, "
        "repair string_util.py so that slugify() properly lowercases text, "
        "replaces punctuation and whitespace with single hyphens, "
        "and strips leading/trailing hyphens. "
        "Run pytest to verify all tests pass, and report completion."
    )
    target_file: str = "string_util.py"
    test_file: str = "test_string_util.py"

    initial_files: dict[str, str] = field(
        default_factory=lambda: {
            "string_util.py": (
                "def slugify(text: str) -> str:\n"
                '    """Converts a string into a clean URL slug."""\n'
                "    # Naive initial implementation that fails edge cases\n"
                '    return text.replace(" ", "-")\n'
            ),
            "test_string_util.py": (
                "from string_util import slugify\n\n\n"
                "def test_basic_slug():\n"
                '    assert slugify("Hello World") == "hello-world"\n\n\n'
                "def test_special_characters():\n"
                '    assert slugify("Hello, World! 2026") == "hello-world-2026"\n\n\n'
                "def test_leading_trailing_and_multiple_hyphens():\n"
                '    assert slugify("  --Hello   World-- ") == "hello-world"\n\n\n'
                "def test_empty_string():\n"
                '    assert slugify("") == ""\n'
            ),
            "pyproject.toml": (
                "[project]\n"
                'name = "string-util"\n'
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
        """Validates that string_util.py contains a working slugify implementation."""
        if not sandbox.file_exists(self.target_file):
            return False, f"Target file '{self.target_file}' is missing from sandbox."
        content = sandbox.read_file(self.target_file)
        if "re.sub" in content or ("lower" in content and "strip" in content):
            return True, None
        return False, f"Implementation in '{self.target_file}' does not look like a robust slugify fix."
