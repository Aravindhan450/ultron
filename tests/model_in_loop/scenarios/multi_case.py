"""
Multi-case bug scenario for Model-in-the-Loop validation (Scenario 3).
Exercises handling multiple distinct input/edge cases across a single module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DurationFormatterScenario:
    """
    Multi-case duration formatting scenario:
    - duration_util.py has an incomplete format_duration() function that only handles seconds
    - test_duration_util.py tests 0s, seconds (<60s), minutes/seconds (e.g. 125s -> '2m 5s'),
      and hours/minutes/seconds (e.g. 3665s -> '1h 1m 5s')
    """

    name: str = "duration_formatter_multi_case"
    prompt: str = (
        "Fix the failing tests in test_duration_util.py. "
        "Inspect the repository, identify all failing edge cases in duration_util.py for format_duration(), "
        "implement correct handling for seconds (<60s), minutes ('Xm Ys'), and hours ('Xh Ym Zs'), "
        "run pytest to verify all test cases pass, and report completion."
    )
    target_file: str = "duration_util.py"
    test_file: str = "test_duration_util.py"

    initial_files: dict[str, str] = field(
        default_factory=lambda: {
            "duration_util.py": (
                "def format_duration(seconds: int) -> str:\n"
                '    """Formats seconds into human readable duration (e.g. 45s, 2m 5s, 1h 1m 5s)."""\n'
                "    # Incomplete implementation that only handles bare seconds\n"
                '    return f"{seconds}s"\n'
            ),
            "test_duration_util.py": (
                "from duration_util import format_duration\n\n\n"
                "def test_zero_seconds():\n"
                '    assert format_duration(0) == "0s"\n\n\n'
                "def test_seconds_only():\n"
                '    assert format_duration(45) == "45s"\n\n\n'
                "def test_minutes_and_seconds():\n"
                '    assert format_duration(125) == "2m 5s"\n'
                '    assert format_duration(60) == "1m 0s"\n\n\n'
                "def test_hours_minutes_and_seconds():\n"
                '    assert format_duration(3665) == "1h 1m 5s"\n'
                '    assert format_duration(7200) == "2h 0m 0s"\n'
            ),
            "pyproject.toml": (
                "[project]\n"
                'name = "duration-util"\n'
                'version = "0.1.0"\n\n'
                "[tool.pytest.ini_options]\n"
                'testpaths = ["."]\n'
                'pythonpath = ["."]\n'
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
        """Validates that duration_util.py implements multi-case formatting."""
        if not sandbox.file_exists(self.target_file):
            return False, f"Target file '{self.target_file}' is missing from sandbox."
        content = sandbox.read_file(self.target_file)
        if "//" in content or "divmod" in content or "%" in content or "3600" in content or "60" in content:
            return True, None
        return False, f"Implementation in '{self.target_file}' lacks duration division logic."
