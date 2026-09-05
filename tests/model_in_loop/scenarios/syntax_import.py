"""
Syntax and import error recovery scenario for Model-in-the-Loop validation (Scenario 5).
Exercises diagnosing and repairing invalid syntax and broken import paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SyntaxImportRecoveryScenario:
    """
    Syntax and import recovery scenario:
    - parser.py has a deprecated/broken import (`from collections import Mapping` instead of `from collections.abc import Mapping`)
      and a missing closing parenthesis in a helper function.
    - test_parser.py verifies parsing dictionary configurations.
    """

    name: str = "syntax_import_recovery"
    prompt: str = (
        "Fix the syntax and import errors in parser.py causing test_parser.py to fail. "
        "Inspect the repository, fix the broken import from collections.abc and any syntax defects, "
        "run pytest to verify all tests pass, and report completion."
    )
    target_file: str = "parser.py"
    test_file: str = "test_parser.py"

    initial_files: dict[str, str] = field(
        default_factory=lambda: {
            "parser.py": (
                "# Broken import in Python 3.10+\n"
                "from collections import Mapping\n"
                "from typing import Any\n\n\n"
                "def parse_config(data: Any) -> dict[str, Any]:\n"
                '    """Parses mapping data into a normalized dictionary."""\n'
                "    if not isinstance(data, Mapping):\n"
                '        raise TypeError("Expected mapping type")\n'
                "    # Missing closing parenthesis syntax error\n"
                "    result = {k.strip(): v for k, v in data.items(\n"
                "    return result\n"
            ),
            "test_parser.py": (
                "import pytest\n"
                "from parser import parse_config\n\n\n"
                "def test_parse_valid_mapping():\n"
                '    data = {" host ": "localhost", "port": 8080}\n'
                "    res = parse_config(data)\n"
                '    assert res == {"host": "localhost", "port": 8080}\n\n\n'
                "def test_parse_invalid_type():\n"
                "    with pytest.raises(TypeError):\n"
                '        parse_config(["not", "a", "mapping"])\n'
            ),
            "pyproject.toml": (
                "[project]\n"
                'name = "config-parser"\n'
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
        """Validates that parser.py has fixed imports and valid syntax."""
        if not sandbox.file_exists(self.target_file):
            return False, f"Target file '{self.target_file}' is missing from sandbox."
        content = sandbox.read_file(self.target_file)
        if "collections.abc" not in content and "from collections.abc import Mapping" not in content:
            return False, "parser.py still contains invalid import (expected collections.abc)."
        if "from collections import Mapping" in content:
            return False, "parser.py still contains deprecated `from collections import Mapping`."
        return True, None
