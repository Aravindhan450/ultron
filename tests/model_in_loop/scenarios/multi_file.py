"""
Multi-file change scenario for Model-in-the-Loop validation (Scenario 4).
Exercises coordinated changes across interdependent files in a repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfigServiceScenario:
    """
    Multi-file configuration and service client scenario:
    - config.py defines ServerConfig
    - client.py defines APIClient which depends on ServerConfig
    - test_client.py verifies custom base url and timeout handling
    """

    name: str = "config_service_multi_file"
    prompt: str = (
        "Fix the failing tests in test_client.py. "
        "Inspect config.py and client.py. "
        "Ensure ServerConfig supports a custom `port: int = 8080` and `timeout_seconds: int = 30`, "
        "and update APIClient in client.py to construct `base_url` using `http://{config.host}:{config.port}/api/v1` "
        "and pass `config.timeout_seconds` to requests. "
        "Run pytest to verify all tests pass, and report completion."
    )
    target_files: list[str] = field(default_factory=lambda: ["config.py", "client.py"])
    test_file: str = "test_client.py"

    initial_files: dict[str, str] = field(
        default_factory=lambda: {
            "config.py": (
                "from dataclasses import dataclass\n\n\n"
                "@dataclass\n"
                "class ServerConfig:\n"
                "    host: str\n"
                "    # Missing port and timeout fields expected by client\n"
            ),
            "client.py": (
                "from config import ServerConfig\n\n\n"
                "class APIClient:\n"
                "    def __init__(self, config: ServerConfig) -> None:\n"
                "        self.config = config\n"
                "        # Hardcoded port instead of using config.port\n"
                '        self.base_url = f"http://{config.host}:8000/api/v1"\n'
                "        self.timeout = 10\n\n"
                "    def get_endpoint(self, path: str) -> str:\n"
                '        return f"{self.base_url}/{path.lstrip(\'/\')}"\n'
            ),
            "test_client.py": (
                "from config import ServerConfig\n"
                "from client import APIClient\n\n\n"
                "def test_default_config_client():\n"
                '    cfg = ServerConfig(host="localhost", port=8080, timeout_seconds=30)\n'
                "    client = APIClient(cfg)\n"
                '    assert client.base_url == "http://localhost:8080/api/v1"\n'
                "    assert client.timeout == 30\n"
                '    assert client.get_endpoint("users") == "http://localhost:8080/api/v1/users"\n\n\n'
                "def test_custom_port():\n"
                '    cfg = ServerConfig(host="api.service.internal", port=9443, timeout_seconds=60)\n'
                "    client = APIClient(cfg)\n"
                '    assert client.base_url == "http://api.service.internal:9443/api/v1"\n'
                "    assert client.timeout == 60\n"
            ),
            "pyproject.toml": (
                "[project]\n"
                'name = "service-client"\n'
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
        """Validates that both config.py and client.py were correctly updated."""
        if not sandbox.file_exists("config.py"):
            return False, "config.py is missing from sandbox."
        if not sandbox.file_exists("client.py"):
            return False, "client.py is missing from sandbox."

        cfg_content = sandbox.read_file("config.py")
        client_content = sandbox.read_file("client.py")

        if "port" not in cfg_content or "timeout_seconds" not in cfg_content:
            return False, "config.py missing required port / timeout_seconds fields."
        if "{config.port}" not in client_content and "config.port" not in client_content:
            return False, "client.py not using config.port in base_url."

        return True, None
