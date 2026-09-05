"""
Pytest configuration and fixtures for Model-in-the-Loop validation suite.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from tests.model_in_loop.harness.sandbox import MITLSandbox
from tests.model_in_loop.scenarios.bug_fix import CalculatorBugFixScenario
from ultron.core.engine.llama_cpp import LlamaCppEngine
from ultron.core.engine.server import LlamaServerManager


@pytest.fixture(scope="session")
def mitl_server() -> Generator[LlamaCppEngine, None, None]:
    """
    Manages local llama-server lifecycle for Model-in-the-Loop tests.
    Reuses existing running llama-server if active, or starts a dedicated local instance.
    """
    server_manager = LlamaServerManager()

    # Verify binary and model prerequisites
    try:
        server_manager.resolve_binary()
        server_manager.validate_model_path()
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(
            f"MITL prerequisites unavailable: {exc}. "
            "Please ensure llama-server is installed and ULTRON_LLAMA_SERVER_MODEL_PATH is configured."
        )

    server_started = False
    try:
        if not server_manager.check_endpoint_occupied():
            try:
                server_manager.start()
                server_started = True
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"Failed to start local llama-server for MITL: {exc}")

        engine = LlamaCppEngine(
            base_url=server_manager.endpoint_url,
            default_model=str(Path(server_manager.model_path).name) if server_manager.model_path else "default",
            timeout=120.0,
        )
        yield engine
    finally:
        if server_started:
            server_manager.stop()


@pytest.fixture
def mitl_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MITLSandbox:
    """
    Creates an isolated sandbox containing the CalculatorBugFixScenario repository.
    Strictly confines all tool operations to the sandbox.
    """
    scenario = CalculatorBugFixScenario()
    sandbox_dir = tmp_path / "sandbox_calculator"
    sandbox = MITLSandbox(root_dir=sandbox_dir, files=scenario.initial_files)
    sandbox.apply_security_boundary(monkeypatch)
    return sandbox
