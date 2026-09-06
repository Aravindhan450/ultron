"""
Unit tests verifying that the chat CLI command can be decoupled from local
llama-server startup via --no-server or the ULTRON_NO_SERVER environment variable.
"""

from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from ultron.main import app

runner = CliRunner()


def test_chat_no_server_flag_skips_server_start():
    with patch("ultron.main.async_chat", new_callable=AsyncMock) as mock_async_chat:
        result = runner.invoke(app, ["chat", "--no-server"])
        assert result.exit_code == 0
        mock_async_chat.assert_awaited_once_with(agent_type="simple", no_server=True)


def test_chat_env_var_skips_server_start(monkeypatch):
    monkeypatch.setenv("ULTRON_NO_SERVER", "1")

    with patch("ultron.main.async_chat", new_callable=AsyncMock) as mock_async_chat:
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        mock_async_chat.assert_awaited_once_with(agent_type="simple", no_server=True)


def test_chat_default_attempts_server_start_when_unoccupied():
    with patch("ultron.main.async_chat", new_callable=AsyncMock) as mock_async_chat:
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        mock_async_chat.assert_awaited_once_with(agent_type="simple", no_server=False)
