"""
Unit tests verifying that the chat CLI command can be decoupled from local
llama-server startup via --no-server or the ULTRON_NO_SERVER environment variable.
"""

from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from ultron.main import app

runner = CliRunner()


def test_chat_no_server_flag_skips_server_start():
    with patch("ultron.main.LlamaServerManager") as mock_mgr_cls, patch(
        "ultron.main.async_chat", new_callable=AsyncMock
    ) as mock_async_chat:
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.check_endpoint_occupied.return_value = False

        result = runner.invoke(app, ["chat", "--no-server"])

        assert result.exit_code == 0
        mock_mgr.start.assert_not_called()
        mock_async_chat.assert_awaited_once_with(agent_type="simple")


def test_chat_env_var_skips_server_start(monkeypatch):
    monkeypatch.setenv("ULTRON_NO_SERVER", "1")

    with patch("ultron.main.LlamaServerManager") as mock_mgr_cls, patch(
        "ultron.main.async_chat", new_callable=AsyncMock
    ) as mock_async_chat:
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.check_endpoint_occupied.return_value = False

        result = runner.invoke(app, ["chat"])

        assert result.exit_code == 0
        mock_mgr.start.assert_not_called()
        mock_async_chat.assert_awaited_once_with(agent_type="simple")


def test_chat_default_attempts_server_start_when_unoccupied():
    with patch("ultron.main.LlamaServerManager") as mock_mgr_cls, patch(
        "ultron.main.async_chat", new_callable=AsyncMock
    ) as mock_async_chat:
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.check_endpoint_occupied.return_value = False

        result = runner.invoke(app, ["chat"])

        assert result.exit_code == 0
        mock_mgr.start.assert_called_once()
        mock_async_chat.assert_awaited_once_with(agent_type="simple")
        mock_mgr.stop.assert_called_once()
