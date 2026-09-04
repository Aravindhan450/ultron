from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings for Ultron, loaded from env variables and/or .env files.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ULTRON_",
        extra="ignore",
    )

    model: str = "gemini-2.5-flash"
    llama_cpp_base_url: str = "http://127.0.0.1:8080"
    timeout: float = 120.0
    llama_server_binary: str = "llama-server"
    llama_server_model_path: str | None = None
    llama_server_host: str = "127.0.0.1"
    llama_server_port: int = 8080
    llama_server_context_length: int = 32768
    llama_server_gpu_layers: int = 99
    llama_server_startup_timeout: float = 60.0
    llama_server_shutdown_timeout: float = 5.0
    wake_word: str = "ultron"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".ultron")
    memory_backend: Literal["sqlite", "json", "in-memory"] = "sqlite"
    security_mode: Literal["strict", "permissive", "interactive"] = "interactive"
    database_type: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str | None = None


# Global settings singleton
settings = Settings()


def update_env_file(key: str, value: str) -> None:
    """
    Updates or creates a key=value pair in the project's .env file on disk
    so configuration changes persist across CLI session restarts.
    """
    env_path = Path.cwd() / ".env"
    key_prefix = f"{key}="

    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if line.strip().startswith(key_prefix):
                new_lines.append(f"{key}={value}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"{key}={value}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(f"{key}={value}\n", encoding="utf-8")


_update_env_file = update_env_file
