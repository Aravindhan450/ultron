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
    wake_word: str = "ultron"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".ultron")
    memory_backend: Literal["sqlite", "json", "in-memory"] = "sqlite"
    security_mode: Literal["strict", "permissive", "interactive"] = "interactive"

# Global settings singleton
settings = Settings()
